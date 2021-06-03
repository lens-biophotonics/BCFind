import os
import json
import argparse
import numpy as np
import pandas as pd
import functools as ft
import tensorflow as tf

from skimage import io

from bcfind.config_manager import Configuration
from bcfind.utils import preprocessing, metrics
from bcfind.train import build_unet, sigmoid
from bcfind.blob_dog import BlobDoG


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, prog="evaluate_train.py")
    parser.add_argument(
        "config",
        type=str,
        help="Path to .yaml file containing the needed configuration settings.",
    )

    args = parser.parse_args()
    return args


def predict_file(
    unet,
    dog,
    img,
    save_pred=False,
    outfile=None,
):
    img_shape = img.shape
    img = img[tf.newaxis, ..., tf.newaxis].astype("float32")

    if outfile is not None:
        file_name = outfile.split("/")[-1].split(".")[0]
        print(f"UNet prediction on file {file_name}...")
    else:
        print("UNet prediction...")
    nn_pred = sigmoid(np.array(unet(img, training=False)).reshape(img_shape)) * 255

    if outfile is not None:
        print(f"DoG prediction on file {file_name}...")
    else:
        print("DoG prediction...")
    pred_centers = dog.predict(nn_pred)

    if save_pred:
        outdir = os.path.dirname(outfile)
        os.makedirs(outdir, exist_ok=True)
        np.save(outfile, pred_centers)

    return pred_centers


def evaluate_prediction(dog, predicted, gt_file_path):
    true = pd.read_csv(open(gt_file_path, "r"))
    true = true[["#x", " y", " z"]].dropna(0)
    res = dog.evaluate(predicted, true)
    return res


def main():
    args = parse_args()
    opts = Configuration(args.config)

    # Build UNet and load weights
    unet = build_unet(
        opts.exp.n_filters,
        opts.exp.k_size,
        opts.exp.k_stride,
        opts.data.data_shape,
        opts.exp.learning_rate,
    )
    unet.load_weights(f"{opts.exp.unet_checkpoint_dir}/model.h5")

    # Build DoG with trained parameters
    dog = BlobDoG(len(opts.data.data_shape), opts.data.dim_resolution)
    try:
        dog_par = json.load(open(f"{opts.exp.dog_checkpoint_dir}/parameters.json"))
        dog.set_parameters(dog_par)
    except FileNotFoundError:
        print("No optimal DoG parameters found, assigning default values.")
        pass

    # Define callable preprocessing function
    preprocessing_fun = ft.partial(
        preprocessing,
        transpose=opts.preproc.transpose,
        flip_axis=opts.preproc.flip_axis,
        clip_threshold=opts.preproc.clip_threshold,
        gamma_correction=opts.preproc.gamma_correction,
        downscale_factors=opts.preproc.downscale_factors,
        pad_output_shape=opts.data.data_shape,
    )

    # Predict and evaluate on train-set
    print(f"{opts.exp.name}: BCFind predictions on {opts.data.name} train-set.")
    train_files = [
        f
        for f in os.listdir(opts.data.train_tif_dir)
        if f.endswith(".tif") or f.endswith(".tiff")
    ]

    train_res = []
    for f_name in train_files:
        img = io.imread(f"{opts.data.train_tif_dir}/{f_name}")
        img = preprocessing_fun(img)
        out_f_name = f_name.split(".")[0]

        bcfind_pred = predict_file(
            unet,
            dog,
            img,
            True,
            f"{opts.exp.predictions_dir}/Pred_centers/{out_f_name}.npy",
        )

        print(f"Evaluating BCFind predictions on file {f_name}")
        bcfind_res = evaluate_prediction(
            dog, bcfind_pred, f"{opts.data.train_gt_dir}/{f_name}.marker"
        )
        train_res.append(bcfind_res)

    train_res = pd.concat(train_res)
    train_res.index = train_files
    train_res.to_csv(f"{opts.exp.predictions_dir}/train_eval.csv")

    perf = metrics(train_res)
    print(f"{opts.exp.name}: Train-set of {opts.data.name} evaluated with {perf}")
    print("")

    # Predict and evaluate on test-set
    print(f"{opts.exp.name}: BCFind predictions on {opts.data.name} test-set.")
    test_files = [
        f
        for f in os.listdir(opts.data.test_tif_dir)
        if f.endswith(".tif") or f.endswith(".tiff")
    ]

    test_res = []
    for f_name in test_files:
        img = io.imread(f"{opts.data.test_tif_dir}/{f_name}")
        img = preprocessing_fun(img)
        out_f_name = f_name.split(".")[0]

        bcfind_pred = predict_file(
            unet,
            dog,
            img,
            True,
            f"{opts.exp.predictions_dir}/Pred_centers/{out_f_name}.npy",
        )

        print("Evaluating BCFind test-set predictions")
        bcfind_res = evaluate_prediction(
            dog, bcfind_pred, f"{opts.data.test_gt_dir}/{f_name}.marker"
        )
        test_res.append(bcfind_res)

    test_res = pd.concat(test_res)
    test_res.index = test_files
    test_res.to_csv(f"{opts.exp.predictions_dir}/test_eval.csv")

    perf = metrics(test_res)
    print(f"{opts.exp.name}: Test-set of {opts.data.name} evaluated with {perf}")


if __name__ == "__main__":
    main()

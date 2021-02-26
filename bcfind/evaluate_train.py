import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from config_manager import Configuration
from utils import get_substack, metrics
from train import build_unet, sigmoid
from blob_dog import BlobDoG


def predict_file(
    unet,
    dog,
    tif_path,
    data_shape,
    save_pred=False,
    outdir=None,
    transpose=None,
    flip_axis=None,
    clip_threshold=None,
    gamma_correction=None,
    downscale_factors=None,
):
    tif_name = tif_path.split("/")[-1]

    img = get_substack(
        tif_path,
        data_shape,
        transpose,
        flip_axis,
        clip_threshold,
        gamma_correction,
        downscale_factors,
    )
    img = img[tf.newaxis, ..., tf.newaxis].astype("float32")

    print(f"Unet prediction on file {tif_name}")
    nn_pred = sigmoid(np.array(unet(img, training=False)).reshape(data_shape)) * 255

    print(f"DoG prediction on file {tif_name}")
    pred_centers = dog.predict(nn_pred)

    if save_pred:
        os.makedirs(outdir, exist_ok=True)
        f_name = tif_name.split(".")[0]
        np.save(f"{outdir}/{f_name}.npy", pred_centers)

    return pred_centers


def evaluate_prediction(dog, predicted, gt_file_path):
    true = pd.read_csv(open(gt_file_path, "r"))
    true = true[["#x", " y", " z"]].dropna(0)
    res = dog.evaluate(predicted, true)
    return res


def main(args):
    opts = Configuration(args.config)

    # Build UNet and load weights
    unet = build_unet(
        opts.exp.n_filters,
        opts.exp.e_size,
        opts.exp.e_stride,
        opts.exp.d_size,
        opts.exp.d_stride,
        opts.data.data_shape,
        opts.exp.learning_rate,
    )
    unet.load_weights(f"{opts.exp.unet_checkpoint_dir}/model.h5")

    # Build DoG with trained parameters
    dog = BlobDoG(len(opts.data.data_shape), opts.data.dim_resolution)
    try:
        with open(f"{opts.exp.dog_checkpoint_dir}/parameters.json") as f:
            dog_par = json.load(f)
        dog.set_parameters(dog_par)
    except FileNotFoundError:
        print("No optimal DoG parameters found, assigning default values.")
        pass

    # Predict and evaluate on train-set
    print(f"{opts.exp.name}: BCFind predictions on {opts.data.name} train-set.")
    train_files = [
        f
        for f in os.listdir(opts.data.train_tif_dir)
        if f.endswith(".tif") or f.endswith(".tiff")
    ]

    train_res = []
    for f_name in train_files:
        bcfind_pred = predict_file(
            unet,
            dog,
            f"{opts.data.train_tif_dir}/{f_name}",
            opts.data.data_shape,
            True,
            f"{opts.exp.predictions_dir}/Pred_centers",
            opts.preproc.transpose,
            opts.preproc.flip_axis,
            opts.preproc.clip_threshold,
            opts.preproc.gamma_correction,
            opts.preproc.downscale_factors,
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
        bcfind_pred = predict_file(
            unet,
            dog,
            f"{opts.data.test_tif_dir}/{f_name}",
            opts.data.data_shape,
            True,
            f"{opts.exp.predictions_dir}/Pred_centers",
            opts.preproc.transpose,
            opts.preproc.flip_axis,
            opts.preproc.clip_threshold,
            opts.preproc.gamma_correction,
            opts.preproc.downscale_factors,
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


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, prog="train.py")
    parser.add_argument("config", type=str, help="Configuration file")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)

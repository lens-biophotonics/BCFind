import os
import h5py
import json
import argparse
import numpy as np
import pandas as pd
import functools as ft
import tensorflow as tf
import concurrent.futures as cf

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


def localize_and_evaluate(dog, x, y_file, max_match_dist, outdir):
    f_name = y_file.split("/")[-1].split(".")[0]
    y = pd.read_csv(open(y_file, "r"))[["#x", " y", " z"]]

    evaluation = dog.predict_and_evaluate(x, y, max_match_dist)
    evaluation.to_csv(f"{outdir}/Pred_centers/pred_{y_file}")

    TP = np.sum(evaluation.label == "TP")
    FP = np.sum(evaluation.label == "FP")
    FN = np.sum(evaluation.label == "FN")

    counts_df = pd.DataFrame([f_name, TP, FP, FN, TP + FP, y.shape[0]]).T
    counts_df.columns = ["file", "TP", "FP", "FN", "tot_pred", "tot_true"]
    return counts_df


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

    f_names = np.load(f"{opts.exp.h5_dir}/file_names.npy")
    with h5py.File(f"{opts.exp.h5_dir}/X_train.h5", "r") as fx:
        X_train = fx["x"][()]

    print(f"{opts.exp.name}: UNet predictions on {opts.data.name} train-set")
    X_emb = unet.predict(X_train, batch_size=4)
    X_emb = sigmoid(np.reshape(X_emb, X_train.shape)) * 255

    print(
        f"{opts.exp.name}: DoG predictions and evaluation on {opts.data.name} train-set"
    )
    n_cpu = 10
    with cf.ThreadPoolExecutor(n_cpu) as pool:
        futures = [
            pool.submit(
                localize_and_evaluate(
                    x,
                    f"{opts.data.train_gt_dir}/{f_name}.marker",
                    10,
                    opts.exp.predictions_dir,
                )
                for x, f_name in zip(X_train, f_names)
            )
        ]
        res = [future.result() for future in cf.as_completed(futures)]

    res = pd.concat(res)
    res.to_csv(f"{opts.exp.predictions_dir}/train_eval.csv")
    perf = metrics(res)

    print(f"{opts.exp.name}: Train-set of {opts.data.name} evaluated with {perf}")
    print("")
    del X_train

    # Predict and evaluate on test-set
    print(f"{opts.exp.name}: creating .h5 file for {opts.data.name} test-set.")
    test_files = [
        f
        for f in os.listdir(opts.data.test_tif_dir)
        if f.endswith(".tif") or f.endswith(".tiff")
    ]

    fx = h5py.File(f"{opts.exp.h5_dir}/X_test.h5", "w")
    fx.create_dataset(
        "x", data=np.zeros((len(test_files), *opts.data.data_shape)), dtype=np.uint8
    )

    for i, f_name in enumerate(test_files):
        img = io.imread(f"{opts.data.test_tif_dir}/{f_name}")
        img = preprocessing_fun(img)

        fx["x"][i, ...] = (img * 255).astype(np.uint8)

    X_test = fx["x"][()]
    fx.close()

    print(f"{opts.exp.name}: UNet predictions on {opts.data.name} test-set")
    X_emb = unet.predict(X_test, batch_size=4)
    X_emb = sigmoid(np.reshape(X_emb, X_test.shape)) * 255

    print(
        f"{opts.exp.name}: DoG predictions and evaluation on {opts.data.name} test-set"
    )
    n_cpu = 10
    with cf.ThreadPoolExecutor(n_cpu) as pool:
        futures = [
            pool.submit(
                localize_and_evaluate(
                    x,
                    f"{opts.data.train_gt_dir}/{f_name}.marker",
                    10,
                    opts.exp.predictions_dir,
                )
                for x, f_name in zip(X_test, test_files)
            )
        ]
        res = [future.result() for future in cf.as_completed(futures)]

    res = pd.concat(res)
    res.to_csv(f"{opts.exp.predictions_dir}/test_eval.csv")
    perf = metrics(res)

    print(f"{opts.exp.name}: Test-set of {opts.data.name} evaluated with {perf}")
    print("")
    del X_test


if __name__ == "__main__":
    main()

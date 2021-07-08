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


def localize_and_evaluate(dog, x, y_file, max_match_dist, outdir=None):
    f_name = y_file.split("/")[-1].split(".")[0]
    y = pd.read_csv(open(y_file, "r"))[["#x", " y", " z"]]

    evaluation = dog.predict_and_evaluate(x, y, max_match_dist)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(f"{outdir}/Pred_centers", exist_ok=True)
        evaluation.to_csv(f"{outdir}/Pred_centers/pred_{f_name}.marker")

    TP = np.sum(evaluation.label == "TP")
    FP = np.sum(evaluation.label == "FP")
    FN = np.sum(evaluation.label == "FN")

    counts_df = pd.DataFrame([f_name, TP, FP, FN, TP + FP, y.shape[0]]).T
    counts_df.columns = ["file", "TP", "FP", "FN", "tot_pred", "tot_true"]
    return counts_df


def main():
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args = parse_args()
    opts = Configuration(args.config)
    BATCH_SIZE = 2

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
    f_names = np.load(f"{opts.exp.h5_dir}/file_names.npy")
    with h5py.File(f"{opts.exp.h5_dir}/X_train.h5", "r") as fx:
        X_train = fx["x"][()].astype("float32")
    data = (
        tf.data.Dataset.from_tensor_slices(X_train[..., np.newaxis] / 255)
        .batch(BATCH_SIZE)
        .prefetch(2)
    )

    del X_train

    print(f"{opts.exp.name}: UNet predictions on {opts.data.name} train-set")
    X_emb = []
    for batch in data:
        pred_batch = unet.predict(batch)
        pred_batch = sigmoid(pred_batch) * 255
        for pred in pred_batch:
            X_emb.append(pred.reshape(opts.data.data_shape))

    print(
        f"{opts.exp.name}: DoG predictions and evaluation on {opts.data.name} train-set"
    )
    n_cpu = 10
    with cf.ThreadPoolExecutor(n_cpu) as pool:
        futures = [
            pool.submit(
                localize_and_evaluate,
                dog,
                x,
                f"{opts.data.train_gt_dir}/{f_name}.marker",
                10,
                opts.exp.predictions_dir,
            )
            for x, f_name in zip(X_emb, f_names)
        ]
        res = [future.result() for future in cf.as_completed(futures)]

    res = pd.concat(res)
    res.to_csv(f"{opts.exp.predictions_dir}/train_eval.csv")
    perf = metrics(res)

    print(f"{opts.exp.name}: Train-set of {opts.data.name} evaluated with {perf}")
    print("")

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

    X_test = fx["x"][()].astype("float32")
    fx.close()
    data = (
        tf.data.Dataset.from_tensor_slices(X_test[..., np.newaxis] / 255)
        .batch(BATCH_SIZE)
        .prefetch(2)
    )
    del X_test

    print(f"{opts.exp.name}: UNet predictions on {opts.data.name} test-set")
    X_emb = []
    for batch in data:
        pred_batch = unet.predict(batch)
        pred_batch = sigmoid(pred_batch) * 255
        for pred in pred_batch:
            X_emb.append(pred.reshape(opts.data.data_shape))

    print(
        f"{opts.exp.name}: DoG predictions and evaluation on {opts.data.name} test-set"
    )
    n_cpu = 10
    with cf.ThreadPoolExecutor(n_cpu) as pool:
        futures = [
            pool.submit(
                localize_and_evaluate,
                dog,
                x,
                f"{opts.data.test_gt_dir}/{f_name}.marker",
                10,
                opts.exp.predictions_dir,
            )
            for x, f_name in zip(X_emb, test_files)
        ]
        res = [future.result() for future in cf.as_completed(futures)]

    res = pd.concat(res)
    res.to_csv(f"{opts.exp.predictions_dir}/test_eval.csv")
    perf = metrics(res)

    print(f"{opts.exp.name}: Test-set of {opts.data.name} evaluated with {perf}")
    print("")


if __name__ == "__main__":
    main()

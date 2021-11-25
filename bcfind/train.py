import os
import h5py
import json
import shutil
import argparse
import numba as nb
import numpy as np
import pandas as pd
import tensorflow as tf

from bcfind.config_manager import Configuration
from bcfind.data_generator import get_tf_data, Scaler
from bcfind.losses import FramedCrossentropy3D
from bcfind.blob_dog import BlobDoG
from bcfind.utils import sigmoid
from bcfind.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, prog="train.py")
    parser.add_argument(
        "config",
        type=str,
        help="Path to .yaml file containing the needed configuration settings.",
    )
    args = parser.parse_args()
    return args


def build_unet(n_filters, k_size, k_stride, input_shape, learning_rate, exclude_border):
    model = UNet(
        n_filters,
        k_size,
        k_stride,
    )
    model.build((None, *input_shape, 1))

    if exclude_border is not None:
        loss = FramedCrossentropy3D(exclude_border, input_shape, from_logits=True)
    else:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    metrics = [tf.metrics.BinaryAccuracy(threshold=0.2)]

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )
    return model


def get_callbacks(checkpoint_dir, tensorboard_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir, ignore_errors=True)

    MC_callback = tf.keras.callbacks.ModelCheckpoint(
        f"{checkpoint_dir}/model.h5",
        save_best_only=True,
        save_weights_only=True,
        save_freq="epoch",
        monitor="loss",
        mode="min",
        verbose=1,
    )

    TB_callback = tf.keras.callbacks.TensorBoard(
        tensorboard_dir,
        update_freq="epoch",
        profile_batch=0,
    )
    return [MC_callback, TB_callback]


def fit_unet(
    unet,
    train_data,
    val_data,
    epochs,
    checkpoint_dir,
    tensorboard_dir,
):
    callbacks = get_callbacks(
        checkpoint_dir,
        tensorboard_dir,
    )

    unet.fit(
        train_data,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_data,
        verbose=1,
    )

    return unet


def fit_dog(
    X,
    Y,
    dim_resolution,
    exclude_border,
    max_match_dist,
    iterations=30,
    logs_dir=None,
    checkpoint_dir=None,
):
    print(checkpoint_dir)
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir, ignore_errors=True)
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    dog = BlobDoG(len(X[0].shape), dim_resolution, exclude_border)
    dog.fit(
        X,
        Y,
        max_match_dist,
        n_iter=iterations,
        logs_dir=logs_dir,
        checkpoint_dir=checkpoint_dir,
        n_cpu=10,
        verbose=1,
    )
    return dog


def main():
    args = parse_args()
    conf = Configuration(args.config)

    x_file = f"{conf.exp.h5_dir}/X_train.h5"
    y_file = f"{conf.exp.h5_dir}/Y_train.h5"
    pred_file = f"{conf.exp.h5_dir}/Y_hat_train.h5"

    # Unet training
    print("Loading data")
    with h5py.File(x_file, "r") as fx:
        n = fx["x"].shape[0]
        X = fx["x"][()]

    with h5py.File(y_file, "r") as fy:
        assert n == fy["y"].shape[0], "Inputs and targets do not have the same length"
        Y = fy["y"][()]

    scaler = Scaler(conf.preproc.normalization, conf.preproc.standardization)
    X = scaler.fit(X)

    print("Building data generator")
    data = get_tf_data(
        X,
        Y,
        conf.unet.batch_size,
        conf.unet.input_shape,
        conf.unet.val_fold,
        conf.unet.val_seed,
        conf.data_aug.augment,
        conf.data_aug.deform,
        conf.data_aug.brightness,
        conf.data_aug.gamma,
        conf.data_aug.contrast,
        conf.data_aug.zoom,
    )

    if isinstance(conf.unet.val_fold, str):
        train, val = data
    elif conf.unet.val_fold is None:
        train = data
        val = None

    print("Building and training U-Net model")
    unet = build_unet(
        conf.unet.n_filters,
        conf.unet.k_size,
        conf.unet.k_stride,
        conf.unet.input_shape,
        conf.unet.learning_rate,
        conf.unet.exclude_border,
    )
    unet.summary()

    unet = fit_unet(
        train,
        val,
        conf.unet.epochs,
        conf.unet.checkpoint_dir,
        conf.unet.tensorboard_dir,
    )

    # DoG training
    print("Preparing data for DoG training")
    f_names = np.load(f"{conf.exp.h5_dir}/train_files.npy")

    Y = []
    for name in f_names:
        print(f"Loading file {conf.data.train_gt_dir}/{name}.marker")
        y = pd.read_csv(open(f"{conf.data.train_gt_dir}/{name}.marker", "r"))
        y = y[conf.data.marker_columns].dropna(0)
        Y.append(y)

    print(f"Saving U-Net predictions in {pred_file}")
    fx = h5py.File(pred_file, "w")
    fx.create_dataset(name="y_hat", data=np.zeros(X.shape), dtype="float32")

    for i, x in enumerate(X):
        print(f"Unet prediction on file {i+1}/{X.shape[0]}")
        x = x[np.newaxis, ..., np.newaxis].astype("float32")

        pred = sigmoid(unet.predict(x).reshape(conf.data.data_shape))
        fx["y_hat"][i, ...] = (pred * 255).astype("float32")

    fx.close()

    with h5py.File(pred_file, "r") as fx:
        X = fx["y_hat"][()]

    nb.cuda.close()

    dog = fit_dog(
        X,
        Y,
        conf.data.dim_resolution,
        conf.dog.exclude_border,
        conf.dog.max_match_dist,
        conf.dog.iterations,
        conf.dog.checkpoint_dir,
        conf.dog.n_cpu,
    )

    dog_par = dog.get_parameters()
    print(f"Best parameters found for DoG: {dog_par}")

    os.makedirs(conf.dog.checkpoint_dir, exist_ok=True)
    with open(f"{conf.dog.checkpoint_dir}/parameters.json", "w") as outfile:
        json.dump(dog_par, outfile)


if __name__ == "__main__":
    main()

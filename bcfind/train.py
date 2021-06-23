import os
import h5py
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from bcfind.data_generator import get_tf_data
from bcfind.utils import sigmoid
from bcfind.config_manager import Configuration
from bcfind.unet import UNet
from bcfind.blob_dog import BlobDoG


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, prog="train.py")
    parser.add_argument(
        "config",
        type=str,
        help="Path to .yaml file containing the needed configuration settings.",
    )
    args = parser.parse_args()
    return args


def build_unet(n_filters, k_size, k_stride, input_shape, learning_rate):
    model = UNet(
        n_filters,
        k_size,
        k_stride,
    )
    model.build((None, *input_shape, 1))

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    metrics = ["accuracy", "mse"]

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )
    return model


def get_callbacks(checkpoint_dir, tensorboard_dir, check_every):
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
        update_freq=check_every,
        write_graph=False,
    )
    return [MC_callback, TB_callback]


def unet_fit(
    x_file,
    y_file,
    batch_size,
    input_shape,
    epochs,
    learning_rate,
    n_filters,
    k_size,
    k_stride,
    checkpoint_dir,
    tensorboard_dir,
    check_every,
):
    print("Preparing data and building model for U-Net training")
    data = get_tf_data(x_file, y_file, batch_size, input_shape)
    callbacks = get_callbacks(
        checkpoint_dir,
        tensorboard_dir,
        check_every,
    )

    unet = build_unet(
        n_filters,
        k_size,
        k_stride,
        input_shape,
        learning_rate,
    )
    unet.summary()

    print("U-Net training")
    unet.fit(
        data,
        epochs=epochs,
        callbacks=callbacks,
    )

    return unet


def dog_fit(
    unet,
    train_data_dir,
    train_gt_dir,
    dim_resolution,
    max_match_dist,
    iterations=30,
    logs_dir=None,
    checkpoint_dir=None,
):
    print("Preparing data for DoG training")
    Y = []
    f_names = np.load(f"{train_data_dir}/file_names.npy")
    with h5py.File(f"{train_data_dir}/X_train.h5", "r") as fx:
        X = fx["x"][()]
    for name in f_names:
        print(f"Loading file {train_gt_dir}/{name}.marker")
        y = pd.read_csv(open(f"{train_gt_dir}/{name}.marker", "r"))
        y = y[["#x", " y", " z"]].dropna(0)
        Y.append(y)

    if not os.path.exists(f"{train_data_dir}/unet_pred.h5"):
        print(f"Saving U-Net predictions in {train_data_dir}/unet_pred.h5")
        X_emb = h5py.File(f"{train_data_dir}/unet_pred.h5", "w")
        X_emb.create_dataset(name="unet", data=np.zeros(X.shape), dtype=np.uint8)
        for i, x in enumerate(X):
            print(f"Unet prediction on file {i+1}/{X.shape[0]}")
            data_shape = x.shape
            x = x[np.newaxis, ..., np.newaxis].astype("float32")
            pred = sigmoid(np.array(unet(x, training=True)).reshape(data_shape))
            X_emb["unet"][i, ...] = (pred * 255).astype(np.uint8)
        X_emb.close()

    if os.path.exists(f"{train_data_dir}/unet_pred.h5"):
        print("Loading unet predictions {train_data_dir}/unet_pred.h5")
        with h5py.File(f"{train_data_dir}/unet_pred.h5", "r") as fx:
            X_emb = fx["unet"][()]

    print("Training DoG")
    print(checkpoint_dir)
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir, ignore_errors=True)
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    dog = BlobDoG(Y[0].shape[1], dim_resolution)
    dog.fit(
        X_emb,
        Y,
        max_match_dist,
        n_iter=iterations,
        logs_dir=logs_dir,
        checkpoint_dir=checkpoint_dir,
        n_cpu=10,
        n_gpu=1,
        verbose=1,
    )
    return dog


def main():
    args = parse_args()
    conf = Configuration(args.config)

    x_file = f"{conf.exp.train_data_dir}/X_train.h5"
    y_file = f"{conf.exp.train_data_dir}/Y_train.h5"

    # Unet training
    unet = unet_fit(
        x_file,
        y_file,
        conf.exp.batch_size,
        conf.exp.input_shape,
        conf.exp.unet_epochs,
        conf.exp.learning_rate,
        conf.exp.n_filters,
        conf.exp.k_size,
        conf.exp.k_stride,
        conf.exp.unet_checkpoint_dir,
        conf.exp.unet_tensorboard_dir,
        conf.exp.check_every,
    )

    # DoG training
    dog = dog_fit(
        unet,
        conf.exp.train_data_dir,
        conf.data.train_gt_dir,
        conf.data.dim_resolution,
        10.0,  # FIXME: not yet in .yaml configuration file
        conf.exp.dog_iterations,
        conf.exp.dog_logs_dir,
        conf.exp.dog_checkpoint_dir,
    )

    dog_par = dog.get_parameters()
    print(f"Best parameters found for DoG: {dog_par}")

    os.makedirs(conf.exp.dog_checkpoint_dir, exist_ok=True)
    with open(f"{conf.exp.dog_checkpoint_dir}/parameters.json", "w") as outfile:
        json.dump(dog_par, outfile)


if __name__ == "__main__":
    main()

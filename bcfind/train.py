import os
import h5py
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from bcfind.config_manager import Configuration
from bcfind.data_generator import get_tf_data
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
    metrics = ["accuracy"]

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
        update_freq="epoch",
        profile_batch=0,
    )
    return [MC_callback, TB_callback]


def unet_fit(
    x_file,
    y_file,
    batch_size,
    input_shape,
    val_fold,
    epochs,
    learning_rate,
    exclude_border,
    n_filters,
    k_size,
    k_stride,
    checkpoint_dir,
    tensorboard_dir,
    check_every,
):
    print("Building U-Net model")
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
        exclude_border,
    )
    unet.summary()

    if val_fold is not None:
        print("Loading training and validation data")
        train, val = get_tf_data(x_file, y_file, batch_size, input_shape, val_fold)

        print("U-Net model fitting")
        unet.fit(
            train,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val,
            verbose=1,
            # workers=15,
        )
    else:
        print("Loading training data")
        train = get_tf_data(x_file, y_file, batch_size, input_shape)

        print("U-Net model fitting")
        unet.fit(
            train,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
    return unet


def dog_fit(
    unet,
    train_data_dir,
    train_gt_dir,
    dim_resolution,
    exclude_border,
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
    data_shape = X[0].shape

    for name in f_names:
        print(f"Loading file {train_gt_dir}/{name}.marker")
        y = pd.read_csv(open(f"{train_gt_dir}/{name}.marker", "r"))
        y = y[["#x", " y", " z"]].dropna(0)

        if exclude_border is not None:
            y = y.drop(y[y["#x"] <= exclude_border[0]].index)
            y = y.drop(y[y["#x"] >= data_shape[0] - exclude_border[0]].index)
            y = y.drop(y[y[" y"] <= exclude_border[1]].index)
            y = y.drop(y[y[" y"] >= data_shape[1] - exclude_border[1]].index)
            y = y.drop(y[y[" z"] <= exclude_border[2]].index)
            y = y.drop(y[y[" z"] >= data_shape[2] - exclude_border[2]].index)

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

    dog = BlobDoG(Y[0].shape[1], dim_resolution, exclude_border)
    dog.fit(
        X_emb,
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
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args = parse_args()
    conf = Configuration(args.config)

    x_file = f"{conf.exp.h5_dir}/X_train.h5"
    y_file = f"{conf.exp.h5_dir}/Y_train.h5"

    # Unet training
    unet = unet_fit(
        x_file,
        y_file,
        conf.exp.batch_size,
        conf.exp.input_shape,
        conf.exp.val_fold,
        conf.exp.unet_epochs,
        conf.exp.learning_rate,
        conf.exp.exclude_border,
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
        conf.exp.h5_dir,
        conf.data.train_gt_dir,
        conf.data.dim_resolution,
        conf.exp.exclude_border,
        conf.exp.max_match_dist,
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

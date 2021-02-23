import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from make_training_data import get_substack
from data_generator import get_tf_data
from config_manager import Configuration
from unet import UNet
from blob_dog import BlobDoG


def build_unet(
    n_filters, e_size, e_stride, d_size, d_stride, input_shape, learning_rate
):
    model = UNet(
        n_filters,
        e_size,
        e_stride,
        d_size,
        d_stride,
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(opts):
    x_file = f"{opts.data.files_h5_dir}/X_train.h5"
    y_file = f"{opts.data.files_h5_dir}/Y_train.h5"

    print("Preparing data and building model for U-Net training")
    data = get_tf_data(x_file, y_file, opts.exp.batch_size, opts.exp.input_shape)
    callbacks = get_callbacks(
        opts.exp.unet_checkpoint_dir,
        opts.exp.unet_tensorboard_dir,
        opts.exp.check_every,
    )

    unet = build_unet(
        opts.exp.n_filters,
        opts.exp.e_size,
        opts.exp.e_stride,
        opts.exp.d_size,
        opts.exp.d_stride,
        opts.exp.input_shape,
        opts.exp.learning_rate,
    )
    unet.summary()

    print("U-Net training")
    unet.fit(
        data,
        epochs=opts.exp.unet_epochs,
        callbacks=callbacks,
    )

    print("Preparing data for DoG training")
    file_names = [
        f
        for f in os.listdir(opts.data.train_tif_dir)
        if f.endswith(".tiff") or f.endswith(".tif")
    ]
    unet_pred = []
    Y_train = []
    for f_name in file_names:
        print(f"Loading file {f_name}")
        x = get_substack(
            f"{opts.data.train_tif_dir}/{f_name}",
            opts.data.data_shape,
            transpose=opts.preproc.transpose,
            flip_axis=opts.preproc.flip_axis,
            clip_threshold=opts.preproc.clip_threshold,
            gamma_correction=opts.preproc.gamma_correction,
            downscale_factors=opts.preproc.downscale_factors,
        )
        x = x[np.newaxis, ..., np.newaxis]

        print(f"Unet prediction on file {f_name}")
        unet_pred_i = (
            sigmoid(np.array(unet(x, training=False)).reshape(opts.data.data_shape))
            * 255
        )
        unet_pred.append(unet_pred_i)

        print(f"Loading .marker file for {f_name}")
        y = pd.read_csv(open(f"{opts.data.train_gt_dir}/{f_name}.marker", "r"))
        y = y[["#x", " y", " z"]].dropna(0)
        Y_train.append(y)

    print("Training DoG")
    dog_logs_dir = opts.exp.dog_logs_dir
    if os.path.exists(dog_logs_dir):
        shutil.rmtree(dog_logs_dir)

    dog = BlobDoG(Y_train[0].shape[1], opts.data.dim_resolution)
    dog.fit(
        unet_pred,
        Y_train,
        n_iter=opts.exp.dog_iterations,
        n_cpu=10,
        n_gpu=1,
        outdir=dog_logs_dir,
        verbose=1,
    )
    dog_par = dog.get_parameters()
    print(f"Best parameters found for DoG: {dog_par}")

    os.makedirs(opts.exp.dog_checkpoint_dir, exist_ok=True)
    with open(f"{opts.exp.dog_checkpoint_dir}/parameters.json", "w") as outfile:
        json.dump(dog_par, outfile)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, prog="train.py")
    parser.add_argument("config", type=str, help="Configuration file")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    config = Configuration(args.config)
    main(config)

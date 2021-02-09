import os
import shutil
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from make_training_data import get_substack
from data_generator import BatchGenerator
from config_manager import Configuration
from unet import UNet
from blob_dog import BlobDoG

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def prepare_unet_data(x_file, y_file, batch_size, input_shape):
    batch_gen = BatchGenerator(x_file, y_file, batch_size, input_shape)

    data = tf.data.Dataset.from_generator(
        lambda: batch_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, *input_shape, 1], [None, *input_shape, 1]),
    )
    data = data.prefetch(3)
    return data


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
    metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )
    return model


def get_callbacks(outdir, check_every):
    checkpoint_dir = f"{outdir}/UNet_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_dir = f"{outdir}/UNet_tensorboard"
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir, ignore_errors=True)

    MC_callback = tf.keras.callbacks.ModelCheckpoint(
        f"{checkpoint_dir}/model.h5", save_freq="epoch", monitor="loss"
    )

    TB_callback = tf.keras.callbacks.TensorBoard(
        tensorboard_dir,
        embeddings_freq=check_every,
        update_freq=check_every,
    )
    return [MC_callback, TB_callback]


def main(opts):
    x_file = f"{opts.data.files_h5_dir}/X_train.h5"
    y_file = f"{opts.data.files_h5_dir}/Y_train.h5"
    exp_outdir = f"{opts.exp.basepath}/{opts.exp.name}"

    print("Preparing data and building model for U-Net training")
    data = prepare_unet_data(x_file, y_file, opts.exp.batch_size, opts.exp.input_shape)
    callbacks = get_callbacks(exp_outdir, opts.exp.check_every)

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
        epochs=opts.exp.unet_iterations,
        callbacks=callbacks,
    )

    print("Preparing data for DoG training")
    unet_pred = []
    Y_train = []
    for f_name in os.listdir(opts.data.train_tif_dir):
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
        y = pd.read_csv(open(f"{opts.data.train_gt_dir}/{f_name}.marker", "r"))
        y = y[["#x", " y", " z"]].dropna(0)

        print(f"Unet prediction on file {f_name}")
        unet_pred_i = unet.predict(x).reshape(opts.data.data_shape) * 255
        unet_pred.append(unet_pred_i)

        Y_train.append(y)

    print("Training DoG")
    dog_logs_dir = f"{exp_outdir}/DoG_logs"
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
    dog_par = pd.DataFrame(dog.get_parameters())
    print(f"Best parameters found for DoG: {dog_par}")
    dog_checkpoint_dir = f"{exp_outdir}/DoG_checkpoint"
    dog_par.to_csv(dog_checkpoint_dir)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, prog="train.py")
    parser.add_argument("config", type=str, help="Configuration file")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    config = Configuration(args.config)
    main(config)

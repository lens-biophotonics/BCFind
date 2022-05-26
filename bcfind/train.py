import os
from typing import Iterator
import lmdb
import json
import pickle
import shutil
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from bcfind.config_manager import Configuration
from bcfind.models import UNet, SEUNet, ECAUNet
from bcfind.utils import sigmoid
from bcfind.blob_dog import BlobDoG
from bcfind.losses import FramedCrossentropy3D, FramedFocalCrossentropy3D
from bcfind.training_dataset import TrainingDataset, get_input_tf, normalize_tf


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="train.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("config", type=str, help="YAML Configuration file")
    return parser.parse_args()


def main():
    args = parse_args()
    conf = Configuration(args.config)

    ######################################
    ############ CREATE DIRS #############
    ######################################
    config_name = args.config.split('/')[-1]
    if not os.path.isdir(conf.exp.basepath):
        os.makedirs(conf.exp.basepath)
        shutil.copyfile(args.config, f'{conf.exp.basepath}/{config_name}')
    
    if not os.path.isdir(conf.unet.checkpoint_dir):
        os.makedirs(conf.unet.checkpoint_dir)
    
    if not os.path.isdir(conf.dog.checkpoint_dir):
        os.makedirs(conf.dog.checkpoint_dir)
    
    if os.path.isdir(conf.unet.tensorboard_dir):
        shutil.rmtree(conf.unet.tensorboard_dir)
        os.makedirs(conf.unet.tensorboard_dir)

    ####################################
    ############ UNET DATA #############
    ####################################
    print('\n LOADING UNET DATA')
    marker_list = [f'{conf.data.train_gt_dir}/{fname}' for fname in os.listdir(conf.data.train_gt_dir)]
    tiff_list = [f'{conf.data.train_tif_dir}/{fname}' for fname in os.listdir(conf.data.train_tif_dir)]

    ordered_tiff_list = []
    for f in marker_list:
        fname = Path(f).with_suffix('').name
        tiff_file = [f for f in map(lambda f: Path(f), tiff_list) if f.name == fname]
        ordered_tiff_list.append(str(tiff_file[0]))

    data = TrainingDataset(
        tiff_list=ordered_tiff_list, 
        marker_list=marker_list, 
        batch_size=conf.unet.batch_size, 
        dim_resolution=conf.data.dim_resolution, 
        output_shape=conf.unet.input_shape, 
        augmentations=conf.data_aug.op_args, 
        augmentations_prob=conf.data_aug.op_probs)

    ####################################
    ############## UNET ################
    ####################################
    print()
    print('\n BUILDING UNET')
    # loss = FramedFocalCrossentropy3D(exclude_border, input_shape, gamma=3, alpha=None, from_logits=True)
    loss = FramedCrossentropy3D(conf.unet.exclude_border, conf.unet.input_shape, from_logits=True)
    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(conf.unet.learning_rate)

    unet = UNet(
        input_shape=[None, None, None, 1], 
        n_filters=conf.unet.n_filters, 
        k_size=conf.unet.k_size, 
        k_stride=conf.unet.k_stride, 
        dropout=conf.unet.dropout, 
        regularizer=conf.unet.regularizer
        )
    unet.compile(loss=loss, optimizer=optimizer)
    unet.summary()

    MC_callback = tf.keras.callbacks.ModelCheckpoint(
        f"{conf.unet.checkpoint_dir}/model.tf",
        save_best_only=True,
        save_format='tf',
        save_freq="epoch",
        monitor="loss",
        mode="min",
        verbose=1,
    )

    TB_callback = tf.keras.callbacks.TensorBoard(
        conf.unet.tensorboard_dir,
        update_freq="epoch",
        profile_batch=0,
    )

    unet.fit(
        data,
        epochs=conf.unet.epochs,
        callbacks=[MC_callback, TB_callback],
        validation_data=None,
        verbose=1,
    )

    ####################################
    ############ LOAD UNET #############
    ####################################
    del unet

    # Load UNet and weights
    custom_objects = {
        'FramedFocalCrossentropy3D': FramedFocalCrossentropy3D,
        'FramedCrossentropy3D': FramedCrossentropy3D
    }
    unet = tf.keras.models.load_model(f"{conf.unet.checkpoint_dir}/model.tf", custom_objects=custom_objects)

    ####################################
    ############ DOG DATA ##############
    ####################################
    print("\n LOADING DoG DATA")

    Y = []
    for marker_file in marker_list:
        print(f"Loading file {marker_file}")
        y = pd.read_csv(open(marker_file, "r"))
        y = y[conf.data.marker_columns].dropna(0)
        Y.append(np.array(y))

    print(f"Saving U-Net predictions in {conf.exp.basepath}/Train_pred.h5")

    n = len(marker_list)
    nbytes = np.prod(conf.data.shape) * 4
    db = lmdb.open(f'{conf.exp.basepath}/Train_pred_lmdb', map_size=n*nbytes*10)

    with db.begin(write=True) as fx:
        for i, tiff_file in enumerate(ordered_tiff_list):
            print(f"Unet prediction on file {i+1}/{len(marker_list)}")
            
            x = get_input_tf(tf.constant(tiff_file))
            x = normalize_tf(x)        
            x = x[tf.newaxis, ..., tf.newaxis]

            max_attempts = 10
            attempts = 0
            while True:
                try:
                    pred = unet(x, training=False)
                    break
                except tf.errors.InvalidArgumentError as e:
                    attempts += 1
                    if attempts % 2 != 0:
                        print('Invalid input shape for concat layer. Try padding axis 1')
                        paddings = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]])
                    else:
                        print('Invalid input shape for concat layer. Try padding axes [2, 3]')
                        paddings = tf.constant([[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]])
                    
                    x = tf.pad(x, paddings)
                    if attempts == max_attempts:
                        raise e

            pred = sigmoid(tf.squeeze(pred)) * 255
            
            fx.put(key=f'{i:03}'.encode(), value=pickle.dumps(pred))

    ####################################
    ############### DOG ################
    ####################################
    dog = BlobDoG(3, conf.data.dim_resolution, conf.dog.exclude_border)
    with db.begin() as fx:
        X = fx.cursor()

        dog.fit(
            X=X,
            Y=Y,
            max_match_dist=conf.dog.max_match_dist,
            n_iter=conf.dog.iterations,
            checkpoint_dir=conf.dog.checkpoint_dir,
            n_cpu=conf.dog.n_cpu,
        )

    dog_par = dog.get_parameters()
    print(f"Best parameters found for DoG: {dog_par}")

    with open(f"{conf.dog.checkpoint_dir}/parameters.json", "w") as outfile:
        json.dump(dog_par, outfile)

    db.close()

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)

    main()
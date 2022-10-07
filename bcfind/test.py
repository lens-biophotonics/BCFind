import os
import lmdb
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import concurrent.futures as cf

from bcfind.config_manager import Configuration
from bcfind.localizers import BlobDoG
from bcfind.utils import sigmoid, metrics
from bcfind.data import get_input_tf, normalize_tf


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="test.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("config", type=str, help="path to the YAML configuration file")
    parser.add_argument(
        '--save-pred', 
        default=False, 
        action='store_true', 
        help="wheter to save the predicted locations in the experiment directory or not"
        )
    return parser.parse_args()


def main():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[-1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[-1], True)

    args = parse_args()
    conf = Configuration(args.config)
    
    ####################################
    ############ LOAD UNET #############
    ####################################
    unet = tf.keras.models.load_model(f"{conf.unet.checkpoint_dir}/model.tf")
    unet.build((None, None, None, None, 1))

    ###########################################
    ############ UNET PREDICTIONS #############
    ###########################################
    print('\n', 'PREPARING TEST DATA')

    marker_list = sorted([f'{conf.data.test_gt_dir}/{fname}' for fname in os.listdir(conf.data.test_gt_dir)])
    tiff_list = sorted([f'{conf.data.test_tif_dir}/{fname}' for fname in os.listdir(conf.data.test_tif_dir)])

    assert len(tiff_list) == len(marker_list), f'Number of tiff files, {len(tiff_list)}, differs from that of marker files, {len(marker_list)}.'

    Y = []
    for marker_file in marker_list:
        print(f"Loading file {marker_file}")

        y = pd.read_csv(open(marker_file, "r"))
        y = y[conf.data.marker_columns].dropna(0)
        Y.append(np.array(y))

    print(f"\n UNet predictions on test-set")

    n = len(marker_list)
    nbytes = np.prod(conf.data.shape) * 1 # 4 bytes for float32: 1 byte for uint8
    db = lmdb.open(f'{conf.exp.basepath}/Test_pred_lmdb', map_size=n*nbytes*10)

    with db.begin(write=True) as fx:
        for i, tiff_file in enumerate(tiff_list):
            print(f"Unet prediction on file {i+1}/{len(tiff_list)}")
            
            x = get_input_tf(tf.constant(tiff_file))
            x = normalize_tf(x)
            x = x[tf.newaxis, ..., tf.newaxis]

            max_attempts = 10
            attempt = 0
            while True:
                try:
                    pred = unet(x, training=False)
                    break
                except (tf.errors.InvalidArgumentError, ValueError) as e:
                    if attempt == max_attempts:
                        raise e
                    else:
                        print('Invalid input shape for concat layer. Need padding')
                        attempt += 1
                        if attempt % 2 != 0:
                            paddings = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]])
                        else:
                            paddings = tf.constant([[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]])
                        
                        x = tf.pad(x, paddings)                    
            
            pred = sigmoid(tf.squeeze(pred)) * 255

            pred = pred.astype('uint8')
            fname = tiff_file.split('/')[-1]
            fx.put(key=fname.encode(), value=pickle.dumps(pred))

    ##########################################
    ############ DOG PREDICTIONS #############
    ##########################################
    print(f"\n DoG predictions and evaluation on test-set")

    dog = BlobDoG(3, conf.data.dim_resolution, conf.dog.exclude_border)
    dog_par = json.load(open(f"{conf.dog.checkpoint_dir}/BlobDoG_parameters.json", "r"))
    dog.set_parameters(dog_par)

    print(f"Best parameters found for DoG: {dog_par}")
    with db.begin() as fx:
        X_emb = fx.cursor().iternext(keys=False)

        with cf.ThreadPoolExecutor(5) as pool:
            futures = [
                pool.submit(
                    dog.predict_and_evaluate,
                    x, 
                    y, 
                    conf.dog.max_match_dist,
                    'counts'
                )
                for x, y in zip(X_emb, Y)
            ]
            res = [future.result() for future in cf.as_completed(futures)]

    res = pd.concat(res)
    names = [f.split('/')[-1] for f in sorted(tiff_list)]
    res['file'] = names
    res.to_csv(f"{conf.exp.basepath}/test_eval.csv")
    perf = metrics(res)

    print(f"\n Test-set evaluated with {perf}")
    print("")


if __name__ == '__main__':   
    main()
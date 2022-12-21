import os
import lmdb
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from  numba import cuda

from bcfind.config_manager import TrainConfiguration
from bcfind.data.utils import get_gt_as_numpy
from bcfind.localizers import BlobDoG
from bcfind.utils import metrics
from bcfind.data import get_input_tf
from bcfind.models import predict


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="test.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    parser.add_argument(
        '--save-pred', 
        default=False, 
        action='store_true', 
        help="Wheter to save the predicted locations in the experiment directory or not"
        )
    parser.add_argument('--gpu', type=int, default=-1, help='Which GPU to use. Default to -1')
    return parser.parse_args()


def main():
    args = parse_args()
    
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    conf = TrainConfiguration(args.config)
    
    marker_list = sorted([f'{conf.data.test_gt_dir}/{fname}.marker' for fname in os.listdir(conf.data.test_tif_dir)])
    tiff_list = sorted([f'{conf.data.test_tif_dir}/{fname}' for fname in os.listdir(conf.data.test_tif_dir)])

    ####################################
    ############ LOAD UNET #############
    ####################################
    unet = tf.keras.models.load_model(f"{conf.unet.checkpoint_dir}/model.tf")
    unet.build((None, None, None, None, 1))

    ###########################################
    ############ UNET PREDICTIONS #############
    ###########################################
    print('\n', 'PREPARING TEST DATA')

    # True cell coordinates
    Y = []
    for marker_file in marker_list:
        print(f"Loading file {marker_file}")
        y = get_gt_as_numpy(marker_file)
        Y.append(y)

    # UNet predictions
    print(f"\n UNet predictions on test-set")
    n = len(marker_list)
    nbytes = np.prod(conf.data.shape) * 1 # 4 bytes for float32: 1 byte for uint8
    db = lmdb.open(f'{conf.exp.basepath}/Test_pred_lmdb', map_size=n*nbytes*10)

    with db.begin(write=True) as fx:
        for i, tiff_file in enumerate(tiff_list):
            print(f"Unet prediction on file {i+1}/{len(tiff_list)}")
            
            x = get_input_tf(tiff_file, **conf.preproc)
            pred = predict(x, unet).numpy()
            pred = (pred * 255).astype('uint8')

            fname = tiff_file.split('/')[-1]
            fx.put(key=fname.encode(), value=pickle.dumps(pred))

    db.close()
    cuda.close()
    
    ##########################################
    ############ DOG PREDICTIONS #############
    ##########################################
    print(f"\n DoG predictions and evaluation on test-set")
    
    db = lmdb.open(f'{conf.exp.basepath}/Test_pred_lmdb')

    dog = BlobDoG(3, conf.data.dim_resolution, conf.dog.exclude_border)
    dog_par = json.load(open(f"{conf.dog.checkpoint_dir}/BlobDoG_parameters.json", "r"))
    dog.set_parameters(dog_par)
    print(f"Best parameters found for DoG: {dog_par}")

    with db.begin() as fx:
        res = []
        for i, file_path in enumerate(tiff_list):
            fname = file_path.split('/')[-1]
            x = fx.get(fname.encode())

            pred = dog.predict_and_evaluate(
                x,
                Y[i],
                conf.dog.max_match_dist,
                'counts'
                )

            pred['f1'] = pred['TP'] / (pred['TP'] + .5 * (pred['FP'] + pred['FN']))
            pred['file'] = fname
            res.append(pred)
    db.close()

    res = pd.concat(res)
    res.to_csv(f"{conf.exp.basepath}/Test_eval.csv", index=False)
    perf = metrics(res)

    print(f"\n Test-set evaluated with {perf}")
    print("")


if __name__ == '__main__':   
    main()
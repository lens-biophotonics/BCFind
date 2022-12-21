import os
import lmdb
import pickle
import shutil
import argparse

import numpy as np
import tensorflow as tf

from numba import cuda

from bcfind.config_manager import TrainConfiguration
from bcfind.data.artificial_targets import get_gt_as_numpy
from bcfind.models import UNet, SEUNet, ECAUNet, AttentionUNet, MoUNets, predict
from bcfind.localizers.blob_dog import BlobDoG
from bcfind.losses import FramedCrossentropy3D
from bcfind.data import TrainingDataset, get_input_tf
from bcfind.metrics import Precision, Recall, F1


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="train.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("config", type=str, help="YAML Configuration file")
    parser.add_argument('--gpu', type=int, default=-1, help='Index of GPU to use')
    parser.add_argument('--lmdb', default=False, action='store_true', help='In case of huge dataset store it as lmdb to save RAM usage')
    parser.add_argument('--only-dog', default=False, action='store_true', help='Skip UNet training and train only the DoG')
    parser.add_argument('--test-as-val', default=False, action='store_true', help='Test set will be used as validation during training. No early stopping will be however applied')
    return parser.parse_args()


def main():
    args = parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    conf = TrainConfiguration(args.config)

    train_fnames = os.listdir(conf.data.train_tif_dir)
    train_tiff_files = sorted([f'{conf.data.train_tif_dir}/{fname}' for fname in train_fnames])
    train_marker_files = sorted([f'{conf.data.train_gt_dir}/{fname}.marker' for fname in train_fnames])
    
    test_fnames = os.listdir(conf.data.test_tif_dir)
    test_tiff_files = sorted([f'{conf.data.test_tif_dir}/{fname}' for fname in test_fnames])
    test_marker_files = sorted([f'{conf.data.test_gt_dir}/{fname}.marker' for fname in test_fnames])

    if not args.only_dog:
        ######################################
        ############ CREATE DIRS #############
        ######################################

        # Create experiment directory and copy the config file there
        if not os.path.isdir(conf.exp.basepath):
            os.makedirs(conf.exp.basepath, exist_ok=True)
            config_name = args.config.split('/')[-1]
            shutil.copyfile(args.config, f'{conf.exp.basepath}/{config_name}')
        
        # Create UNet checkpoint dir
        if not os.path.isdir(conf.unet.checkpoint_dir):
            os.makedirs(conf.unet.checkpoint_dir, exist_ok=True)
        
        # Create UNet tensorboard dir
        if os.path.isdir(conf.unet.tensorboard_dir):
            shutil.rmtree(conf.unet.tensorboard_dir, ignore_errors=True)
            os.makedirs(conf.unet.tensorboard_dir)

        ####################################
        ############ UNET DATA #############
        ####################################
        print('\n LOADING UNET DATA')
        
        train_data = TrainingDataset(
            tiff_list=train_tiff_files, 
            marker_list=train_marker_files, 
            batch_size=conf.unet.batch_size, 
            dim_resolution=conf.data.dim_resolution, 
            output_shape=conf.unet.input_shape, 
            augmentations=conf.data_aug.op_args, 
            augmentations_prob=conf.data_aug.op_probs,
            use_lmdb_data=args.lmdb,
            **conf.preproc
            )
        test_data = None
        
        if args.test_as_val:
            print('ATTN!! Loading test-set to use as validation')
            test_data = TrainingDataset(
                tiff_list=test_tiff_files, 
                marker_list=test_marker_files, 
                batch_size=conf.unet.batch_size, 
                dim_resolution=conf.data.dim_resolution, 
                output_shape=conf.unet.input_shape, 
                augmentations=None, 
                augmentations_prob=None,
                use_lmdb_data=args.lmdb,
                )

        ####################################
        ############## UNET ################
        ####################################
        print()
        print('\n BUILDING UNET')
        if conf.unet.model == 'unet':
            unet = UNet(
                n_blocks=conf.unet.n_blocks, 
                n_filters=conf.unet.n_filters, 
                k_size=conf.unet.k_size, 
                k_stride=conf.unet.k_stride,
                dropout=conf.unet.dropout, 
                regularizer=conf.unet.regularizer
                )
        elif conf.unet.model == 'se-unet':
            unet = SEUNet(
                n_blocks=conf.unet.n_blocks, 
                n_filters=conf.unet.n_filters, 
                k_size=conf.unet.k_size, 
                k_stride=conf.unet.k_stride,
                squeeze_factor=conf.unet.squeeze_factor,
                dropout=conf.unet.dropout, 
                regularizer=conf.unet.regularizer
                )
        elif conf.unet.model == 'eca-unet':
            unet = ECAUNet(
                n_blocks=conf.unet.n_blocks, 
                n_filters=conf.unet.n_filters, 
                k_size=conf.unet.k_size, 
                k_stride=conf.unet.k_stride,
                dropout=conf.unet.dropout, 
                regularizer=conf.unet.regularizer
                )
        elif conf.unet.model == 'attention-unet':
            unet = AttentionUNet(
                n_blocks=conf.unet.n_blocks, 
                n_filters=conf.unet.n_filters, 
                k_size=conf.unet.k_size, 
                k_stride=conf.unet.k_stride,
                dropout=conf.unet.dropout, 
                regularizer=conf.unet.regularizer
                )
        elif conf.unet.model == 'moe-unet':
            unet = MoUNets(
                n_blocks=conf.unet.n_blocks, 
                n_filters=conf.unet.n_filters, 
                k_size=conf.unet.k_size,
                k_stride=conf.unet.k_stride,
                dropout=conf.unet.dropout, 
                regularizer=conf.unet.regularizer,
                n_experts=conf.unet.moe_n_experts,
                keep_top_k=conf.unet.moe_top_k_experts,
                add_noise=conf.unet.moe_noise,
                balance_loss=conf.unet.moe_balance_loss,
                )
        else:
            raise ValueError(f'UNet model must be one of ["unet", "se-unet", "eca-unet", "attention-unet", "moe-unet"]. \
                Received {conf.unet.model}.')
        
        
        # loss = FramedFocalCrossentropy3D(exclude_border, input_shape, gamma=3, alpha=None, from_logits=True)
        loss = FramedCrossentropy3D(conf.unet.exclude_border, conf.unet.input_shape, from_logits=True)
        
        prec = Precision(.036, conf.unet.input_shape, conf.unet.exclude_border, from_logits=True)
        rec = Recall(.036, conf.unet.input_shape, conf.unet.exclude_border, from_logits=True)
        f1 = F1(.036, conf.unet.input_shape, conf.unet.exclude_border, from_logits=True)

        optimizer = tf.keras.optimizers.Adam(conf.unet.learning_rate)

        unet.build((None, None, None, None, 1))
        unet.compile(loss=loss, optimizer=optimizer, metrics=[prec, rec, f1])
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
            train_data,
            epochs=conf.unet.epochs,
            callbacks=[MC_callback, TB_callback],
            validation_data=test_data,
            verbose=1,
        )
        del unet

    ####################################
    ############ LOAD UNET #############
    ####################################
    unet = tf.keras.models.load_model(f"{conf.unet.checkpoint_dir}/model.tf")
    unet.build((None, None, None, None, 1))
    
    ####################################
    ############ DOG DATA ##############
    ####################################
    
    # Create DoG checkpoint dir
    if not os.path.isdir(conf.dog.checkpoint_dir):
        os.makedirs(conf.dog.checkpoint_dir)

    print("\n LOADING DoG DATA")
    
    # UNet predictions
    print(f"Saving U-Net predictions in {conf.exp.basepath}/Train_pred_lmdb")
    
    n = len(train_tiff_files)
    nbytes = np.prod(conf.data.shape) * 1 # 4 bytes for float32: 1 byte for uint8
    db = lmdb.open(f'{conf.exp.basepath}/Train_pred_lmdb', map_size=n*nbytes*10)
    
    with db.begin(write=True) as fx:
        for i, tiff_file in enumerate(train_tiff_files):
            print(f"Unet prediction on file {i+1}/{len(train_tiff_files)}")
            
            x = get_input_tf(tiff_file, **conf.preproc)    
            pred = predict(x, unet).numpy()
            pred = (pred * 255).astype('uint8')

            fname = tiff_file.split('/')[-1]
            fx.put(key=fname.encode(), value=pickle.dumps(pred))
    
    db.close()
    cuda.close()

    # True cell coordinates
    Y = []
    for marker_file in train_marker_files:
        print(f"Loading file {marker_file}")
        y = get_gt_as_numpy(marker_file)
        Y.append(y)

    ####################################
    ############### DOG ################
    ####################################
    db = lmdb.open(f'{conf.exp.basepath}/Train_pred_lmdb')

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

    db.close()

if __name__ == '__main__':
    main()
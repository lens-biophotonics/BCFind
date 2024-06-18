import os
import lmdb
import json
import pickle
import shutil
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from numba import cuda

from bcfind.config_manager import TrainConfiguration
from bcfind.data import TrainingDataset
from bcfind.losses import FramedCrossentropy3D
from bcfind.metrics import Precision, Recall, F1
from bcfind.models import UNet, SEUNet, ECAUNet, AttentionUNet, MoUNets, ResUNet
from bcfind.localizers import BlobDoG
from bcfind.utils.models import predict
from bcfind.utils.data import get_input_tf, get_gt_as_numpy
from bcfind.utils.base import evaluate_df


class Trainer:
    def __init__(self):
        self.seed = 2408
        self.reduce_loss = True

    def make_unet_data(
        self,
        train_inputs,
        train_targets,
        dim_resolution,
        input_shape,
        augmentations,
        augmentations_prob,
        batch_size,
        val_inputs=None,
        val_targets=None,
        use_lmdb=False,
        **preprocessing_kwargs,
    ):
        self.unet_train_data = TrainingDataset(
            tiff_list=train_inputs,
            marker_list=train_targets,
            batch_size=batch_size,
            dim_resolution=dim_resolution,
            output_shape=input_shape,
            augmentations=augmentations,
            augmentations_prob=augmentations_prob,
            use_lmdb_data=use_lmdb,
            **preprocessing_kwargs,
        )

        if val_inputs and val_targets:
            self.unet_val_data = TrainingDataset(
                tiff_list=val_inputs,
                marker_list=val_targets,
                batch_size=1,
                dim_resolution=dim_resolution,
                output_shape=input_shape,
                augmentations=augmentations,
                augmentations_prob=augmentations_prob,
                use_lmdb_data=use_lmdb,
                **preprocessing_kwargs,
            )
        else:
            self.unet_val_data = None

    def build_unet(
        self,
        n_blocks,
        n_filters,
        k_size,
        k_stride,
        dropout=None,
        regularizer=None,
        mult_skip=False,
        model_type="unet",
        squeeze_factor=None,
        moe_n_experts=None,
        moe_top_k_experts=None,
        moe_noise=None,
        moe_balance_loss=None,
    ):
        if model_type == "unet":
            self.unet = UNet(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
            )
        elif model_type == "se-unet":
            self.unet = SEUNet(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                squeeze_factor=squeeze_factor,
                dropout=dropout,
                regularizer=regularizer,
            )
        elif model_type == "eca-unet":
            self.unet = ECAUNet(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
            )
        elif model_type == "attention-unet":
            self.unet = AttentionUNet(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
            )
        elif model_type == "moe-unet":
            self.unet = MoUNets(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
                n_experts=moe_n_experts,
                keep_top_k=moe_top_k_experts,
                add_noise=moe_noise,
                balance_loss=moe_balance_loss,
            )
            self.reduce_loss = False
        elif model_type == "res-unet":
            self.unet = ResUNet(
                n_blocks=n_blocks,
                n_filters=n_filters,
                k_size=k_size,
                k_stride=k_stride,
                dropout=dropout,
                regularizer=regularizer,
            )
        else:
            raise ValueError(
                f'UNet model must be one of ["unet", "res-unet", "se-unet", "eca-unet", "attention-unet", "moe-unet"]. \
                Received {model_type}.'
            )

        self.unet.build((None, None, None, None, 1))

    def compile_unet(
        self,
        exclude_border,
        input_shape,
        learning_rate,
        decay_steps=100,
        t_mul=2,
        m_mul=0.8,
        alpha=1e-4,
    ):
        loss = FramedCrossentropy3D(
            exclude_border, input_shape, from_logits=True, reduce=self.reduce_loss
        )

        prec = Precision(0.006, input_shape, exclude_border, from_logits=True)
        rec = Recall(0.006, input_shape, exclude_border, from_logits=True)
        f1 = F1(0.006, input_shape, exclude_border, from_logits=True)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            learning_rate,
            decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
        )

        optimizer = tf.keras.optimizers.SGD(
            lr_schedule, momentum=0.9, nesterov=True, weight_decay=7e-4
        )

        self.unet.compile(loss=loss, optimizer=optimizer, metrics=[prec, rec, f1])

    def fit_unet(
        self, epochs, checkpoint_dir=None, tensorboard_dir=None, val_from_test=False
    ):
        callbacks = []
        if checkpoint_dir:
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)

            monitor = "loss"
            if val_from_test:
                monitor = "val_loss"

            MC_callback = tf.keras.callbacks.ModelCheckpoint(
                f"{checkpoint_dir}/model.tf",
                initial_value_threshold=0.1,
                save_best_only=True,
                save_format="tf",
                save_freq="epoch",
                monitor=monitor,
                mode="min",
                verbose=1,
            )
            callbacks.append(MC_callback)

        if tensorboard_dir:
            if os.path.isdir(tensorboard_dir):
                shutil.rmtree(tensorboard_dir, ignore_errors=True)
            os.makedirs(tensorboard_dir)

            TB_callback = tf.keras.callbacks.TensorBoard(
                tensorboard_dir,
                # update_freq="epoch",
                profile_batch=0,
                write_graph=True,
                write_images=True,
            )
            callbacks.append(TB_callback)

        if len(callbacks) == 0:
            callbacks = None

        self.unet.fit(
            self.unet_train_data,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=self.unet_val_data,
            verbose=1,
        )

    def make_dog_data(
        self, tiff_files, marker_files, data_shape, lmdb_dir, **preprocessing_kwargs
    ):
        n = len(tiff_files)
        nbytes = np.prod(data_shape) * 1  # 4 bytes for float32: 1 byte for uint8
        db = lmdb.open(lmdb_dir, map_size=n * nbytes * 10)

        # UNet predictions
        print(f"Saving U-Net predictions in {lmdb_dir}")
        with db.begin(write=True) as fx:
            for i, tiff_file in enumerate(tiff_files):
                print(f"\nUnet prediction on file {i+1}/{len(tiff_files)}")

                x = get_input_tf(tiff_file, **preprocessing_kwargs)
                pred = predict(x, self.unet).numpy()
                pred = tf.sigmoid(tf.squeeze(pred)).numpy()
                pred = (pred * 255).astype("uint8")

                fname = tiff_file.split("/")[-1]
                fx.put(key=fname.encode(), value=pickle.dumps(pred))

        db.close()
        dog_inputs = lmdb.open(lmdb_dir, readonly=True)

        # True cell coordinates
        dog_targets = []
        for marker_file in sorted(marker_files):
            print(f"Loading file {marker_file}")
            y = get_gt_as_numpy(marker_file)
            dog_targets.append(y)

        return dog_inputs, dog_targets

    def fit_dog(
        self,
        inputs_lmdb,
        targets_list,
        iterations,
        dim_resolution,
        exclude_border,
        max_match_dist,
        checkpoint_dir=None,
        n_cpu=1,
    ):
        cuda.close()
        if checkpoint_dir:
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        self.dog = BlobDoG(3, dim_resolution, exclude_border)
        with inputs_lmdb.begin() as fx:
            X = fx.cursor()
            self.dog.fit(
                X=X,
                Y=targets_list,
                max_match_dist=max_match_dist,
                n_iter=iterations,
                checkpoint_dir=checkpoint_dir,
                n_cpu=n_cpu,
            )

        inputs_lmdb.close()
        return self.dog

    def run(
        self, config_file, only_dog=False, val_from_test=None, use_lmdb=False, gpu=-1
    ):
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu], True)

        conf = TrainConfiguration(config_file)

        train_tiff_files, train_marker_files = get_inputs_target_paths(
            conf.data.train_tif_dir, conf.data.train_gt_dir
        )
        test_tiff_files, test_marker_files = get_inputs_target_paths(
            conf.data.test_tif_dir, conf.data.test_gt_dir
        )

        if val_from_test:
            print("ATTN!! Using part of the test-set as validation")
            nt = len(test_tiff_files)

            np.random.seed(self.seed)
            val_idx = np.random.choice(nt, size=nt // 3, replace=False)

            val_tiff_files = [test_tiff_files[i] for i in val_idx]
            val_marker_files = [test_marker_files[i] for i in val_idx]
            for vtf, vmf in zip(val_tiff_files, val_marker_files):
                test_tiff_files.remove(vtf)
                test_marker_files.remove(vmf)

        if not only_dog:
            ######################################
            ############ CREATE DIRS #############
            ######################################
            # Create experiment directory and copy the config file there
            if not os.path.isdir(conf.exp.basepath):
                os.makedirs(conf.exp.basepath, exist_ok=True)
                config_name = config_file.split("/")[-1]
                shutil.copyfile(config_file, f"{conf.exp.basepath}/{config_name}")

            ####################################
            ############ UNET DATA #############
            ####################################
            print("\nLOADING UNET DATA")
            if val_from_test:
                self.make_unet_data(
                    train_inputs=train_tiff_files,
                    train_targets=train_marker_files,
                    dim_resolution=conf.data.dim_resolution,
                    input_shape=conf.unet.input_shape,
                    augmentations=conf.data_aug.op_args,
                    augmentations_prob=conf.data_aug.op_probs,
                    batch_size=conf.unet.batch_size,
                    val_inputs=val_tiff_files,
                    val_targets=val_marker_files,
                    use_lmdb=use_lmdb,
                    **conf.preproc,
                )
            else:
                self.make_unet_data(
                    train_inputs=train_tiff_files,
                    train_targets=train_marker_files,
                    dim_resolution=conf.data.dim_resolution,
                    input_shape=conf.unet.input_shape,
                    augmentations=conf.data_aug.op_args,
                    augmentations_prob=conf.data_aug.op_probs,
                    batch_size=conf.unet.batch_size,
                    use_lmdb=use_lmdb,
                    **conf.preproc,
                )

            ####################################
            ############## UNET ################
            ####################################
            print("\nBUILDING UNET")
            self.build_unet(
                n_blocks=conf.unet.n_blocks,
                n_filters=conf.unet.n_filters,
                k_size=conf.unet.k_size,
                k_stride=conf.unet.k_stride,
                dropout=conf.unet.dropout,
                regularizer=conf.unet.regularizer,
                model_type=conf.unet.model,
                squeeze_factor=conf.unet.squeeze_factor,
                moe_n_experts=conf.unet.moe_n_experts,
                moe_top_k_experts=conf.unet.moe_top_k_experts,
                moe_noise=conf.unet.moe_noise,
                moe_balance_loss=conf.unet.moe_balance_loss,
            )

            print("\nCOMPILING UNET")
            self.compile_unet(
                exclude_border=conf.unet.exclude_border,
                input_shape=conf.unet.input_shape,
                learning_rate=conf.unet.learning_rate,
            )
            self.unet.summary()

            print("\nTRAINING UNET")
            self.fit_unet(
                epochs=conf.unet.epochs,
                checkpoint_dir=conf.unet.checkpoint_dir,
                tensorboard_dir=conf.unet.tensorboard_dir,
            )

        else:
            print("\nBUILDING UNET")
            self.unet = tf.keras.models.load_model(
                f"{conf.unet.checkpoint_dir}/model.tf",
                compile=False,
            )
            self.unet.build((None, None, None, None, 1))

        ####################################
        ############ DOG DATA ##############
        ####################################
        print("\nLOADING DoG DATA")
        if val_from_test:
            tiff_files = val_tiff_files
            marker_files = val_marker_files
            db_name = "Val_pred_lmdb"
        else:
            tiff_files = train_tiff_files
            marker_files = train_marker_files
            db_name = "Train_pred_lmdb"

        self.dog_train_inputs, self.dog_train_targets = self.make_dog_data(
            tiff_files=tiff_files,
            marker_files=marker_files,
            data_shape=conf.data.shape,
            lmdb_dir=f"{conf.exp.basepath}/{db_name}",
        )

        ####################################
        ############### DOG ################
        ####################################
        print("\nTRAINING DoG")
        self.fit_dog(
            inputs_lmdb=self.dog_train_inputs,
            targets_list=self.dog_train_targets,
            iterations=conf.dog.iterations,
            dim_resolution=conf.data.dim_resolution,
            exclude_border=conf.dog.exclude_border,
            max_match_dist=conf.dog.max_match_dist,
            checkpoint_dir=conf.dog.checkpoint_dir,
            n_cpu=conf.dog.n_cpu,
        )
        print(f"Best parameters found for DoG: {self.dog.get_parameters()}")

    def test(self, config_file, val_from_test, gpu=-1):
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu], True)

        conf = TrainConfiguration(config_file)

        test_tiff_files, test_marker_files = get_inputs_target_paths(
            conf.data.test_tif_dir, conf.data.test_gt_dir
        )
        if val_from_test:
            print("ATTN!! Using part of the test-set as validation")
            nt = len(test_tiff_files)
            np.random.seed(self.seed)
            val_idx = np.random.choice(0, nt, size=nt // 3, replace=False)
            val_tiff_files = [test_tiff_files[i] for i in val_idx]
            val_marker_files = [test_marker_files[i] for i in val_idx]
            for vtf, vmf in zip(val_tiff_files, val_marker_files):
                test_tiff_files.remove(vtf)
                test_marker_files.remove(vmf)

        ####################################
        ############ LOAD UNET #############
        ####################################
        self.unet = tf.keras.models.load_model(
            f"{conf.unet.checkpoint_dir}/model.tf",
            compile=False,
        )
        self.unet.build((None, None, None, None, 1))

        ###########################################
        ############ UNET PREDICTIONS #############
        ###########################################
        print(f"\nPREPARING TEST DATA")
        self.dog_test_inputs, self.dog_test_targets = self.make_dog_data(
            tiff_files=test_tiff_files,
            marker_files=test_marker_files,
            data_shape=conf.data.shape,
            lmdb_dir=f"{conf.exp.basepath}/Test_pred_lmdb",
            **conf.preproc,
        )
        cuda.close()

        ##########################################
        ############ DOG PREDICTIONS #############
        ##########################################
        print(f"\nDoG predictions and evaluation on test-set")
        self.dog = BlobDoG(3, conf.data.dim_resolution, conf.dog.exclude_border)
        dog_par = json.load(
            open(f"{conf.dog.checkpoint_dir}/BlobDoG_parameters.json", "r")
        )
        self.dog.set_parameters(dog_par)
        print(f"Best parameters found for DoG: {dog_par}")

        with self.dog_test_inputs.begin() as fx:
            db_iterator = fx.cursor()
            res = []
            for i, (fname, x) in enumerate(db_iterator):
                pred = self.dog.predict_and_evaluate(
                    x, self.dog_test_targets[i], conf.dog.max_match_dist, "counts"
                )

                pred["f1"] = pred["TP"] / (pred["TP"] + 0.5 * (pred["FP"] + pred["FN"]))
                pred["file"] = fname.decode()
                res.append(pred)
        self.dog_test_inputs.close()

        res = pd.concat(res)
        res.to_csv(f"{conf.exp.basepath}/Test_eval.csv", index=False)
        perf = evaluate_df(res)

        print(f"\nTest-set evaluated with {perf}")
        print("")


def get_inputs_target_paths(input_dir, target_dir):
    fnames = os.listdir(input_dir)
    input_files = sorted([f"{input_dir}/{fname}" for fname in fnames])
    target_files = sorted([f"{target_dir}/{fname}.marker" for fname in fnames])
    return input_files, target_files


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="train.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("config", type=str, help="YAML Configuration file")
    parser.add_argument("--gpu", type=int, default=-1, help="Index of GPU to use")
    parser.add_argument(
        "--lmdb",
        default=False,
        action="store_true",
        help="In case of huge dataset store it as lmdb to save RAM usage",
    )
    parser.add_argument(
        "--only-dog",
        default=False,
        action="store_true",
        help="Skip UNet training and train only the DoG",
    )
    parser.add_argument(
        "--val-from-test",
        default=False,
        action="store_true",
        help="part of the test-set as validation. UNet weights will be saved only when validation loss improves",
    )
    parser.add_argument(
        "--only-test",
        default=False,
        action="store_true",
        help="Run only evaluation on the test-set",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = Trainer()
    if not args.only_test:
        trainer.run(args.config, args.only_dog, args.val_from_test, args.lmdb, args.gpu)

    trainer.test(args.config, args.val_from_test, args.gpu)


if __name__ == "__main__":
    main()

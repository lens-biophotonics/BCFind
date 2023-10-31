import json
import os
import argparse
import numpy as np
import tensorflow as tf

from bcfind.data.artificial_targets import get_target_tf
from bcfind.localizers import BlobDoG
from bcfind.config_manager import TrainConfiguration
from bcfind.utils.data import get_input_tf, get_gt_as_numpy
from bcfind.utils.models import predict
from bcfind.plot import make_video

import bcfind.metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="make_video.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("tiff_dir", type=str, help="Path to tiff directory")
    parser.add_argument("gt_dir", type=str, help="Path to marker directory")
    parser.add_argument("exp_dir", type=str, help="Path to experiment directory")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument(
        "--gpu", type=int, default=-1, help="Which GPU to use. Default to -1"
    )
    return parser.parse_args()


def main(args):
    config_file = [f for f in os.listdir(args.exp_dir) if f.endswith(".yaml")][0]
    conf = TrainConfiguration(f"{args.exp_dir}/{config_file}")

    fnames = sorted(os.listdir(args.tiff_dir))
    tiff_files = [f"{args.tiff_dir}/{f}" for f in fnames]
    marker_files = [f"{args.gt_dir}/{f}.marker" for f in fnames]

    for i in range(len(tiff_files)):
        fname = tiff_files[i].split("/")[-1]

        model = tf.keras.models.load_model(
            f"{args.exp_dir}/UNet_checkpoints/model.tf", compile=False
        )
        model.build((None, None, None, None, 1))

        dog = BlobDoG(3, conf.data.dim_resolution, conf.dog.exclude_border)
        dog_par = json.load(
            open(f"{args.exp_dir}/DoG_checkpoints/BlobDoG_parameters.json")
        )
        dog.set_parameters(dog_par)

        print()
        print(fname)
        print("Loading input and making model prediction")
        x = get_input_tf(tiff_files[i], **conf.preproc)
        pred = predict(x, model).numpy()
        print(
            f"Pred max = {pred.max()}, pred shape = {pred.shape}, input shape = {x.shape}"
        )

        if pred.shape != x.shape:
            print("Changing input shape according to prediction shape")
            x = tf.slice(x, [0, 0, 0], [pred.shape[0], pred.shape[1], pred.shape[2]])

        print("Loading target")
        target = get_target_tf(marker_files[i], pred.shape, conf.data.dim_resolution)

        # print('Making DoG prediction')
        marker = get_gt_as_numpy(marker_files[i])
        center_pred = dog.predict_and_evaluate(
            pred * 255, marker, conf.dog.max_match_dist
        )
        TP = np.sum(center_pred["name"] == "TP")
        FN = np.sum(center_pred["name"] == "FN")
        FP = np.sum(center_pred["name"] == "FP")
        f1 = TP / (TP + 0.5 * (FP + FN))
        print(f"File {fname} evaluated with F1 = {f1}")

        print("Making video")
        outdir = f"{args.exp_dir}/Videos"
        os.makedirs(outdir, exist_ok=True)

        _ = make_video(
            x.numpy() * 255,
            target.numpy() * 255,
            pred * 255,
            center_pred,
            out_filename=f"{outdir}/{fname}.mp4",
        )


if __name__ == "__main__":
    args = parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[args.gpu], "GPU")
    tf.config.experimental.set_memory_growth(gpus[-1], True)

    main(args)

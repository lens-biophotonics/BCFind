import os
import re
import json
import argparse
import threading
import numpy as np
import pandas as pd
import functools as ft
import tensorflow as tf

from pathlib import Path

from skimage import io
from queue import Queue
from zetastitcher import VirtualFusedVolume

from bcfind.localizers import BlobDoG
from bcfind.config_manager import Configuration
from bcfind.utils import sigmoid, preprocessing
from bcfind.losses import FramedCrossentropy3D, FramedFocalCrossentropy3D


def substack_name(x0, y0, z0, patch_shape, overlap):
    return f"sub_{x0}_{y0}_{z0}____{patch_shape[0]}_{patch_shape[1]}_{patch_shape[2]}____{overlap[0]}_{overlap[1]}_{overlap[2]}"


def substack_generator(
    vfv, patch_shape, overlap, preprocessing_fun=None, vfv_mask=None, not_to_do=None
):
    vfv_shape = np.array(vfv.shape)
    no_overlap_shape = np.ceil(patch_shape - overlap).astype(int)
    nz, ny, nx = np.ceil(vfv_shape / no_overlap_shape).astype(int)

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                # Vertices of substack volume
                idx = np.array([z, y, x])
                z0, y0, x0 = no_overlap_shape * idx - np.array(overlap / 2).astype(int)
                z1, y1, x1 = np.minimum(vfv_shape, np.array([z0, y0, x0] + patch_shape))

                if idx[0] == 0:
                    z0 = 0
                if idx[1] == 0:
                    y0 = 0
                if idx[2] == 0:
                    x0 = 0

                # Substack name to save and retrieve position in VFV
                sub_name = substack_name(z0, y0, x0, patch_shape, overlap)
                if sub_name in not_to_do:
                    continue

                # VFV mask handling
                if vfv_mask is not None:
                    mask_shape = np.array(vfv_mask.shape)
                    mask_rescale_factors = vfv_shape // mask_shape

                    m_z0, m_y0, m_x0 = np.array(
                        [z0, y0, x0] / mask_rescale_factors
                    ).astype(int)
                    m_z1, m_y1, m_x1 = np.array(
                        [z1, y1, x1] / mask_rescale_factors
                    ).astype(int)
                    if (vfv_mask[m_z0:m_z1, m_y0:m_y1, m_x0:m_x1] == 0).all():
                        continue

                substack = vfv[z0:z1, y0:y1, x0:x1]

                if preprocessing_fun is not None:
                    substack = preprocessing_fun(substack)

                yield substack, sub_name


def find_cells(img, localizer, frame=None, outfile=None):
    img_shape = img.shape
    centers = localizer.predict(img)

    if centers.shape[0] > 0:
        if frame is not None:
            frame_centers = np.where(
                (centers[:, 0] < frame[0])
                + (centers[:, 1] < frame[1])
                + (centers[:, 2] < frame[2])
                + (centers[:, 0] >= img_shape[0] - frame[0])
                + (centers[:, 1] >= img_shape[1] - frame[1])
                + (centers[:, 2] >= img_shape[2] - frame[2])
            )
            centers = np.delete(centers, frame_centers, axis=0)

    if outfile is not None:
        np.save(outfile, centers)

    return centers


def make_cloud(centers_dir, scale=None):
    f_names = [f for f in os.listdir(centers_dir) if f.endswith(".npy")]
    cloud_df = pd.DataFrame(columns=["x", "y", "z", "sigma_x", "sigma_y", "sigma_z"])

    for f_name in f_names:
        try:
            centers = np.load(f"{centers_dir}/{f_name}")
        except ValueError as e:
            print(f"DEBUG: error loading file {f_name}")
            raise Exception(e)

        if centers.shape[0] > 0:
            # Retrieve offset from file name
            f_info = [int(n) for n in re.split("_|d|m|\.", f_name) if n.isdigit()]
            offset = np.array(f_info[:3])[[2, 1, 0]]
            centers[:, :3] += offset

            if scale is not None:
                centers[:, :3] *= scale
                centers[:, 3:] *= scale

            centers = pd.DataFrame(centers, columns=cloud_df.columns)
            cloud_df = cloud_df.append(centers, ignore_index=True, sort=False)

    return cloud_df


def predict_vfv(
    nn_model,
    localizer,
    vfv,
    patch_shape,
    overlap,
    outdir,
    preprocessing_fun=None,
    vfv_mask=None,
    sub_queue_size=5,
    emb_queue_size=5,
    localizer_threads=10,
):
    os.makedirs(outdir, exist_ok=True)

    substack_q = Queue(maxsize=sub_queue_size)
    embedding_q = Queue(maxsize=emb_queue_size)

    frame = np.array(overlap) // 2

    def nn_worker():
        while True:
            got = substack_q.get()
            if got is None:
                substack_q.task_done()
                break

            substack, name = got

            print(f"UNet prediction on substack {name}")
            nn_pred = np.array(
                nn_model(substack[np.newaxis, ..., np.newaxis], training=False)
            ).reshape(substack.shape)
            nn_pred = sigmoid(nn_pred) * 255

            embedding_q.put([nn_pred, name])
            n = embedding_q.qsize()
            print(f"Embedding_queue has {n} elements")
            substack_q.task_done()

    def localizer_worker():
        while True:
            got = embedding_q.get()
            if got is None:
                embedding_q.task_done()
                break

            nn_pred, name = got
            outfile = f"{outdir}/{name}.npy"

            print(f"DoG prediction on substack {name}")
            _ = find_cells(nn_pred, localizer, frame, outfile)

            embedding_q.task_done()

    # Prepare thread for Neural Network predictions
    nn_thread = threading.Thread(target=nn_worker)
    nn_thread.start()

    # Prepare threads for localizer predictions
    loc_threads = []
    for t in range(localizer_threads):
        loc_threads.append(threading.Thread(target=localizer_worker))
        loc_threads[t].start()

    # Start populating substack queue
    not_to_do = [f for f in os.listdir(outdir) if f.endswith(".npy")]
    sub_gen = substack_generator(
        vfv, patch_shape, overlap, preprocessing_fun, vfv_mask, not_to_do
    )
    for substack, name in sub_gen:
        substack_q.put([substack, name])
        n = substack_q.qsize()
        print(f"Substack_queue has {n} elements.")

    # Close threads
    print("Closing threads")
    for q in range(sub_queue_size):
        substack_q.put(None)
    for q in range(emb_queue_size):
        embedding_q.put(None)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="vfv_pred.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("config", type=str, help="YAML Configuration file")
    return parser.parse_args()


def main():
    args = parse_args()
    conf = Configuration(args.config)

    # Preparing U-Net
    print("Loading UNet and DoG parameters...")
    unet = tf.keras.models.load_model(f"{conf.unet.checkpoint_dir}/model.tf")
    unet.build((None, None, None, None, 1))

    # Preparing DoG
    dog = BlobDoG(len(conf.vfv.patch_shape), conf.data.dim_resolution)
    dog_par_file = Path(conf.dog.checkpoint_dir) / "parameters.json"
    if dog_par_file.exists():
        dog_par = json.load(open(dog_par_file))
        dog.set_parameters(dog_par)
    else:
        print("No optimal DoG parameteres found, assigning defaults.")

    # Loading Virtual Fused Volume and optional mask
    print("Loading VirtualFusedVolume...")
    vfv = VirtualFusedVolume(conf.vfv.config_file)
    if conf.vfv.mask_path is not None:
        vfv_mask = io.imread(conf.vfv.mask_path)
    else:
        vfv_mask = None

    # Define callable preprocessing function
    output_shape = conf.vfv.patch_shape
    if conf.preproc.transpose is not None:
        output_shape = [conf.vfv.patch_shape[i] for i in conf.preproc.transpose]

    preprocessing_fun = ft.partial(
        preprocessing,
        transpose=conf.preproc.transpose,
        flip_axis=None,
        clip_threshold=conf.preproc.clip_threshold,
        gamma_correction=conf.preproc.gamma_correction,
        downscale=conf.preproc.downscale,
        pad_output_shape=output_shape,
    )

    # Start predictions
    print("Starting predictions...")
    predict_vfv(
        unet,
        dog,
        vfv,
        conf.vfv.patch_shape,
        conf.vfv.patch_overlap,
        conf.vfv.pred_outdir,
        preprocessing_fun=preprocessing_fun,
        vfv_mask=vfv_mask,
    )

    # Merge and save all substack predictions
    print("VFV predictions finished, saving results...")
    cloud_df = make_cloud(conf.vfv.pred_outdir, conf.data.dim_resolution)
    cloud_df.to_csv(f"{conf.vfv.outdir}/{conf.vfv.name}_cloud.csv", index=False)


if __name__ == "__main__":
    main()

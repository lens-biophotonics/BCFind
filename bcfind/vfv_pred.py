import os
import re
import json
import shutil
import argparse
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
import skimage.io as skio
import concurrent.futures as cf
import zetastitcher.io.zipwrapper as zw

from queue import Queue
from zetastitcher import VirtualFusedVolume
from cachetools import LRUCache

from bcfind.utils.data import get_preprocess_func
from bcfind.localizers import BlobDoG
from bcfind.config_manager import VFVConfiguration


def substack_name(z0, y0, x0, patch_shape, overlap):
    return f"sub_{z0}_{y0}_{x0}____{patch_shape[0]}_{patch_shape[1]}_{patch_shape[2]}____{overlap[0]}_{overlap[1]}_{overlap[2]}"


def is_out_of_mask(box_coord, vfv_mask, vfv_shape):
    (z0, z1), (y0, y1), (x0, x1) = box_coord
    mask_shape = np.array(vfv_mask.shape)
    mask_rescale_factors = vfv_shape // mask_shape
    if (mask_rescale_factors != 1).all():
        print(
            f"""Found different shapes between VFV ({vfv_shape}) and mask ({mask_shape}). 
            Rescaling locations for mask indexing"""
        )

    m_z0, m_y0, m_x0 = np.array([z0, y0, x0] / mask_rescale_factors).astype(int)
    m_z1, m_y1, m_x1 = np.array([z1, y1, x1] / mask_rescale_factors).astype(int)

    mask_out = False
    if mask_shape[0] == 1:
        if (vfv_mask[0, m_y0:m_y1, m_x0:m_x1] == 0).all():
            mask_out = True
    elif mask_shape[1] == 1:
        if (vfv_mask[m_z0:m_z1, 0, m_x0:m_x1] == 0).all():
            mask_out = True
    elif mask_shape[2] == 1:
        if (vfv_mask[m_z0:m_z1, m_y0:m_y1, 0] == 0).all():
            mask_out = True
    else:
        if (vfv_mask[m_z0:m_z1, m_y0:m_y1, m_x0:m_x1] == 0).all():
            mask_out = True
    return mask_out


def put_substack_in_q(
    idx,
    patch_shape,
    vfv,
    overlap,
    queue,
    min_thresh=0,
    preprocessing_fun=None,
    vfv_mask=None,
    not_to_do=[],
):
    vfv_shape = np.array(vfv.shape)
    no_overlap_shape = np.ceil(patch_shape - overlap).astype(int)

    z0, y0, x0 = no_overlap_shape * idx - (overlap // 2)

    if z0 < 0:
        z0 = 0
    if y0 < 0:
        y0 = 0
    if x0 < 0:
        x0 = 0

    z1, y1, x1 = np.minimum(vfv_shape, np.array([z0, y0, x0] + patch_shape))

    # Substack name to save and retrieve position in VFV
    sub_name = substack_name(z0, y0, x0, patch_shape, overlap)
    print(f"\nRetrieving substack {sub_name}")
    print(not_to_do)
    if sub_name in not_to_do:
        print("Already done! Skipping")
        return

    # mask handling
    if vfv_mask is not None:
        print("Handling mask...")
        box_coord = ((z0, z1), (y0, y1), (x0, x1))
        mask_out = is_out_of_mask(box_coord, vfv_mask, vfv_shape)

        if mask_out:
            print(f"Out of mask! Skipping")
            return

    substack = np.zeros(patch_shape)
    substack[: z1 - z0, : y1 - y0, : x1 - x0] = vfv[z0:z1, y0:y1, x0:x1]

    if np.mean(substack) < min_thresh:
        print(f"Substack mean = {np.mean(substack)} < {min_thresh}. Skipping ")
        return

    if preprocessing_fun is not None:
        print("Preprocessing...")
        substack = preprocessing_fun(substack)

    print("Putting in queue...")
    queue.put([substack, sub_name])
    n = queue.qsize()
    print(f"Substack_queue has {n} elements.")
    return


def find_cells(img, localizer, outfile=None):
    centers = localizer.predict(img)
    if outfile is not None:
        print(f"Saving coordinates to {outfile}")
        np.save(outfile, centers)

    return centers


def make_cloud(centers_dir, scale=None):
    f_names = [f for f in os.listdir(centers_dir) if f.endswith(".npy")]
    cloud_df = []

    for f_name in f_names:
        try:
            centers = np.load(f"{centers_dir}/{f_name}")
        except ValueError as e:
            print(f"DEBUG: error loading file {f_name}")
            raise Exception(e)

        if centers.shape[0] > 0:
            # Retrieve offset from file name
            f_info = [int(n) for n in re.split("_|d|m|\.", f_name) if n.isdigit()]
            offset = np.array(f_info[:3])
            centers[:, :3] += offset

            if scale is not None:
                centers[:, :3] *= scale
                centers[:, 3:] *= scale

            centers = pd.DataFrame(
                centers, columns=["z", "y", "x", "sigma_z", "sigma_y", "sigma_x"]
            )
            cloud_df.append(centers)

    cloud_df = pd.concat(cloud_df, ignore_index=True)
    return cloud_df


def predict_vfv(
    nn_model,
    localizer,
    vfv,
    patch_shape,
    overlap,
    outdir,
    min_thresh=0,
    from_to=None,
    preprocessing_fun=None,
    vfv_mask=None,
    sub_queue_size=5,
    emb_queue_size=10,
    localizer_threads=5,
):
    substack_q = Queue(maxsize=sub_queue_size)
    embedding_q = Queue(maxsize=emb_queue_size)

    not_to_do = [f.split(".")[0] for f in os.listdir(outdir) if f.endswith(".npy")]

    def nn_worker():
        while True:
            got = substack_q.get()
            if got is None:
                substack_q.task_done()
                break

            substack, name = got

            print(f"UNet prediction on substack {name}")
            nn_pred = nn_model(
                substack[tf.newaxis, ..., tf.newaxis],
                training=False,
            )
            nn_pred = tf.sigmoid(tf.squeeze(nn_pred)).numpy() * 255

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
            cells = find_cells(nn_pred, localizer, outfile)
            embedding_q.task_done()
            if cells.shape[0] > 0:
                print(f"{cells.shape[0]} cells found.")

    # Prepare threads for localizer predictions
    loc_threads = []
    for t in range(localizer_threads):
        loc_threads.append(threading.Thread(target=localizer_worker))
        loc_threads[t].start()

    # Prepare thread for Neural Network predictions
    nn_thread = threading.Thread(target=nn_worker)
    nn_thread.start()

    vfv_shape = np.array(vfv.shape)
    no_overlap_shape = np.ceil(patch_shape - overlap).astype(int)
    nz, ny, nx = np.ceil(vfv_shape / no_overlap_shape).astype(int)

    i = -1
    with cf.ThreadPoolExecutor(15) as pool:
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    i += 1
                    if from_to is None:
                        _ = pool.submit(
                            put_substack_in_q,
                            [z, y, x],
                            patch_shape,
                            vfv,
                            overlap,
                            substack_q,
                            min_thresh,
                            preprocessing_fun,
                            vfv_mask,
                            not_to_do,
                        )
                    elif i >= from_to[0] and i < from_to[1]:
                        _ = pool.submit(
                            put_substack_in_q,
                            [z, y, x],
                            patch_shape,
                            vfv,
                            overlap,
                            substack_q,
                            min_thresh,
                            preprocessing_fun,
                            vfv_mask,
                            not_to_do,
                        )

    # Close threads
    print("Closing threads")
    for q in range(sub_queue_size):
        substack_q.put(None)
    for q in range(emb_queue_size):
        embedding_q.put(None)

    nn_thread.join()
    for t in loc_threads:
        t.join()


def get_number_of_patches(vfv_shape, patch_shape, overlap):
    no_overlap_shape = np.ceil(patch_shape - overlap).astype(int)
    nz, ny, nx = np.ceil(vfv_shape / no_overlap_shape).astype(int)
    return nz * ny * nx


def mask_cloud_df(cloud_df, mask, vfv_shape):
    mask_shape = np.array(mask.shape)
    mask_rescale_factors = vfv_shape / mask_shape
    # mask_resolution = conf.vfv.dim_resolution * mask_rescale_factors

    cloud_df_rescaled = cloud_df[["z", "y", "x"]] / (mask_rescale_factors)
    z_coords, y_coords, x_coords = (
        cloud_df_rescaled["z"],
        cloud_df_rescaled["y"],
        cloud_df_rescaled["x"],
    )

    if mask_shape[0] == 1:
        in_mask = [
            mask[0, int(np.floor(y) - 1), int(np.floor(x) - 1)]
            for y, x in zip(y_coords, x_coords)
        ]
    elif mask_shape[1] == 1:
        in_mask = [
            mask[int(np.floor(z)), 0, int(np.floor(x))]
            for z, x in zip(z_coords, x_coords)
        ]
    elif mask_shape[2] == 1:
        in_mask = [
            mask[int(np.floor(z)), int(np.floor(y)), 0]
            for z, y in zip(z_coords, y_coords)
        ]
    else:
        in_mask = [
            mask[int(np.floor(z)), int(np.floor(y)), int(np.floor(x))]
            for z, y, x in zip(z_coords, y_coords, x_coords)
        ]

    cloud_df = cloud_df[np.array(in_mask, dtype=bool)]
    return cloud_df


def save_point_cloud(
    pred_outdir, outfile, dim_resolution=(1, 1, 1), mask=None, vfv_shape=None
):
    print("\nVFV predictions finished, making cloud...")
    cloud_df = make_cloud(pred_outdir, dim_resolution)

    # If mask is given, remove predicted points outside of it
    if mask is not None:
        assert vfv_shape is not None, "If mask is given, also vfv_shape is needed."
        coords_only = cloud_df[["z", "y", "x"]]
        coords_only = mask_cloud_df(coords_only / dim_resolution, mask, vfv_shape)
        cloud_df[["z", "y", "x"]] = coords_only * dim_resolution

    cloud_df.to_csv(outfile, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="vfv_pred.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("config", type=str, help="YAML Configuration file")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=5,
        help="Number of parallel threads to use for blob detection. Default to 5",
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, help="Index of GPU to use. Default to -1"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="""A float in [0, 1] indicating the starting substack index expressed 
             as a percentage of the total number of substacks in the VirtualFusedVolume. 
             Default to 0.""",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=1.0,
        help=""" A float in [0, 1] indicating the ending substack index expressed 
        as a percentage of the total number of substacks in the 
        VirtualFusedVolume. Default to 1.""",
    )
    parser.add_argument(
        "--vfv-cache",
        type=int,
        default=32,
        help="Number of VFV calls to cache. Default to 32.",
    )
    parser.add_argument(
        "--min-thresh",
        type=int,
        default=0,
        help="""Substacks whose mean is below this threshold will be discarded. 
        Default to 0.""",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[args.gpu], "GPU")
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    conf = VFVConfiguration(args.config)
    # create directory to save predictions and copy the config file in there
    os.makedirs(conf.vfv.pred_outdir, exist_ok=True)
    config_name = os.path.basename(args.config)
    dst = os.path.join(conf.vfv.outdir, config_name)
    if os.path.exists(f"{conf.vfv.outdir}/{config_name}"):
        if os.path.samefile(args.config, dst):
            pass
        else:
            os.remove(dst)
            shutil.copy(args.config, conf.vfv.outdir)

    # Preparing U-Net
    print("\nLoading UNet and DoG parameters...")
    unet = tf.keras.models.load_model(
        f"{conf.unet.checkpoint_dir}/model.tf", compile=False
    )
    unet.build((None, None, None, None, 1))

    # Preparing DoG
    dog = BlobDoG(
        len(conf.vfv.patch_shape),
        conf.vfv.dim_resolution,
        np.array(conf.vfv.patch_overlap) // 2,
    )
    dog_par_file = f"{conf.dog.checkpoint_dir}/BlobDoG_parameters.json"
    if os.path.isfile(dog_par_file):
        dog_par = json.load(open(dog_par_file))
        dog.set_parameters(dog_par)
    else:
        print("No optimal DoG parameteres found, assigning defaults.")

    # Loading Virtual Fused Volume and optional mask
    print("\nLoading VirtualFusedVolume...")
    if conf.vfv.config_file.endswith(".yml"):
        zw.set_cache(LRUCache(maxsize=args.vfv_cache))
        vfv = VirtualFusedVolume(conf.vfv.config_file)
    elif conf.vfv.config_file.endswith(".tif") or conf.vfv.config_file.endswith(
        ".tiff"
    ):
        vfv = skio.imread(conf.vfv.config_file)
    else:
        raise ValueError(
            f'VFV not found. {conf.vfv.config_file}: file format not supported, not in [".yml", ".tif", ".tiff"]'
        )

    n = get_number_of_patches(
        vfv.shape,
        conf.vfv.patch_shape,
        conf.vfv.patch_overlap,
    )
    print(f"VirtualFusedVolume with shape = {vfv.shape}")
    print(f"{n} patches to complete the whole prediction.")

    if conf.vfv.mask_path is not None:
        vfv_mask = skio.imread(conf.vfv.mask_path)
        print(
            f"""\nMask found! Shape = {vfv_mask.shape}, values = 
              {np.unique(vfv_mask)}"""
        )
    else:
        vfv_mask = None

    # Start predictions
    print("\nStarting predictions...")
    s, e = int(args.start * n), int(args.end * n)
    predict_vfv(
        unet,
        dog,
        vfv,
        conf.vfv.patch_shape,
        conf.vfv.patch_overlap,
        conf.vfv.pred_outdir,
        min_thresh=args.min_thresh,
        from_to=[s, e],
        preprocessing_fun=get_preprocess_func(**conf.preproc),
        vfv_mask=vfv_mask,
        localizer_threads=args.n_jobs,
    )

    # Merge and save all substack predictions
    if args.end == 1.0:
        save_point_cloud(
            conf.vfv.pred_outdir,
            f"{conf.vfv.outdir}/{conf.vfv.name}_cloud.csv",
            conf.vfv.dim_resolution,
            vfv_mask,
            vfv.shape,
        )


if __name__ == "__main__":
    main()

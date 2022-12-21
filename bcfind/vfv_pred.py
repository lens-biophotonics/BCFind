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

from queue import Queue
from zetastitcher import VirtualFusedVolume

from bcfind.data.utils import get_preprocess_func
from bcfind.localizers import BlobDoG
from bcfind.config_manager import VFVConfiguration


def substack_name(z0, y0, x0, patch_shape, overlap):
    return f"sub_{z0}_{y0}_{x0}____{patch_shape[0]}_{patch_shape[1]}_{patch_shape[2]}____{overlap[0]}_{overlap[1]}_{overlap[2]}"


def put_substack_in_q(idx, patch_shape, vfv, overlap, queue, preprocessing_fun=None, vfv_mask=None, not_to_do=None):
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
    print(f'\nRetrieving substack {sub_name}')
    if sub_name in not_to_do:
        print('Already done! Skipping')
        return

    # VFV mask handling
    if vfv_mask is not None:
        print('Handling mask...')
        mask_shape = np.array(vfv_mask.shape)
        mask_rescale_factors = vfv_shape // mask_shape
        if (mask_rescale_factors != 1).all():
            print(f'Found different shapes between VFV ({vfv_shape}) and mask ({mask_shape}). Rescaling locations for mask indexing')

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

        if mask_out:
            print(f'Out of mask! Skipping')
            return

    substack = np.zeros(patch_shape)
    substack[:z1-z0, :y1-y0, :x1-x0] = vfv[z0:z1, y0:y1, x0:x1]

    if preprocessing_fun is not None:
        print('Preprocessing...')
        substack = preprocessing_fun(substack)
    
    print('Putting in queue...')
    queue.put([substack, sub_name])
    n = queue.qsize()
    print(f"Substack_queue has {n} elements.")
    return


def find_cells(img, localizer, outfile=None):
    centers = localizer.predict(img)
    if outfile is not None:
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

            centers = pd.DataFrame(centers, columns=["z", "y", "x", "sigma_z", "sigma_y", "sigma_x"])
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
    preprocessing_fun=None,
    vfv_mask=None,
    sub_queue_size=5,
    emb_queue_size=5,
    localizer_threads=5,
):
    substack_q = Queue(maxsize=sub_queue_size)
    embedding_q = Queue(maxsize=emb_queue_size)
    
    not_to_do = [f.split('.')[0] for f in os.listdir(outdir) if f.endswith(".npy")]

    def nn_worker():
        while True:
            got = substack_q.get()
            if got is None:
                substack_q.task_done()
                break

            substack, name = got

            print(f"UNet prediction on substack {name}")
            nn_pred = nn_model(substack[tf.newaxis, ..., tf.newaxis], training=False)
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
            print(cells.shape)

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

    with cf.ThreadPoolExecutor(15) as pool:
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    _ = pool.submit(
                            put_substack_in_q,
                            [z, y, x],
                            patch_shape,
                            vfv,
                            overlap,
                            substack_q,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="vfv_pred.py",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=52, width=90
        ),
    )
    parser.add_argument("config", type=str, help="YAML Configuration file")
    parser.add_argument('--gpu', type=int, default=-1, help='Index of GPU to use. Default to -1')
    return parser.parse_args()


def main():
    args = parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    conf = VFVConfiguration(args.config)
    # create directory to save predictions and copy the config file in there
    os.makedirs(conf.vfv.pred_outdir, exist_ok=True)
    try:
        shutil.copy(args.config, conf.vfv.outdir)
    except shutil.SameFileError:
        pass

    # Preparing U-Net
    print("Loading UNet and DoG parameters...")
    unet = tf.keras.models.load_model(f"{conf.unet.checkpoint_dir}/model.tf")
    unet.build((None, None, None, None, 1))

    # Preparing DoG
    dog = BlobDoG(len(conf.vfv.patch_shape), conf.vfv.dim_resolution, np.array(conf.vfv.patch_overlap) // 2)
    dog_par_file = f'{conf.dog.checkpoint_dir}/BlobDoG_parameters.json'
    if os.path.isfile(dog_par_file):
        dog_par = json.load(open(dog_par_file))
        dog.set_parameters(dog_par)
    else:
        print("No optimal DoG parameteres found, assigning defaults.")

    # Loading Virtual Fused Volume and optional mask
    print("Loading VirtualFusedVolume...")
    if conf.vfv.config_file.endswith('.yml'):
        vfv = VirtualFusedVolume(conf.vfv.config_file)
    elif conf.vfv.config_file.endswith('.tif') or conf.vfv.config_file.endswith('.tiff'):
        vfv = skio.imread(conf.vfv.config_file)
    else:
        raise ValueError(f'VFV not found. {conf.vfv.config_file}: file format not supported, not in [".yml", ".tif", ".tiff"]')
    print(f"VirtualFusedVolume with shape = {vfv.shape}")
    
    if conf.vfv.mask_path is not None:
        vfv_mask = skio.imread(conf.vfv.mask_path)
        print(f'Mask found! Shape = {vfv_mask.shape}, values = {np.unique(vfv_mask)}')
    else:
        vfv_mask = None

    # Start predictions
    print("Starting predictions...")
    predict_vfv(
        unet,
        dog,
        vfv,
        conf.vfv.patch_shape,
        conf.vfv.patch_overlap,
        conf.vfv.pred_outdir,
        preprocessing_fun=get_preprocess_func(**conf.preproc),
        vfv_mask=vfv_mask,
    )

    # Merge and save all substack predictions
    cloud_df = make_cloud(conf.vfv.pred_outdir, conf.vfv.dim_resolution)
    
    # If mask is given, remove predicted points outside of it
    if vfv_mask is not None:
        mask_shape = np.array(vfv_mask.shape)
        mask_rescale_factors = vfv.shape // mask_shape

        cloud_df_rescaled = cloud_df[['z', 'y', 'x']] / mask_rescale_factors
        z_coords, y_coords, x_coords = cloud_df_rescaled['z'], cloud_df_rescaled['y'], cloud_df_rescaled['x']
        
        if mask_shape[0] == 1:
            in_mask = [vfv_mask[0, int(np.floor(y)), int(np.floor(x))] for y, x in zip(y_coords, x_coords)]
        elif mask_shape[1] == 1:
            in_mask = [vfv_mask[int(np.floor(z)), 0, int(np.floor(x))] for z, x in zip(z_coords, x_coords)]
        elif mask_shape[2] == 1:
            in_mask = [vfv_mask[int(np.floor(z)), int(np.floor(y)), 0] for z, y in zip(z_coords, y_coords)]
        else:
            in_mask = [vfv_mask[int(np.floor(z)), int(np.floor(y)), int(np.floor(x))] for z, y, x in zip(z_coords, y_coords, x_coords)]
        
        cloud_df = cloud_df[np.array(in_mask, dtype=bool)]

    print("VFV predictions finished, saving results...")
    cloud_df.to_csv(f"{conf.vfv.outdir}/{conf.vfv.name}_cloud.csv", index=False)


if __name__ == "__main__":
    main()

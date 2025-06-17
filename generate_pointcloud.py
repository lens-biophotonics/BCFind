import argparse
import yaml
import sys
import skimage.io as skio
import os
import numpy as np
import time

import zetastitcher.io.zipwrapper as zw
from zetastitcher import VirtualFusedVolume
from cachetools import LRUCache

from pipeline import *


def parse_args():
    """
        Returns the users provided args
    """
    parser = argparse.ArgumentParser(
        description="Generate point clouds"
    )
    parser.add_argument(
        "config",
        metavar="CONFIG_YAML",
        help="Path to the YAML configuration file"
    )

    return parser.parse_args()


def load_config(path: str) -> dict:
    """
        Load and return YAML config as a dict.
    """
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(f"Error: config file not found: {path}")
    except yaml.YAMLError as e:
        sys.exit(f"Error parsing YAML ({path}):\n{e}")


def main():
    """
        Responsible for conducting all the task to the generate point clouds of the neurons
    """
    # read the user provided argument
    args = parse_args()
    # get the params from the YAML config file
    params = load_config(args.config)

    # acquistion's stitch yml file path
    stitch_yml_path = params['paths']['stitch_yml']
    # acquistion's mask file path
    mask_path = params['paths']['mask']

    # path to the U-Net model 
    unet_dir = os.path.join(params['paths']['models'], "UNet_checkpoints")
    # path to the DoG config
    dog_dir = os.path.join(params['paths']['models'], 'DoG_checkpoints')

    # Virtual Fused Volume cache size
    vfv_cache_size = params['data_structures_size']['vfv_cache_size']
    # batch size of the patches
    batch_size = params['data_structures_size']['batch_size']
    # queue size of the batches
    queue_size = params['data_structures_size']['queue_size']

    # patch shape
    patch_shape = params['acquistion']['patch_shape']
    # patch overlap region
    patch_overlap = params['acquistion']['patch_overlap']
    # scale
    dim_resolution = params['acquistion']['dim_resolution']

    # preprocessing params for the patches
    preprocessing = params['acquistion']['preprocessing']
    # preprocessing function
    preprocessing_func = get_preprocessing_func(**preprocessing)

    # from-till percentage process the patches, inclusive and exclusive respectively.
    patches_progression_pcts = params['pipeline']['patches_progression_pcts']

     # output dir path
    outdir = params['paths']['outdir']
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # param if to save all the patches corrdinates
    save_patches_coordinates = {}
    save_patches_coordinates['save'] = params['pipeline']['save_patches_coordinates']
    save_patches_coordinates['path'] = os.path.join(outdir, 'patches_coordinates.csv')

    # acquistion's Virtual Fused Volume
    vfv = VirtualFusedVolume(stitch_yml_path)
    # set zipwrapper cache to LRUCache
    zw.set_cache(LRUCache(maxsize=vfv_cache_size))

    # initialise acquistion's mask
    mask = None
    include_neuron_mask_region_flag = False
    if mask_path:
        mask = skio.imread(mask_path)
        # set the z-axis as a flat-axis when the mask is 2D
        if (len(mask.shape) == 2):
            mask = mask[np.newaxis, :, :]
        # when there are other values in the mask, other than 0 and 1.
        include_neuron_mask_region_flag = not np.all((mask == 0) | (mask == 1))

    # initisialing U-Net and DoG models
    unet = get_unet(unet_dir)
    dog = get_dog(dog_dir, patch_shape, patch_overlap, dim_resolution)

    # csv file path to save the detected neurons coords
    progression_name = "_".join([str(ptc) for ptc in patches_progression_pcts])
    neurons_coords_filename = progression_name + '_neurons_coords.csv'
    neurons_coords_csv_path  = os.path.join(outdir, neurons_coords_filename)
    # get last processed patch name, else returns 'patch_-1_-1_-1'
    last_processed_patch_name = get_last_processed_patch_name(path=neurons_coords_csv_path)

    # initialise the patch generator
    patches = patch_generator(
        vfv=vfv,
        patch_shape=patch_shape,
        patch_overlap=patch_overlap,
        mask=mask,
        preprocessing_func=preprocessing_func,
        save_patches_coordinates=save_patches_coordinates,
        patches_progression_pcts=patches_progression_pcts,
        last_processed_patch_name=last_processed_patch_name
    )

    start_time = time.time()

    print("Starting The Process To Detect Neurons!")

    # start the final main process
    final_process(
        patches=patches,
        unet=unet,
        dog=dog,
        batch_size=batch_size, 
        queue_size=queue_size,
        dim_resolution=dim_resolution,
        mask=mask,
        include_neuron_mask_region_flag=include_neuron_mask_region_flag,
        vfv_shape=vfv.shape,
        neurons_coords_csv_path=neurons_coords_csv_path
    )

    end_time = time.time()

    print(f"Neurons Detection Process Is Finised from {patches_progression_pcts[0]} till {patches_progression_pcts[1]}!\n\tTotal time: {(end_time-start_time)/60} hrs")

    # after the last progression has been processed
    if patches_progression_pcts[1] == 100:
        print("Clearning And Merging All The Neurons Coords Files.")

        # clean and merge all the detected neurons coords in all the
        # processed progressions into one csv file
        clean_merge_neurons_coords_csv_files(
            outdir=outdir,
            merged_filename='neurons_coords.csv'
        )

        print("Cleaning And Merging Process Complete!")

if __name__ == "__main__":
    main()


import numpy as np
import time
import pandas as pd
from math import ceil
from tqdm import tqdm

from .mask import is_patch_outside_mask
    


def get_patch_name(z0, y0, x0) -> str:
    """
    This function creates and returns the name of a patch based on it starting coords.

    Parameters:
        z0 : starting z coord.
        y0 : starting y coord..
        z0 : starting x coord.

    Returns:
        patch_name : The patch name
    """
    try:
        return f"patch_{z0}_{y0}_{x0}"
    except Exception as exc:
        raise Exception(
            f"@@@@@@@ Error occured when generating a patch name.\n{exc}"
        )



def get_start_end_indices(start_pct, end_pct, n_of_patches):
    """
    This function is responsible for getting the starting and ending index given the
    total number of patches and also the percentage value to perform slicing.

    Parameters:
        start_pct : starting percentage value as a 'from index'.
        end_pct : ending percentage value as a 'till index'.
        n_of_patches : Total number of patches

    Returns:
        indices : from index, till index
    """
    try:
        if start_pct < 0:
            start_pct = 0
        if end_pct > 100:
            end_pct = 100
        if end_pct < start_pct:
            raise ValueError("@@@@@@@ end_pct must be ≥ start_pct")
        if n_of_patches < 1:
            raise ValueError("@@@@@@@ n_of_patches must be ≥ 1")
        
        start_i = ceil(start_pct * n_of_patches / 100)
        end_i = ceil(end_pct * n_of_patches / 100)

        # exta clamp
        start_i = max(0, min(start_i, n_of_patches))
        end_i   = max(0, min(end_i,   n_of_patches))

        return start_i, end_i
    except Exception as exc:
        raise Exception(
            f"@@@@@@@ Error when getting the start and till indices based on the provided pct.\n{exc}"
        )


def check_already_processed(to_be_processed_patch_name, last_processed_patch_name) -> bool:
    """
    This function is responsible for checking if a patch has alrady been processed or not.
    Note: The last processed patch here is not referring to a local last processed patch, but
    refers to the last processed patch if the same process was running before but got interuppted.

    Parameters:
        to_be_processed_patch_name : the about to be processed patch name.
        last_processed_patch_name : the last processed patch name.

    Returns:
        already_processed : boolean true is the to be processed patch has already been processed.
    """
    try:
        to_be_processed_patch_coords = [int(coord) for coord in to_be_processed_patch_name.split("_") if coord.isdigit()]
        last_processed_patch_coords = [int(coord) for coord in last_processed_patch_name.split("_") if coord.isdigit()]

        return to_be_processed_patch_coords <= last_processed_patch_coords
    except Exception as exc:
        raise Exception(
            f"@@@@@@@@ Error when check if a patch has been already processed.\n{exc}"
        )



def get_coordinates(
        vfv,
        patch_shape,
        patch_overlap,
        mask,
        save_patches_coordinates,
        patches_progression_pcts
    ) -> list:
    """
    This function returns a list of all the patches coordinates (with their respective name and offset) that needs to be process based on if the mask is provided

    Parameters:
        vfv                       : Virtual Fused Volumne.
        patch_shape,              : The shape of the patch.
        patch_overlap,            : The overlap region.
        mask,                     : The mask.
        save_patches_coordinates, : A dict to specify if to save the coordinates of all the patches.
        patches_progression_pcts, : A list to specify the range of patches to be process [from pct (inclusive), till pct (exclusive)]

    Returns:
        patches_coordinates : A list containing all the patches
    """
    try:
        # vfv shape
        vfv_z, vfv_y, vfv_x = vfv.shape

        # converting to NumPy arrays
        patch_shape   = np.array(patch_shape,   dtype=int)
        patch_overlap = np.array(patch_overlap, dtype=int)

        # calculating the actual steps between patches (because there is a overlap)
        patch_step_z, patch_step_y, patch_step_x = patch_shape - patch_overlap

        print(f"@@@@@ VFV Shape: {vfv.shape}")
        print(f"@@@@@ Patch Shape : {patch_shape}")
        print(f"@@@@@ Patch Overlap : {patch_overlap}")
        print(f"@@@@@ Steps between two patches removing overlap: {(patch_step_z, patch_step_y, patch_step_x)}")
        
        # list to store all the patches coordinates
        patches_coordinates = []
        # for every patch of size (step_z, step_y, step_x),
        # traversing Z->Y->X
        for z0_patch in range(0, vfv_z, patch_step_z):
            for y0_patch in range(0, vfv_y, patch_step_y):
                for x0_patch in range(0, vfv_x, patch_step_x):
                    # setting the end coordinates of the patch,
                    # or clamp to the volume bounds
                    z1_patch = min(z0_patch + patch_shape[0], vfv_z)
                    y1_patch = min(y0_patch + patch_shape[1], vfv_y)
                    x1_patch = min(x0_patch + patch_shape[2], vfv_x)

                    # patch coords
                    patch_coord = (
                        (z0_patch, z1_patch),
                        (y0_patch, y1_patch),
                        (x0_patch, x1_patch)
                    )
                    # mask check
                    if mask is not None and is_patch_outside_mask(
                        patch_coord,
                        mask, 
                        vfv
                    ):
                        continue


                    # patch name
                    patch_name = get_patch_name(z0_patch, y0_patch, x0_patch)

                    # patch offset
                    patch_offset = np.array([z0_patch, y0_patch, x0_patch])

                    # patch
                    patch_coordinates = {
                        'z0' : z0_patch,
                        'z1' : z1_patch,
                        'y0' : y0_patch,
                        'y1' : y1_patch,
                        'x0' : x0_patch,
                        'x1' : x1_patch,
                        'patch_name' : patch_name,
                        'patch_offset' : patch_offset
                    }

                    patches_coordinates.append(patch_coordinates)

        # save patches coordinates
        if save_patches_coordinates['save']:
            pd.DataFrame(patches_coordinates).drop(columns=['z1', 'y1', 'x1', 'patch_offset']).to_csv(
                path_or_buf=save_patches_coordinates['path'],
                index=False
            )

            
        
        # get the starting (inclusive) and ending (exclusive) indicies according to the given progression
        start_i, end_i = get_start_end_indices(
            start_pct=patches_progression_pcts[0],
            end_pct=patches_progression_pcts[1],
            n_of_patches=len(patches_coordinates)
        )

        patches_to_be_processed = patches_coordinates[start_i:end_i]

        print(f"@@@@@@@ Total number of patches detected {len(patches_coordinates)}.")
        print(f"@@@@@@@ Based on the given progression {patches_progression_pcts[0]}% - {patches_progression_pcts[1]}%")
        print(f"@@@@@@@ Total number of the patches to be processed {len(patches_to_be_processed)}")
        print(f"@@@@@@@ Processing patches from index {start_i} (inclusive) - {end_i} (exclusive).")
        print(f"@@@@@@@ So, the first patch to be processed is {patches_to_be_processed[0]['patch_name']},\n\tand the last patch to be processed is {patches_to_be_processed[-1]['patch_name']}")
                                                                                                                                                                                                    
        return patches_to_be_processed
    except Exception as exc:
        raise Exception(
            f"@@@ Error when getting patches:\n{exc}"
        )



def patch_generator(
        vfv,
        patch_shape,
        patch_overlap,
        mask,
        preprocessing_func,
        save_patches_coordinates,
        patches_progression_pcts,
        last_processed_patch_name
    ):
    """
    This Generator function is responsible for yielding patches

    Parameters:
        vfv                       : Virtual Fused Volumne.
        patch_shape,              : The shape of the patch.
        patch_overlap,            : The overlap region.
        mask,                     : The mask.
        preprocessing_func,       : The preprocessing func to apply over the patch.
        save_patches_coordinates, : A dict to specify if to save the coordinates of all the patches.
        patches_progression_pcts, : A list to specify the range of patches to be process [from pct (inclusive), till pct (exclusive)]
        last_processed_patch_name : The last processed patch name.

    Yield:
        patch : A tuple containing patch corrdinates and metadata like name and offset 
    """
    try:
        print("@@@@ Starting Patch Generator")

        # converting to NumPy arrays
        patch_shape   = np.array(patch_shape,   dtype=int)
        patch_overlap = np.array(patch_overlap, dtype=int)

        # get all the coordinates of the patches to be processed (with mask if provided)
        patches_coordinates = get_coordinates(
            vfv=vfv, 
            patch_shape=patch_shape,
            patch_overlap=patch_overlap,
            mask=mask,
            save_patches_coordinates=save_patches_coordinates,
            patches_progression_pcts=patches_progression_pcts
        )
        
        # for every patch coords
        for patch_coordinates in tqdm(patches_coordinates, desc="Patch Collection"):
            start = time.time()
            
            # patch name
            patch_name = patch_coordinates['patch_name']

            # skip if this patch has already been processed
            if check_already_processed(
                to_be_processed_patch_name=patch_name,
                last_processed_patch_name=last_processed_patch_name
            ):
                print(f"@@@@@@@@ Skipping patch {patch_name}, because it has already been processed, as the last processed patch was {last_processed_patch_name} for the progression {patches_progression_pcts[0]} - {patches_progression_pcts[1]}.")
                continue

            # patch offset
            patch_offset = patch_coordinates['patch_offset']

            # patch coordinates
            z0_patch = patch_coordinates['z0']
            z1_patch = patch_coordinates['z1']
            y0_patch = patch_coordinates['y0']
            y1_patch = patch_coordinates['y1']
            x0_patch = patch_coordinates['x0']
            x1_patch = patch_coordinates['x1'] 

            # creating patch while making sure the shape
            # reamins same for all the cases i.e., edge cases
            patch = np.zeros(patch_shape, dtype=vfv.dtype)
            temp_patch = vfv[
                z0_patch:z1_patch,
                y0_patch:y1_patch,
                x0_patch:x1_patch
            ]
            dz, dy, dx = temp_patch.shape
            patch[:dz,:dy,:dx] = temp_patch
            
            # preprocess
            if preprocessing_func:
                print(f"@@@@@@@@@ Performing PreProcessing on the patch {(z0_patch, y0_patch, x0_patch)}!")
                patch = preprocessing_func(patch)
            
            end = time.time()
            print(f"@@@@@@@@@@ Patch collection time {end - start}")
            
            yield (patch, {'patch_name': patch_name, 'patch_offset': patch_offset})
    except Exception as exc:
        raise Exception(
            f"@@@@ Error occured during the generator is generating a batch:\n{exc}"
        )
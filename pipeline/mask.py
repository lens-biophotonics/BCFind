import numpy as np
import logging

def is_patch_outside_mask(
        patch_coord,
        mask,
        vfv
    ) -> bool:
    """
    Determine if a patch is outside the mask region.

    Parameters:
        patch_coord : ((z0, z1),(y0, y1),(x0, x1))
        mask: vfv mask
        vfv: vfv

    Returns:
        outside: True if the entire patch region is outside the mask (all values == 0).
    """
    try:
        mask_shape = np.array(mask.shape)
        vfv_shape = np.array(vfv.shape)

        # scaling factor from full-res to mask-res
        rescale = vfv_shape / mask_shape
        if not np.allclose(rescale, 1.0):
            logging.debug(
                f"$$$$$$ Rescaling coords from VFV {tuple(vfv_shape)} to mask {tuple(mask_shape)} at recale factor {rescale}"
            )

        # extract starts and ends
        starts = np.array([c[0] for c in patch_coord])
        ends   = np.array([c[1] for c in patch_coord])

        # map to mask indices, floor starts, ceil ends
        m0 = np.floor_divide(starts, rescale).astype(int)
        m1 = np.ceil(ends / rescale).astype(int)
        # clamp to valid range
        m0 = np.clip(m0, 0, mask_shape - 1)
        m1 = np.clip(m1, 0, mask_shape)

        # build slicing tuple, collapse dims of size 1
        slices = []
        for dim, (s0, s1) in enumerate(zip(m0, m1)):
            if mask_shape[dim] == 1:
                slices.append(0)
            else:
                slices.append(slice(s0, s1))
        submask = mask[tuple(slices)]


        # return True if all zeros
        return bool(np.all(submask == 0))
    except Exception as exc:
        raise Exception(
            f"$$$$$$ Error when performming mask check on patch {patch_coord}:\n{exc}"
        )
    

def filter_neurons_outside_mask(
        neurons_coords,
        mask,
        vfv_shape
    ):
    """
    This function is responsible for filtering put all the neurons that are outside the given mask.

    Parameters:
        neurons_coords          : A list containing all the neurons coords detected by the DoG model
        mask                    : The acquistion's mask
        vfv_shape               : The shape of the acquistion's Virtual Fused Volume

    Returns:
        neurons_coords: The neurons coords that are inside the mask.
        neurons_mask_regions: The region the neuron is under in the mask.
    """
    try:
        mask_shape = np.array(mask.shape)
        # scaling factor from full-res to mask-res
        rescale = np.array(vfv_shape) / mask_shape

        # rescaled neurons coords
        rescaled_neuron_coords = np.floor_divide(neurons_coords, rescale).astype(int)

        # when the z-axis is a flat axis in the mask (mask_shape[0] == 1)
        if mask_shape[0] == 1:
            rescaled_neuron_coords[:, 0] = 0

        # when there is a undershoot or overshoot clip to bound. 
        for ax in range(3):
            rescaled_neuron_coords[:,ax] = np.clip(
                rescaled_neuron_coords[:,ax],
                0,
                mask_shape[ax]-1
            )

        mask_values = mask[
            rescaled_neuron_coords[:, 0],
            rescaled_neuron_coords[:, 1],
            rescaled_neuron_coords[:, 2]
        ]

        # a boolean neurons-in-mask array
        in_mask = mask_values.astype(bool)

        # returns only the neurons are inside the mask and thier region
        return neurons_coords[in_mask], mask_values[in_mask]
    except Exception as exc:
        raise Exception(
            f"$$$$$$$$$$$$$$$$$ Error when performing mask filtering over the detected neurons coords.\nDetected Neuron Coords: {neurons_coords}\nException:\n{exc}"
        )

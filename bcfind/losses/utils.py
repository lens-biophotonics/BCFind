import numpy as np
import functools as ft
import tensorflow as tf


def get_mask_fn(target_shape, border_size):
    framing_mask = np.zeros(target_shape)
    framing_mask[
        border_size[0] : target_shape[0] - border_size[0],
        border_size[1] : target_shape[1] - border_size[1],
        border_size[2] : target_shape[2] - border_size[2],
    ] = 1

    framing_mask = tf.convert_to_tensor(
        framing_mask.astype("bool"), dtype=tf.bool
    )
    return ft.partial(tf.boolean_mask, mask=framing_mask)

import tensorflow as tf
import functools as ft
import numpy as np


class FramedCrossentropy3D(tf.keras.losses.Loss):
    """
    Implementation of binary crossentropy loss for 3D images where the predictions
    at the borders are not included in the computation.
    """

    def __init__(self, border_size, target_shape, **kwargs):
        super(FramedCrossentropy3D, self).__init__()

        self.bce = tf.keras.losses.BinaryCrossentropy(reduction="none", **kwargs)

        framing_mask = np.zeros(target_shape)
        framing_mask[
            border_size[0] : target_shape[0] - border_size[0],
            border_size[1] : target_shape[1] - border_size[1],
            border_size[2] : target_shape[2] - border_size[2],
        ] = 1

        self.framing_mask = tf.convert_to_tensor(
            framing_mask.astype("bool"), dtype=tf.bool
        )

        self.mask_fn = ft.partial(tf.boolean_mask, mask=self.framing_mask)

    def call(self, y_true, y_pred):
        loss = self.bce(y_true, y_pred)
        loss = tf.map_fn(self.mask_fn, loss)
        loss = tf.reduce_mean(loss)
        return loss

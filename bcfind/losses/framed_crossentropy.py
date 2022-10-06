import tensorflow as tf

from bcfind.losses.utils import get_mask_fn
from bcfind.losses import DiceLoss


class FramedCrossentropy3D(tf.keras.losses.Loss):
    """
    Implementation of binary crossentropy loss for 3D images where the predictions
    at the borders are not included in the computation.
    """

    def __init__(self, border_size, target_shape, from_logits=False, add_dice=False, **kwargs):
        super(FramedCrossentropy3D, self).__init__(**kwargs)

        self.border_size = border_size
        self.target_shape = target_shape
        self.from_logits = from_logits
        self.add_dice = add_dice

        self.bce = tf.keras.losses.BinaryCrossentropy(self.from_logits, reduction='none')
        self.mask_fn = get_mask_fn(self.target_shape, self.border_size)

        if self.add_dice:
            self.dice = DiceLoss(from_logits=self.from_logits)

    def __call__(self, y_true, y_pred, sample_weight=None):
        ce = tf.keras.metrics.binary_crossentropy(
            y_true, 
            y_pred, 
            from_logits=self.from_logits, 
            label_smoothing=0.0, 
            axis=-1
        )
        
        # framed loss
        ce = tf.map_fn(self.mask_fn, ce)
        ce = tf.reduce_mean(ce, axis=tf.range(1, tf.rank(ce)))
        
        # add dice loss
        if self.add_dice:
            dice = self.dice(y_true, y_pred)
            return ce + dice
        else:
            return ce
            # return tf.reduce_mean(ce)

    def get_config(self):
        config = {
            'border_size': self.border_size,
            'target_shape': self.target_shape,
            'from_logits': self.from_logits,
            'add_dice': self.add_dice,
        }
        base_config = super(FramedCrossentropy3D, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


tf.keras.utils.get_custom_objects().update({'FramedCrossentropy3D': FramedCrossentropy3D})
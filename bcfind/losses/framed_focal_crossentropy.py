import tensorflow as tf

from bcfind.losses import DiceLoss
from bcfind.losses.utils import get_mask_fn

class FramedFocalCrossentropy3D(tf.keras.losses.Loss):
    """
    Implementation of binary focal crossentropy loss for 3D images where the predictions
    at the borders are not included in the computation.
    """
    def __init__(self, border_size, target_shape, from_logits=False, gamma=2.0, add_dice=False, **kwargs):
        super(FramedFocalCrossentropy3D, self).__init__(**kwargs)

        self.border_size = border_size
        self.target_shape = target_shape
        self.from_logits = from_logits
        self.gamma = gamma
        self.add_dice = add_dice
        
        self.focal_bce = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=gamma,
            from_logits=self.from_logits,
            reduction='none',
            )

        if self.add_dice:
            self.dice = DiceLoss(self.from_logits)
        
        self.mask_fn = get_mask_fn(target_shape, border_size)

    def call(self, y_true, y_pred):
        ce = self.focal_bce(y_true, y_pred)

        # framed loss
        ce = tf.map_fn(self.mask_fn, ce)
        ce = tf.reduce_mean(ce, axis=tf.range(1, tf.rank(ce)))
        
        # add dice loss
        if self.add_dice:
            dice = self.dice(y_true, y_pred)
            return ce + dice
        else:
            return ce
    
    def get_config(self):
        config = {
            'border_size': self.border_size,
            'target_shape': self.target_shape,
            'from_logits': self.from_logits,
            'gamma': self.gamma,
            'add_dice': self.add_dice,
        }
        base_config = super(FramedFocalCrossentropy3D, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


tf.keras.utils.get_custom_objects().update({'FramedFocalCrossentropy3D': FramedFocalCrossentropy3D})
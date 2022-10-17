import tensorflow as tf

from bcfind.losses import DiceLoss
from bcfind.losses.utils import get_mask_fn


@tf.keras.utils.register_keras_serializable(package='BCFind', name='FramedFocalCrossentropy3D')
class FramedFocalCrossentropy3D(tf.keras.losses.Loss):
    """
    Implementation of binary focal crossentropy loss for 3D images where the predictions
    at the borders are not included in the computation.
    """
    def __init__(self, border_size, target_shape, from_logits=False, gamma=2.0, add_dice=False, reduce=None, **kwargs):
        super(FramedFocalCrossentropy3D, self).__init__(**kwargs)

        self.border_size = border_size
        self.target_shape = target_shape
        self.from_logits = from_logits
        self.gamma = gamma
        self.add_dice = add_dice
        self.reduce = reduce
        
        self.focal_bce = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=gamma,
            from_logits=self.from_logits,
            reduction='none',
            )

        if self.add_dice:
            self.dice = DiceLoss(self.from_logits)
        
        self.mask_fn = get_mask_fn(target_shape, border_size)

    def call(self, y_true, y_pred):
        loss = self.focal_bce(y_true, y_pred)

        # framed loss
        loss = tf.map_fn(self.mask_fn, loss)
        loss = tf.reduce_mean(loss, axis=tf.range(1, tf.rank(loss)))
        
        # add dice loss
        if self.add_dice:
            dice = self.dice(tf.map_fn(self.mask_fn, y_true), tf.map_fn(self.mask_fn, y_pred))
            loss = loss + dice
        
        # reduction
        if self.reduce:
            return tf.reduce_mean(loss, axis=0)
        else:
            return loss
    
    def get_config(self):
        config = {
            'border_size': self.border_size,
            'target_shape': self.target_shape,
            'from_logits': self.from_logits,
            'gamma': self.gamma,
            'add_dice': self.add_dice,
            'reduce': self.reduce,
        }
        base_config = super(FramedFocalCrossentropy3D, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


tf.keras.utils.get_custom_objects().update({'FramedFocalCrossentropy3D': FramedFocalCrossentropy3D})
import tensorflow as tf
import functools as ft
import numpy as np


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


tf.function
def dice_loss(y_true, y_pred, from_logits=False):
    y_true = tf.cast(y_true, y_pred.dtype)

    if from_logits:
        y_pred = tf.sigmoid(y_pred)
    
    numerator = 2 * tf.reduce_sum(y_true * y_pred, tf.range(1, tf.rank(y_pred)))
    denominator = tf.reduce_sum(y_true + y_pred, tf.range(1, tf.rank(y_pred)))
    return 1 - numerator / denominator


class FramedFocalCrossentropy3D(tf.keras.losses.Loss):
    """
    Implementation of binary focal crossentropy loss for 3D images where the predictions
    at the borders are not included in the computation.
    """
    def __init__(self, border_size, target_shape, from_logits=False, gamma=2.0, alpha=None, add_dice=False):
        super(FramedFocalCrossentropy3D, self).__init__()

        self.border_size = border_size
        self.target_shape = target_shape
        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma
        self.add_dice = add_dice

        self.mask_fn = get_mask_fn(target_shape, border_size)

    def call(self, y_true, y_pred):
        ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)

        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        
        # class imbalance smoothing
        if self.alpha is not None:
            alpha = tf.cast(self.alpha, y_true.dtype)
            balance_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        else:
            balance_factor = 1
        
        # focal smoothing
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        gamma = tf.cast(self.gamma, y_true.dtype)
        focal_factor = tf.pow((1.0 - p_t), gamma)

        # weighted loss
        loss = ce * focal_factor * balance_factor

        # add dice loss
        if self.add_dice:
            dice = dice_loss(y_true, y_pred, from_logits=False)
            # broadcast dice to ce (dice loss is computed per image)
            for _ in tf.range(tf.rank(ce) - 1):
                dice = tf.expand_dims(dice, 1)
            
            loss = loss + dice

        # framed loss
        loss = tf.map_fn(self.mask_fn, loss)

        # loss reduction
        loss = tf.reduce_mean(loss)
        return loss
    
    def get_config(self):
        config = {
            'border_size': self.border_size,
            'target_shape': self.target_shape,
            'from_logits': self.from_logits,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'add_dice': self.add_dice,
        }
        return config


class FramedCrossentropy3D(tf.keras.losses.Loss):
    """
    Implementation of binary crossentropy loss for 3D images where the predictions
    at the borders are not included in the computation.
    """

    def __init__(self, border_size, target_shape, from_logits=False, add_dice=False):
        super(FramedCrossentropy3D, self).__init__()

        self.border_size = border_size
        self.target_shape = target_shape
        self.from_logits = from_logits
        self.add_dice = add_dice

        self.mask_fn = get_mask_fn(target_shape, border_size)

    def call(self, y_true, y_pred):
        loss = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)

        # add dice loss
        if self.add_dice:
            dice = dice_loss(y_true, y_pred, from_logits=self.from_logits)
            # broadcast dice to ce (dice loss is computed per image)
            for _ in tf.range(tf.rank(loss) - 1):
                dice = tf.expand_dims(dice, 1)
            
            loss = loss + dice

        # framed loss
        loss = tf.map_fn(self.mask_fn, loss)

        # loss reduction
        loss = tf.reduce_mean(loss)
        return loss
    
    def get_config(self):
        config = {
            'border_size': self.border_size,
            'target_shape': self.target_shape,
            'from_logits': self.from_logits,
            'add_dice': self.add_dice,
        }
        return config
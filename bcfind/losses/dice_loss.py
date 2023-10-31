import tensorflow as tf

from bcfind.utils.losses import get_mask_fn


@tf.keras.utils.register_keras_serializable("BCFind")
class DiceLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        border_size,
        target_shape,
        beta=0.5,
        smooth=1e-5,
        from_logits=False,
        **kwargs
    ):
        super(DiceLoss, self).__init__(**kwargs)
        self.from_logits = from_logits
        self.target_shape = target_shape
        self.border_size = border_size
        self.beta = beta
        self.smooth = smooth
        self.mask_fn = get_mask_fn(self.target_shape, self.border_size)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = tf.map_fn(self.map_fn, y_true)
        y_pred = tf.map_fn(self.mask_fn, y_pred)

        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)

        numerator = tf.reduce_sum(y_true * y_pred) + self.smooth
        denominator = (
            tf.reduce_sum(
                y_true * y_pred
                + self.beta * (1 - y_true) * y_pred
                + (1 - self.beta) * y_true * (1 - y_pred)
            )
            + self.smooth
        )
        return 1 - numerator / denominator

    def get_config(
        self,
    ):
        config = super(DiceLoss).get_config()
        config.update(
            {
                "target_shape": self.target_shape,
                "border_size": self.border_size,
                "from_logits": self.from_logits,
            }
        )
        return config

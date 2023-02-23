import tensorflow as tf


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)

        numerator = 2 * tf.reduce_sum(y_true * y_pred, tf.range(1, tf.rank(y_pred)))
        denominator = tf.reduce_sum(
            y_true * y_true, tf.range(1, tf.rank(y_pred))
        ) + tf.reduce_sum(y_pred * y_pred, tf.range(1, tf.rank(y_pred)))
        return 1 - numerator / denominator

    def get_config(
        self,
    ):
        config = {"from_logits": self.from_logits}
        base_config = super(DiceLoss).get_config()
        return dict(list(config.items()) + list(base_config.items()))


tf.keras.utils.get_custom_objects().update({"DiceLoss": DiceLoss})

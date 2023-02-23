import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="BCFind", name="AttentionGate")
class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

        self.q_dot_k = tf.keras.layers.Dot([1, 1])
        self.q_dot_w = tf.keras.layers.Dot([-1, -1])
        self.softmax = tf.keras.layers.Activation("softmax")

    @tf.function
    def call(self, queries, keys):
        shape = tf.shape(queries)

        q = tf.reshape(queries, [shape[0], -1, shape[-1]])
        k = tf.reshape(keys, [shape[0], -1, shape[-1]])

        weights = self.q_dot_k([q, k])
        weights = self.softmax(weights)

        att = self.q_dot_w([q, weights])
        att = tf.reshape(att, shape)

        outputs = queries + att
        return outputs

    def get_config(
        self,
    ):
        return super(AttentionGate, self).get_config()

import tensorflow as tf

from bcfind.layers.switch_normalization import SwitchNormalization
from bcfind.layers.attention_gate import AttentionGate


@tf.keras.utils.register_keras_serializable(package="BCFind", name="DecoderBlock")
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_filters,
        k_size,
        k_stride,
        regularizer=None,
        normalization="batch",
        activation="relu",
        attention=False,
        **kwargs,
    ):
        super(DecoderBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.k_size = k_size
        self.k_stride = k_stride
        self.regularizer = regularizer
        self.normalization = normalization
        self.activaton = activation
        self.attention = attention

        self.conv3D_T = tf.keras.layers.Conv3DTranspose(
            filters=self.n_filters,
            kernel_size=self.k_size,
            strides=self.k_stride,
            padding="same",
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
        )
        if self.normalization == "batch":
            self.norm = tf.keras.layers.BatchNormalization()
        elif self.normalization == "switch":
            self.norm = SwitchNormalization()
        elif self.normalization == "instance":
            self.norm == tf.keras.layers.UnitNormalization()
        elif self.normalization == "layer":
            self.norm = tf.keras.layers.LayerNormalization()

        self.activ = tf.keras.layers.Activation(activation)

        if self.attention:
            self.attn_gate = AttentionGate()

    def build(self, input_shape):
        self.conv3D_T.build(input_shape)
        self.norm.build(self.conv3D_T.compute_output_shape(input_shape))
        self.activ.build(self.conv3D_T.compute_output_shape(input_shape))
        if self.attention:
            self.attn_gate.build(self.conv3D_T.compute_output_shape(input_shape))

        self.built = True

    def call(self, inputs, to_concatenate_layer=None, training=None):
        h = self.conv3D_T(inputs)
        h = self.norm(h, training=training)
        h = self.activ(h)

        if to_concatenate_layer is not None:
            if self.attention:
                to_concatenate_layer = self.attn_gate(to_concatenate_layer, h)

            h = tf.keras.layers.concatenate([h, to_concatenate_layer])

        return h

    def get_config(
        self,
    ):
        config = {
            "n_filters": self.n_filters,
            "k_size": self.k_size,
            "k_stride": self.k_stride,
            "regularizer": self.regularizer,
            "normalization": self.normalization,
            "activation": self.activaton,
            "attention": self.attention,
        }
        base_config = super(DecoderBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

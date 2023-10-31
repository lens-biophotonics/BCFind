import tensorflow as tf

from .switch_normalization import SwitchNormalization


@tf.keras.utils.register_keras_serializable("BCFind")
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_filters,
        k_size,
        k_stride=1,
        regularizer=None,
        normalization="batch",
        activation="relu",
        **kwargs
    ):
        super(EncoderBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.k_size = k_size
        self.k_stride = k_stride
        self.regularizer = regularizer
        self.normalization = normalization
        self.activation = activation

        self.conv3D = tf.keras.layers.Conv3D(
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

        self.activ = tf.keras.layers.Activation(self.activation)

    def build(self, input_shape):
        self.conv3D.build(input_shape)
        self.norm.build(self.conv3D.compute_output_shape(input_shape))
        self.activ.build(self.conv3D.compute_output_shape(input_shape))
        self.built = True

    def call(self, inputs, training=None):
        h = self.conv3D(inputs)
        h = self.norm(h, training=training)
        h = self.activ(h)
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
            "activation": self.activation,
        }
        base_config = super(EncoderBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

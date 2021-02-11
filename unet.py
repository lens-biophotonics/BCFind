import tensorflow as tf


# Encoder block
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, conv_size, conv_stride, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.conv_size = conv_size
        self.conv_stride = conv_stride

        self.conv3D = tf.keras.layers.Conv3D(
            filters=n_filters,
            kernel_size=conv_size,
            strides=conv_stride,
            padding="same",
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation("relu")

    def call(self, inputs, training=None):
        h = self.conv3D(inputs)
        h = self.batch_norm(h, training=training)
        h = self.relu(h)
        return h

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update(
            {
                "n_filters": self.n_filters,
                "conv_size": self.conv_size,
                "conv_stride": self.conv_stride,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Decoder block
class InvConvBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, conv_size, conv_stride, activation, **kwargs):
        super(InvConvBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.conv_size = conv_size
        self.conv_stride = conv_stride
        self.activation = activation

        self.conv3D_T = tf.keras.layers.Conv3DTranspose(
            filters=n_filters,
            kernel_size=conv_size,
            strides=conv_stride,
            padding="same",
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activ = tf.keras.layers.Activation(activation)
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs, feat_concat=None, activation=True, training=None):
        h = self.conv3D_T(inputs)
        h = self.batch_norm(h, training=training)
        if activation:
            h = self.activ(h)
        if feat_concat is not None:
            h = self.concat([h, feat_concat])
        return h

    def get_config(self):
        config = super(InvConvBlock, self).get_config()
        config.update(
            {
                "n_filters": self.n_filters,
                "conv_size": self.conv_size,
                "conv_stride": self.conv_stride,
                "activation": self.activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UNet(tf.keras.Model):
    def __init__(self, n_filters, e_size, e_stride, d_size, d_stride, **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.e_size = e_size
        self.e_stride = e_stride
        self.d_size = d_size
        self.d_stride = d_stride

        # Encoder
        self.conv_block_1 = ConvBlock(n_filters, e_size, e_stride)
        self.conv_block_2 = ConvBlock(n_filters * 2, e_size, e_stride)
        self.conv_block_3 = ConvBlock(n_filters * 4, e_size, (1, 1, 1))
        self.conv_block_4 = ConvBlock(n_filters * 8, e_size, (1, 1, 1))

        # Decoder
        self.inv_conv_block_1 = InvConvBlock(n_filters * 4, d_size, (1, 1, 1), "relu")
        self.inv_conv_block_2 = InvConvBlock(n_filters * 2, d_size, (1, 1, 1), "relu")
        self.inv_conv_block_3 = InvConvBlock(n_filters, d_size, d_stride, "relu")
        self.inv_conv_block_4 = InvConvBlock(1, d_size, d_stride, "sigmoid")

    def call(self, inputs, training=None):
        h1 = self.conv_block_1(inputs, training=training)
        h2 = self.conv_block_2(h1, training=training)
        h3 = self.conv_block_3(h2, training=training)
        h = self.conv_block_4(h3, training=training)

        h = self.inv_conv_block_1(h, feat_concat=h3, training=training)
        h = self.inv_conv_block_2(h, feat_concat=h2, training=training)
        h = self.inv_conv_block_3(h, feat_concat=h1, training=training)
        h = self.inv_conv_block_4(h, activation=False, training=training)
        return h

    def get_config(self):
        config = super(UNet, self).get_config()
        config.update(
            {
                "n_filters": self.n_filters,
                "e_size": self.e_size,
                "e_stride": self.e_stride,
                "d_size": self.d_size,
                "d_stride": self.d_stride,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

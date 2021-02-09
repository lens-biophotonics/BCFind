import tensorflow as tf


# Encoder block
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, conv_size, conv_stride, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

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
        if training:
            h = self.batch_norm(h)
        h = self.relu(h)
        return h


# Decoder block
class InvConvBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, conv_size, conv_stride, activation, **kwargs):
        super(InvConvBlock, self).__init__(**kwargs)

        self.conv3D_T = tf.keras.layers.Conv3DTranspose(
            filters=n_filters,
            kernel_size=conv_size,
            strides=conv_stride,
            padding="same",
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activ = tf.keras.layers.Activation(activation)
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs, feat_concat=None, training=None):
        h = self.conv3D_T(inputs)
        if training:
            h = self.batch_norm(h)
        h = self.activ(h)
        if feat_concat is not None:
            h = self.concat([h, feat_concat])
        return h


class UNet(tf.keras.Model):
    def __init__(self, n_filters, e_size, e_stride, d_size, d_stride, **kwargs):
        super(UNet, self).__init__(**kwargs)

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

    def call(self, inputs, **kwargs):
        h1 = self.conv_block_1(inputs, **kwargs)
        h2 = self.conv_block_2(h1, **kwargs)
        h3 = self.conv_block_3(h2, **kwargs)
        h = self.conv_block_4(h3, **kwargs)

        h = self.inv_conv_block_1(h, feat_concat=h3, **kwargs)
        h = self.inv_conv_block_2(h, feat_concat=h2, **kwargs)
        h = self.inv_conv_block_3(h, feat_concat=h1, **kwargs)
        h = self.inv_conv_block_4(h, **kwargs)
        return h

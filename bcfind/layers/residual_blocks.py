import tensorflow as tf

from .switch_normalization import SwitchNormalization
from .attention_gate import AttentionGate


@tf.keras.utils.register_keras_serializable("BCFind")
class ResBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_filters,
        k_size,
        regularizer=None,
        normalization="batch",
        activation="relu",
        **kwargs,
    ):
        super(ResBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.k_size = k_size
        self.regularizer = regularizer
        self.normalization = normalization
        self.activation = activation

        if self.normalization == "batch":
            self.norm_1 = tf.keras.layers.BatchNormalization()
            self.norm_2 = tf.keras.layers.BatchNormalization()
        elif self.normalization == "switch":
            self.norm_1 = SwitchNormalization()
            self.norm_2 = SwitchNormalization()
        elif self.normalization == "instance":
            self.norm_1 == tf.keras.layers.UnitNormalization()
            self.norm_2 == tf.keras.layers.UnitNormalization()
        elif self.normalization == "layer":
            self.norm_1 = tf.keras.layers.LayerNormalization()
            self.norm_2 = tf.keras.layers.LayerNormalization()

        self.activ_1 = tf.keras.layers.Activation(self.activation)
        self.activ_2 = tf.keras.layers.Activation(self.activation)

        self.conv3D_1 = tf.keras.layers.Conv3D(
            filters=self.n_filters,
            kernel_size=self.k_size,
            strides=1,
            padding="same",
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
        )

        self.conv3D_2 = tf.keras.layers.Conv3D(
            filters=self.n_filters,
            kernel_size=self.k_size,
            strides=1,
            padding="same",
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
        )

    def build(self, input_shape):
        self.norm_1.build(input_shape)
        self.activ_1.build(input_shape)
        self.conv3D_1.build(input_shape)
        out_shape_1 = self.conv3D_1.compute_output_shape(input_shape)

        self.norm_2.build(out_shape_1)
        self.activ_2.build(out_shape_1)
        self.conv3D_2.build(out_shape_1)
        out_shape_2 = self.conv3D_2.compute_output_shape(out_shape_1)

        if all([input_shape[i] == out_shape_2[i] for i in range(len(out_shape_2))]):
            self.skip_conv3D = None
        else:
            self.skip_conv3D = tf.keras.layers.Conv3D(
                filters=self.n_filters,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_regularizer=self.regularizer,
                bias_regularizer=self.regularizer,
            )
            self.skip_conv3D.build(input_shape)

        self.built = True

    def call(self, inputs, training=None):
        h = self.norm_1(inputs, training=training)
        h = self.activ_1(h)
        h = self.conv3D_1(h)

        h = self.norm_2(h, training=training)
        h = self.activ_2(h)
        h = self.conv3D_2(h)

        if self.skip_conv3D is not None:
            s = self.skip_conv3D(inputs)
        else:
            s = tf.identity(inputs)

        return h + s


@tf.keras.utils.register_keras_serializable("BCFind")
class ResidualEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_filters,
        k_size,
        downsample=True,
        down_stride=(2, 2, 2),
        regularizer=None,
        normalization="batch",
        activation="relu",
        **kwargs,
    ):
        super(ResidualEncoderBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.k_size = k_size
        self.downsample = downsample
        self.down_stride = down_stride
        self.regularizer = regularizer
        self.normalization = normalization
        self.activation = activation

        if self.downsample:
            self.max_pool = tf.keras.layers.MaxPool3D(
                self.k_size, strides=self.down_stride, padding="same"
            )

        self.res_block_1 = ResBlock(
            self.n_filters,
            self.k_size,
            self.regularizer,
            self.normalization,
            self.activation,
        )
        self.res_block_2 = ResBlock(
            self.n_filters,
            self.k_size,
            self.regularizer,
            self.normalization,
            self.activation,
        )

        self.res_block_3 = ResBlock(
            self.n_filters,
            self.k_size,
            self.regularizer,
            self.normalization,
            self.activation,
        )

    def build(self, input_shape):
        if self.downsample:
            self.max_pool.build(input_shape)
            out_shape_1 = self.max_pool.compute_output_shape(input_shape)
        else:
            out_shape_1 = input_shape

        self.res_block_1.build(out_shape_1)
        out_shape_2 = self.res_block_1.compute_output_shape(out_shape_1)
        self.res_block_2.build(out_shape_2)
        self.res_block_3.build(self.res_block_2.compute_output_shape(out_shape_2))
        self.built = True

    def call(self, inputs, training=None):
        if self.downsample:
            h = self.max_pool(inputs)
        else:
            h = tf.identity(inputs)

        h = self.res_block_1(h, training=training)
        h = self.res_block_2(h, training=training)
        h = self.res_block_3(h, training=training)
        return h

    def get_config(
        self,
    ):
        config = {
            "n_filters": self.n_filters,
            "k_size": self.k_size,
            "downsample": self.downsample,
            "down_stride": self.down_stride,
            "regularizer": self.regularizer,
            "normalization": self.normalization,
            "activation": self.activation,
        }
        base_config = super(ResidualEncoderBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="BCFind")
class ResidualDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_filters,
        k_size,
        upsample=True,
        up_stride=(2, 2, 2),
        regularizer=None,
        normalization="batch",
        activation="relu",
        attention=False,
        **kwargs,
    ):
        super(ResidualDecoderBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.k_size = k_size
        self.upsample = upsample
        self.up_stride = up_stride
        self.regularizer = regularizer
        self.normalization = normalization
        self.activation = activation
        self.attention = attention

        if self.upsample:
            if self.normalization == "batch":
                self.up_norm = tf.keras.layers.BatchNormalization()
            elif self.normalization == "switch":
                self.up_norm = SwitchNormalization()
            elif self.normalization == "instance":
                self.up_norm == tf.keras.layers.UnitNormalization()
            elif self.normalization == "layer":
                self.up_norm = tf.keras.layers.LayerNormalization()

            self.up_activ = tf.keras.layers.Activation(self.activation)
            self.conv3D_T = tf.keras.layers.Conv3DTranspose(
                filters=self.n_filters,
                kernel_size=self.k_size,
                strides=self.up_stride,
                padding="same",
                kernel_regularizer=self.regularizer,
                bias_regularizer=self.regularizer,
            )

        if self.attention:
            self.attn_gate = AttentionGate()
        self.concat = tf.keras.layers.Concatenate()

        self.res_block_1 = ResBlock(
            self.n_filters,
            self.k_size,
            self.regularizer,
            self.normalization,
            self.activation,
        )

        self.res_block_2 = ResBlock(
            self.n_filters,
            self.k_size,
            self.regularizer,
            self.normalization,
            self.activation,
        )

        self.res_block_3 = ResBlock(
            self.n_filters,
            self.k_size,
            self.regularizer,
            self.normalization,
            self.activation,
        )

    def build(self, input_shape):
        if self.upsample:
            self.up_norm.build(input_shape)
            self.up_activ.build(input_shape)
            self.conv3D_T.build(input_shape)
            out_shape_1 = self.conv3D_T.compute_output_shape(input_shape)
        else:
            out_shape_1 = input_shape

        if self.attention:
            self.attn_gate.build(out_shape_1)
        self.concat.build((out_shape_1, out_shape_1))
        out_shape_1 = self.concat.compute_output_shape((out_shape_1, out_shape_1))

        self.res_block_1.build(out_shape_1)
        out_shape_2 = self.res_block_1.compute_output_shape(out_shape_1)
        self.res_block_2.build(out_shape_2)
        self.res_block_3.build(self.res_block_2.compute_output_shape(out_shape_2))

        self.built = True

    def call(self, inputs, to_concatenate_layer, training=None):
        if self.upsample:
            h = self.up_norm(inputs)
            h = self.up_activ(h)
            h = self.conv3D_T(h)
        else:
            h = tf.identity(inputs)

        if self.attention:
            to_concatenate_layer = self.attn_gate(to_concatenate_layer, h)
        h = self.concat([h, to_concatenate_layer])

        h = self.res_block_1(h)
        h = self.res_block_2(h)
        h = self.res_block_3(h)
        return h

    def get_config(
        self,
    ):
        config = {
            "n_filters": self.n_filters,
            "k_size": self.k_size,
            "upsample": self.upsample,
            "up_stride": self.up_stride,
            "regularizer": self.regularizer,
            "normalization": self.normalization,
            "activation": self.activation,
            "attention": self.attention,
        }
        base_config = super(ResidualDecoderBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

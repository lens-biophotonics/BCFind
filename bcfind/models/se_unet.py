import tensorflow as tf

from bcfind.layers import EncoderBlock, DecoderBlock, SqueezeAndExcite


class SEUNet(tf.keras.Model):
    def __init__(
        self,
        n_blocks,
        n_filters,
        k_size,
        k_stride,
        squeeze_factor=2,
        dropout=None,
        regularizer=None,
        **kwargs
    ):
        """Constructor method.

        Parameters
        ----------
        n_blocks : int
            depth of the UNet encoder
        n_filters : int
            number of filters for the first layer. Consecutive layers increase esponentially their number of filters.
        k_size : int or tuple of ints
            size of the kernel for convolutional layers
        k_stride : int or tuple of ints
            stride for the convolutional layers. The last two encoding and the first two decoding layers will however have no stride.
        squeeze_factor: int
            channel reduction factor in the squeeze module, by default 2.
        dropout : float, optional
            dropout rate to add after each convolutional block, by default None.
        regularizer : string or tf.keras.regularizers, optional
            a regularization method for keras layers, by default None.
        """
        super(SEUNet, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.k_size = k_size
        self.k_stride = k_stride
        self.squeeze_factor = squeeze_factor
        self.dropout = dropout
        self.regularizer = regularizer

        # Input channel expansion
        self.conv_block_1 = EncoderBlock(
            n_filters=self.n_filters,
            k_size=self.k_size,
            k_stride=(1, 1, 1),
            regularizer=self.regularizer,
            normalization="batch",
            activation="relu",
        )
        self.se_1 = SqueezeAndExcite(self.n_filters, self.squeeze_factor)

        # Encoder
        self.encoder_blocks = []
        self.encoder_se = []
        for i in range(self.n_blocks):
            if i >= self.n_blocks - 2:  # last two blocks have no stride
                encoder_block = EncoderBlock(
                    n_filters=self.n_filters * (2 ** (i + 1)),
                    k_size=self.k_size,
                    k_stride=(1, 1, 1),
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )
            else:
                encoder_block = EncoderBlock(
                    n_filters=self.n_filters * (2 ** (i + 1)),
                    k_size=self.k_size,
                    k_stride=self.k_stride,
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )

            se = SqueezeAndExcite(self.n_filters * (2 ** (i + 1)), self.squeeze_factor)

            self.encoder_blocks.append(encoder_block)
            self.encoder_se.append(se)

        # Decoder
        self.decoder_blocks = []
        self.decoder_se = []
        for i in range(self.n_blocks):
            if i < 2:  # first two blocks have no stride
                decoder_block = DecoderBlock(
                    n_filters=self.n_filters * (2 ** (self.n_blocks - i - 1)),
                    k_size=self.k_size,
                    k_stride=(1, 1, 1),
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )
            else:
                decoder_block = DecoderBlock(
                    n_filters=self.n_filters * (2 ** (self.n_blocks - i - 1)),
                    k_size=self.k_size,
                    k_stride=self.k_stride,
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )

            se = SqueezeAndExcite(
                self.n_filters * (2 ** (self.n_blocks - i - 1)) * 2,
                self.squeeze_factor,
            )
            self.decoder_se.append(se)
            self.decoder_blocks.append(decoder_block)

        # Maybe dropout layers
        if dropout:
            self.dropouts = []
            for i in range(self.n_blocks * 2 - 1):
                if i == 0:
                    drp = tf.keras.layers.SpatialDropout3D(dropout / 2)
                    self.dropouts.append(drp)
                else:
                    drp = tf.keras.layers.SpatialDropout3D(dropout)
                    self.dropouts.append(drp)

        # Last predictor layer
        self.predictor = DecoderBlock(
            n_filters=1,
            k_size=self.k_size,
            k_stride=(1, 1, 1),
            regularizer=None,
            normalization="batch",
            activation="linear",
        )

    def call(self, inputs, training=None):
        h0 = self.conv_block_1(inputs)
        h0 = self.se_1(h0)

        encodings = []
        for i_e in range(len(self.encoder_blocks)):
            if i_e == 0:
                h = self.encoder_blocks[i_e](h0, training=training)
            else:
                h = self.encoder_blocks[i_e](h, training=training)

            h = self.encoder_se[i_e](h)

            if self.dropout:
                h = self.dropouts[i_e](h, training=training)

            encodings.append(h)

        for i_d in range(len(self.decoder_blocks)):
            if i_d == 0:
                h = self.decoder_blocks[i_d](
                    encodings[-1], encodings[-2], training=training
                )
                h = self.decoder_se[i_d](h)
            elif i_d < self.n_blocks - 1:
                h = self.decoder_blocks[i_d](h, encodings[-i_d - 2], training=training)
                h = self.decoder_se[i_d](h)
            elif i_d == self.n_blocks - 1:
                h = self.decoder_blocks[i_d](h, h0, training=training)

            if self.dropout:
                h = self.dropouts[i_e + i_d](h, training=training)

        pred = self.predictor(h, training=training)
        return pred

    def get_config(
        self,
    ):
        config = {
            "n_blocks": self.n_blocks,
            "n_filters": self.n_filters,
            "k_size": self.k_size,
            "k_stride": self.k_stride,
            "squeeze_factor": self.squeeze_factor,
            "dropout": self.dropout,
            "regularizer": self.regularizer,
        }
        base_config = super(SEUNet, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


if __name__ == "__main__":
    unet = SEUNet(4, 32, 3, 2)
    unet.build((None, None, None, None, 1))
    unet.summary()

    x = tf.random.normal((4, 48, 48, 48, 1))
    pred = unet(x, training=False)
    print(pred.shape)

    unet.save("prova.tf")
    del unet

    unet = tf.keras.models.load_model("prova.tf")
    unet.build((None, None, None, None, 1))
    unet.summary()

    x = tf.random.normal((4, 48, 100, 100, 1))
    pred = unet(x, training=False)
    print(pred.shape)

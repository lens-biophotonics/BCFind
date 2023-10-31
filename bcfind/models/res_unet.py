import tensorflow as tf

from bcfind.layers import ResBlock, ResidualEncoderBlock, ResidualDecoderBlock


class ResUNet(tf.keras.Model):
    """Class for 3D Res-UNet model.


    Refers to:
        - O. Ronneberger et al. 'UNet: Convolutional Networks for Biomedical Image Segmenation <https://arxiv.org/pdf/1505.04597.pdf>'
    """

    def __init__(
        self,
        n_blocks,
        n_filters,
        k_size,
        k_stride,
        dropout=None,
        regularizer=None,
        mult_skip=False,
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
        dropout : bool, optional
            whether or not to add dropout layer after each convolutional block, by default None.
        regularizer : string or tf.keras.regularizers, optional
            a regularization method for keras layers, by default None.
        """
        super(ResUNet, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.k_size = k_size
        self.k_stride = k_stride
        self.dropout = dropout
        self.regularizer = regularizer
        self.mult_skip = mult_skip

        # Inputs channel expansion
        self.channel_expansion = tf.keras.layers.Conv3D(
            filters=self.n_filters,
            kernel_size=(s * 2 + 1 for s in self.k_size),
            strides=(1, 1, 1),
            padding="same",
        )
        self.initial_block = ResBlock(
            n_filters=self.n_filters,
            k_size=self.k_size,
            regularizer=self.regularizer,
            normalization="batch",
            activation="relu",
        )

        # Encoder
        self.encoder_blocks = []
        for i in range(self.n_blocks):
            if i >= self.n_blocks - 2:  # last two blocks have no stride
                encoder_block = ResidualEncoderBlock(
                    n_filters=self.n_filters * (2 ** (i + 1)),
                    k_size=self.k_size,
                    downsample=False,
                    down_stride=(1, 1, 1),
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )
            else:
                encoder_block = ResidualEncoderBlock(
                    n_filters=self.n_filters * (2 ** (i + 1)),
                    k_size=self.k_size,
                    downsample=True,
                    down_stride=self.k_stride,
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )

            self.encoder_blocks.append(encoder_block)

        # Decoder
        self.decoder_blocks = []
        for i in range(self.n_blocks):
            if i < 2:  # first two blocks have no stride
                decoder_block = ResidualDecoderBlock(
                    n_filters=self.n_filters * (2 ** (self.n_blocks - i - 1)),
                    k_size=self.k_size,
                    upsample=True,
                    up_stride=(1, 1, 1),
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )
            else:
                decoder_block = ResidualDecoderBlock(
                    n_filters=self.n_filters * (2 ** (self.n_blocks - i - 1)),
                    k_size=self.k_size,
                    upsample=True,
                    up_stride=self.k_stride,
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )

            self.decoder_blocks.append(decoder_block)

        # Maybe dropout layers
        if dropout:
            self.dropouts = []
            for i in range(self.n_blocks * 2 + 1):
                drpt = tf.keras.layers.SpatialDropout3D(dropout)
                self.dropouts.append(drpt)

        # Last predictor layer
        self.pred_norm = tf.keras.layers.BatchNormalization()
        self.pred_activ = tf.keras.layers.Activation("relu")
        self.pred_conv = tf.keras.layers.Conv3D(
            filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same"
        )
        # self.predictor = ResBlock(
        #     n_filters=1,
        #     k_size=(1, 1, 1),
        #     regularizer=None,
        #     normalization="batch",
        #     activation="relu",
        # )

    def call(self, inputs, training=None):
        # Input channel expansion
        h1 = self.channel_expansion(inputs)
        h1 = self.initial_block(h1, training=training)
        if self.dropout:
            h1 = self.dropouts[0](h1, training=training)

        # Encoder
        encodings = []
        for i_e, encoder_block in enumerate(self.encoder_blocks):
            if i_e == 0:
                h = encoder_block(h1, training=training)
            else:
                h = encoder_block(h, training=training)

            if self.dropout:
                h = self.dropouts[i_e + 1](h, training=training)

            encodings.append(h)

        # Decoder
        for i_d, decoder_block in enumerate(self.decoder_blocks):
            if i_d == 0:
                h = decoder_block(encodings[-1], encodings[-2], training=training)
            elif i_d < self.n_blocks - 1:
                h = decoder_block(h, encodings[-i_d - 2], training=training)
            elif i_d == self.n_blocks - 1:
                h = decoder_block(h, h1, training=training)

            if self.dropout:
                h = self.dropouts[i_e + 2 + i_d](h, training=training)

        # Predictor
        h = self.pred_norm(h, training=training)
        h = self.pred_activ(h)
        pred = self.pred_conv(h, training=training)
        # pred = self.predictor(h, training=training)

        if self.mult_skip:
            pred = pred * inputs  # prova
        return pred

    def get_config(
        self,
    ):
        config = super(ResUNet, self).get_config()
        config.update(
            {
                "n_blocks": self.n_blocks,
                "n_filters": self.n_filters,
                "k_size": self.k_size,
                "k_stride": self.k_stride,
                "dropout": self.dropout,
                "regularizer": self.regularizer,
                "mult_skip": self.mult_skip,
            }
        )
        return config


if __name__ == "__main__":
    unet = ResUNet(4, 32, 3, 2)
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

    x = tf.random.normal((4, 48, 96, 96, 1))
    pred = unet(x, training=False)
    print(pred.shape)

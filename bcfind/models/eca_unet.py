import tensorflow as tf

from bcfind.layers.encoder_block import EncoderBlock
from bcfind.layers.decoder_block import DecoderBlock
from bcfind.layers.efficient_channel_attention import (
    EfficientChannelAttention,
    _get_eca_kernel_size,
)


class ECAUNet(tf.keras.Model):
    """Class for UNet model with Efficient Channel Attention module after each convolutional layer.

    Refers to:\n
        - 'Q. Wang et al. "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks" <https://arxiv.org/pdf/1910.03151.pdf>'
    """

    def __init__(
        self,
        n_blocks,
        n_filters,
        k_size,
        k_stride,
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
        dropout : bool, optional
            whether or not to add dropout layer after each convolutional block, by default None.
        regularizer : string or tf.keras.regularizers, optional
            a regularization method for keras layers, by default None.
        """
        super(ECAUNet, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.k_size = k_size
        self.k_stride = k_stride
        self.dropout = dropout
        self.regularizer = regularizer

        # Encoder
        self.encoder_blocks = []
        self.encoder_eca = []
        for i in range(self.n_blocks):
            if i >= self.n_blocks - 2:  # last two blocks have no stride
                encoder_block = EncoderBlock(
                    n_filters=self.n_filters * (2**i),
                    k_size=self.k_size,
                    k_stride=(1, 1, 1),
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )
            else:
                encoder_block = EncoderBlock(
                    n_filters=self.n_filters * (2**i),
                    k_size=self.k_size,
                    k_stride=self.k_stride,
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )

            eca = EfficientChannelAttention(
                _get_eca_kernel_size(self.n_filters * (2**i))
            )

            self.encoder_blocks.append(encoder_block)
            self.encoder_eca.append(eca)

        # Decoder
        self.decoder_blocks = []
        self.decoder_eca = []
        for i in range(self.n_blocks):
            if i < 2:  # first two blocks have no stride
                decoder_block = DecoderBlock(
                    n_filters=self.n_filters * (2 ** (self.n_blocks - i - 2)),
                    k_size=self.k_size,
                    k_stride=(1, 1, 1),
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )
            elif i < self.n_blocks - 1:
                decoder_block = DecoderBlock(
                    n_filters=self.n_filters * (2 ** (self.n_blocks - i - 2)),
                    k_size=self.k_size,
                    k_stride=self.k_stride,
                    regularizer=self.regularizer,
                    normalization="batch",
                    activation="relu",
                )
            elif (
                i == self.n_blocks - 1
            ):  # last block have only one filter and no regularization
                decoder_block = DecoderBlock(
                    n_filters=1,
                    k_size=self.k_size,
                    k_stride=self.k_stride,
                    regularizer=None,
                    normalization="batch",
                )

            if i < self.n_blocks - 1:  # last block has no ECA module
                eca = EfficientChannelAttention(
                    _get_eca_kernel_size(
                        self.n_filters * (2 ** (self.n_blocks - i - 2)) * 2
                    )
                )
                self.decoder_eca.append(eca)

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
        encodings = []
        for i_e in range(len(self.encoder_blocks)):
            if i_e == 0:
                h = self.encoder_blocks[i_e](inputs, training=training)
            else:
                h = self.encoder_blocks[i_e](h, training=training)

            h = self.encoder_eca[i_e](h)

            if self.dropout:
                h = self.dropouts[i_e](h, training=training)

            encodings.append(h)

        for i_d in range(len(self.decoder_blocks)):
            if i_d == 0:
                h = self.decoder_blocks[i_d](
                    encodings[-1], encodings[-2], training=training
                )
                h = self.decoder_eca[i_d](h)
            elif i_d < self.n_blocks - 1:
                h = self.decoder_blocks[i_d](h, encodings[-i_d - 2], training=training)
                h = self.decoder_eca[i_d](h)
            elif i_d == self.n_blocks - 1:
                h = self.decoder_blocks[i_d](h, inputs, training=training)

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
            "dropout": self.dropout,
            "regularizer": self.regularizer,
        }
        base_config = super(ECAUNet, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))

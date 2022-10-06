import tensorflow as tf

from bcfind.layers.encoder_block import EncoderBlock
from bcfind.layers.decoder_block import DecoderBlock


class AttentionUNet(tf.keras.Model):
    def __init__(self, n_blocks, n_filters, k_size, k_stride, dropout=None, regularizer=None, **kwargs):
        super(AttentionUNet, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.n_filters = n_filters 
        self.k_size = k_size
        self.k_stride = k_stride 
        self.dropout = dropout
        self.regularizer = regularizer

        # Encoder
        self.encoder_blocks = []
        for i in range(self.n_blocks):
            if i >= self.n_blocks - 2:  # last two blocks have no stride
                encoder_block = EncoderBlock(
                    n_filters=self.n_filters * (2**i), 
                    k_size=self.k_size, 
                    k_stride=(1, 1, 1), 
                    regularizer=self.regularizer, 
                    normalization='batch', 
                    activation='relu'
                    )
            else:
                encoder_block = EncoderBlock(
                    n_filters=self.n_filters * (2**i), 
                    k_size=self.k_size, 
                    k_stride=self.k_stride, 
                    regularizer=self.regularizer, 
                    normalization='batch', 
                    activation='relu'
                    )
            
            self.encoder_blocks.append(encoder_block)
        
        # Decoder
        self.decoder_blocks = []
        for i in range(self.n_blocks):
            if i < 2:  # first two blocks have no stride
                decoder_block = DecoderBlock(
                    n_filters=self.n_filters * (2 ** (self.n_blocks - i - 2)), 
                    k_size=self.k_size, 
                    k_stride=(1, 1, 1), 
                    regularizer=self.regularizer, 
                    normalization='batch', 
                    activation='relu', 
                    attention=True
                    )
            elif i < self.n_blocks - 1:
                decoder_block = DecoderBlock(
                    n_filters=self.n_filters * (2 ** (self.n_blocks - i - 2)), 
                    k_size=self.k_size, 
                    k_stride=self.k_stride, 
                    regularizer=self.regularizer, 
                    normalization='batch', 
                    activation='relu', 
                    attention=True
                    )
            elif i == self.n_blocks - 1:  # last block have only one filter and no regularization
                decoder_block = DecoderBlock(
                    n_filters=1, 
                    k_size=self.k_size, 
                    k_stride=self.k_stride, 
                    regularizer=None, 
                    normalization='batch'
                    )
            
            self.decoder_blocks.append(decoder_block)                
        
        # Maybe dropout layers
        if dropout:
            self.dropouts = []
            for i in range(self.n_blocks * 2 - 1):
                if i == 0:
                    drp = tf.keras.layers.Dropout(dropout / 2)
                    self.dropouts.append(drp)
                else:
                    drp = tf.keras.layers.Dropout(dropout)
                    self.dropouts.append(drp)
        
        # Last predictor layer
        self.predictor = DecoderBlock(
            n_filters=1, 
            k_size=self.k_size, 
            k_stride=(1, 1, 1), 
            regularizer=None, 
            normalization='batch', 
            activation='linear'
            )

    def call(self, inputs, training=None):
        encodings = []
        for i_e, encoder_block in enumerate(self.encoder_blocks):
            if i_e == 0:
                h = encoder_block(inputs, training=training)
            else:
                h = encoder_block(h, training=training)
            
            if self.dropout:
                h = self.dropouts[i_e](h, training=training)
    
            encodings.append(h)
        
        for i_d, decoder_block in enumerate(self.decoder_blocks):
            if i_d == 0:
                h = decoder_block(encodings[-1], encodings[-2], training=training)
            elif i_d < self.n_blocks - 1:
                h = decoder_block(h, encodings[- i_d - 2], training=training)
            elif i_d == self.n_blocks - 1:
                h = decoder_block(h, inputs, training=training)                
            
            if self.dropout:
                h = self.dropouts[i_e + i_d](h, training=training)
            
        pred = self.predictor(h, training=training)
        return pred
    
    def get_config(self, ):
        config = {
            'n_blocks': self.n_blocks,
            'n_filters': self.n_filters,
            'k_size': self.k_size,
            'k_stride': self.k_stride,
            'dropout': self.dropout,
            'regularizer': self.regularizer,
        }
        base_config = super(AttentionUNet, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


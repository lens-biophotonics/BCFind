import tensorflow as tf
import numpy as np


def _combine_layers(input_layer, layerlist: list):
    """
    combines a list of sequential layers into a single one

    Parameters
    ----------
    input_layer : keras layer
        input_layer
    layerlist : list
        list of layers to stack

    Returns
    -------
    combined_layer : keras_layer
        stacked layers
    """

    layer_in = input_layer
    for layerfunc in layerlist:
        layer_in = layerfunc(layer_in)

    return layer_in


def _encoder_block(input_layer, n_filters, conv_size, conv_stride):
    conv3D = tf.keras.layers.Conv3D(
        filters=n_filters,
        kernel_size=conv_size,
        strides=conv_stride,
        padding='same',
    )
    batch_norm = tf.keras.layers.BatchNormalization()
    relu = tf.keras.layers.Activation('relu')

    layer_list = [
        conv3D,
        batch_norm,
        relu,
    ]

    layer_out = _combine_layers(input_layer, layer_list)
    return layer_out


def _decoder_block(input_layer, to_concatenate_layer, n_filters, conv_size, conv_stride, activation):
    conv3D_T = tf.keras.layers.Conv3DTranspose(
        filters=n_filters,
        kernel_size=conv_size,
        strides=conv_stride,
        padding='same',
    )

    batch_norm = tf.keras.layers.BatchNormalization()
    activ = tf.keras.layers.Activation(activation)

    layer_list = [
        conv3D_T,
        batch_norm,
        activ,
    ]

    layer_out = _combine_layers(input_layer, layer_list)

    if to_concatenate_layer is not None:
        layer_out = tf.keras.layers.concatenate([layer_out, to_concatenate_layer])

    return layer_out


def _attention_gate(g, x, stride):
    n_channels = x.shape[-1]
    
    phi_g = tf.keras.layers.Conv3D(n_channels, 1, 1, padding='same')(g)
    theta_x = tf.keras.layers.Conv3D(n_channels, 1, stride, padding='same')(x)

    weights = phi_g + theta_x
    weights = tf.keras.layers.Activation('relu')(weights)
    weights = tf.keras.layers.Conv3D(1, 1, 1, padding='same')(weights)
    weights = tf.keras.layers.Activation('relu')(weights)
    weights = tf.keras.layers.Conv3DTranspose(1, (3, 3, 3), stride, padding='same')(weights)
    weights = tf.keras.layers.Activation('sigmoid')(weights)

    outputs = x * weights
    # outputs = tf.keras.layers.Conv3D(n_channels, 1, 1, padding='same')(outputs)
    # outputs = tf.keras.layers.BatchNormalization()(outputs)
    # outputs = tf.keras.layers.Activation('relu')(outputs)

    return outputs


def _efficient_channel_attention(input_layer, eca_k_size=3):

    weights = tf.keras.layers.GlobalAveragePooling3D()(input_layer)
    weights = tf.expand_dims(weights, -1)

    weights = tf.keras.layers.Conv1D(1, kernel_size=1, padding='same')(weights)
    weights = tf.keras.layers.Activation('relu')(weights)
    weights = tf.keras.layers.Conv1D(1, kernel_size=eca_k_size, padding='same', use_bias=False)(weights)
    
    weights = tf.expand_dims(tf.expand_dims(tf.transpose(weights, (0, 2, 1)), 1), 1)
    weights = tf.keras.layers.Activation('sigmoid')(weights)

    return input_layer * weights


def get_eca_kernel_size(n_filters, gamma=2, b=1):
    t = int(abs( np.log2(n_filters) + b ) / gamma)
    k = t if t % 2 else t + 1
    return k


def _squeeze_and_excite(input_layer, squeeze_factor=2):
    n_channels = input_layer.shape[-1]

    global_pool = tf.keras.layers.GlobalAveragePooling3D()
    reshape = tf.keras.layers.Reshape((1, 1, 1, n_channels))
    dense_encoder = tf.keras.layers.Conv3D(n_channels // squeeze_factor, 1, 1, activation='relu')
    dense_decoder = tf.keras.layers.Conv3D(n_channels, 1, 1, activation='sigmoid')
    
    layer_list = [
        global_pool,
        reshape,
        dense_encoder,
        dense_decoder,
        
    ]
    weights = _combine_layers(input_layer,  layer_list)

    return input_layer * weights


def _patch_embedding(input_layer, emb_dim, patch_size, stride):
    b, h, w, d, c = tf.unstack(tf.shape(input_layer))

    patch_emb = tf.keras.layers.Conv3D(emb_dim, patch_size, stride, padding='same')(input_layer)
    patch_emb = tf.keras.layers.BatchNormalization()(patch_emb)

    patch_emb = tf.reshape(patch_emb, (b, h*w*d, emb_dim))
    return patch_emb


def UNet(input_shape, n_filters, k_size, k_stride):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    conv_block_1 = _encoder_block(inputs, n_filters, k_size, k_stride)
    conv_block_2 = _encoder_block(conv_block_1, n_filters * 2, k_size, k_stride)
    conv_block_3 = _encoder_block(conv_block_2, n_filters * 4, k_size, (1, 1, 1))
    conv_block_4 = _encoder_block(conv_block_3, n_filters * 8, k_size, (1, 1, 1))

    # Decoder
    inv_conv_block_1 = _decoder_block(conv_block_4, conv_block_3, n_filters * 4, k_size, (1, 1, 1), 'relu')
    inv_conv_block_2 = _decoder_block(inv_conv_block_1, conv_block_2, n_filters * 2, k_size, (1, 1, 1), 'relu')    
    inv_conv_block_3 = _decoder_block(inv_conv_block_2, conv_block_1, n_filters, k_size, k_stride, 'relu')    
    inv_conv_block_4 = _decoder_block(inv_conv_block_3, None, 1, k_size, k_stride, 'linear')

    model = tf.keras.Model(inputs=inputs, outputs=inv_conv_block_4)

    return model


def AttentionUNet(input_shape, n_filters, k_size, k_stride):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    conv_block_1 = _encoder_block(inputs, n_filters, k_size, k_stride)
    conv_block_2 = _encoder_block(conv_block_1, n_filters * 2, k_size, k_stride)
    conv_block_3 = _encoder_block(conv_block_2, n_filters * 4, k_size, (1, 1, 1))
    conv_block_4 = _encoder_block(conv_block_3, n_filters * 8, k_size, (1, 1, 1))

    # Decoder
    conv_block_3 = _attention_gate(conv_block_4, conv_block_3, (1, 1, 1))
    inv_conv_block_1 = _decoder_block(conv_block_4, conv_block_3, n_filters * 4, k_size, (1, 1, 1), 'relu')

    conv_block_2 = _attention_gate(inv_conv_block_1, conv_block_2, (1, 1, 1))
    inv_conv_block_2 = _decoder_block(inv_conv_block_1, conv_block_2, n_filters * 2, k_size, (1, 1, 1), 'relu')

    conv_block_1 = _attention_gate(inv_conv_block_2, conv_block_1, k_stride)
    inv_conv_block_3 = _decoder_block(inv_conv_block_2, conv_block_1, n_filters, k_size, k_stride, 'relu')

    inv_conv_block_4 = _decoder_block(inv_conv_block_3, None, 1, k_size, k_stride, 'linear')

    model = tf.keras.Model(inputs=inputs, outputs=inv_conv_block_4)

    return model


def ECAUNet(input_shape, n_filters, k_size, k_stride):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    conv_block_1 = _encoder_block(inputs, n_filters, k_size, k_stride)
    conv_block_1 = _efficient_channel_attention(conv_block_1, get_eca_kernel_size(n_filters))
    
    conv_block_2 = _encoder_block(conv_block_1, n_filters * 2, k_size, k_stride)
    conv_block_2 = _efficient_channel_attention(conv_block_2, get_eca_kernel_size(n_filters * 2))
    
    conv_block_3 = _encoder_block(conv_block_2, n_filters * 4, k_size, (1, 1, 1))
    conv_block_3 = _efficient_channel_attention(conv_block_3, get_eca_kernel_size(n_filters * 4))
    
    conv_block_4 = _encoder_block(conv_block_3, n_filters * 8, k_size, (1, 1, 1))
    conv_block_4 = _efficient_channel_attention(conv_block_4, get_eca_kernel_size(n_filters * 8))

    # Decoder
    inv_conv_block_1 = _decoder_block(conv_block_4, conv_block_3, n_filters * 4, k_size, (1, 1, 1), 'relu')
    inv_conv_block_1 = _efficient_channel_attention(inv_conv_block_1, get_eca_kernel_size(n_filters * 8))
    
    inv_conv_block_2 = _decoder_block(inv_conv_block_1, conv_block_2, n_filters * 2, k_size, (1, 1, 1), 'relu')
    inv_conv_block_2 = _efficient_channel_attention(inv_conv_block_2, get_eca_kernel_size(n_filters * 4))
    
    inv_conv_block_3 = _decoder_block(inv_conv_block_2, conv_block_1, n_filters, k_size, k_stride, 'relu')
    inv_conv_block_3 = _efficient_channel_attention(inv_conv_block_3, get_eca_kernel_size(n_filters * 2))
    
    inv_conv_block_4 = _decoder_block(inv_conv_block_3, None, 1, k_size, k_stride, 'linear')

    model = tf.keras.Model(inputs=inputs, outputs=inv_conv_block_4)

    return model


def SEUNet(input_shape, n_filters, k_size, k_stride, squeeze_factor=2):
    
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    conv_block_1 = _encoder_block(inputs, n_filters, k_size, k_stride)
    conv_block_1 = _squeeze_and_excite(conv_block_1, squeeze_factor)
    
    conv_block_2 = _encoder_block(conv_block_1, n_filters * 2, k_size, k_stride)
    conv_block_2 = _squeeze_and_excite(conv_block_2, squeeze_factor)
    
    conv_block_3 = _encoder_block(conv_block_2, n_filters * 4, k_size, (1, 1, 1))
    conv_block_3 = _squeeze_and_excite(conv_block_3, squeeze_factor)
    
    conv_block_4 = _encoder_block(conv_block_3, n_filters * 8, k_size, (1, 1, 1))
    conv_block_4 = _squeeze_and_excite(conv_block_4, squeeze_factor)

    # Decoder
    inv_conv_block_1 = _decoder_block(conv_block_4, conv_block_3, n_filters * 4, k_size, (1, 1, 1), 'relu')
    inv_conv_block_1 = _squeeze_and_excite(inv_conv_block_1, squeeze_factor)
    
    inv_conv_block_2 = _decoder_block(inv_conv_block_1, conv_block_2, n_filters * 2, k_size, (1, 1, 1), 'relu')
    inv_conv_block_2 = _squeeze_and_excite(inv_conv_block_2, squeeze_factor)
    
    inv_conv_block_3 = _decoder_block(inv_conv_block_2, conv_block_1, n_filters, k_size, k_stride, 'relu')
    inv_conv_block_3 = _squeeze_and_excite(inv_conv_block_3, squeeze_factor)
    
    inv_conv_block_4 = _decoder_block(inv_conv_block_3, None, 1, k_size, k_stride, 'linear')

    model = tf.keras.Model(inputs=inputs, outputs=inv_conv_block_4)

    return model

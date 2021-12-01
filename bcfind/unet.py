import tensorflow as tf


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


def get_model(input_shape, n_filters, k_size, k_stride):
    inputs = tf.keras.Input(input_shape)

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

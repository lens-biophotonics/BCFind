from keras.engine import data_adapter
import tensorflow as tf
import numpy as np

from bcfind.layers.switch_normalization import SwitchNormalization
from bcfind.losses.losses import *


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


def _encoder_block(input_layer, n_filters, conv_size, conv_stride, regularizer=None):
    conv3D = tf.keras.layers.Conv3D(
        filters=n_filters,
        kernel_size=conv_size,
        strides=conv_stride,
        padding='same',
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
    )
    batch_norm = tf.keras.layers.BatchNormalization()
    # switch_norm = SwitchNormalization()
    relu = tf.keras.layers.Activation('relu')

    layer_list = [
        conv3D,
        batch_norm,
        # switch_norm,
        relu,
    ]

    layer_out = _combine_layers(input_layer, layer_list)
    return layer_out


def _decoder_block(
    input_layer, 
    to_concatenate_layer, 
    n_filters, 
    conv_size, 
    conv_stride, 
    activation, 
    regularizer=None,
    attention=False,
    ):
    conv3D_T = tf.keras.layers.Conv3DTranspose(
        filters=n_filters,
        kernel_size=conv_size,
        strides=conv_stride,
        padding='same',
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
    )

    batch_norm = tf.keras.layers.BatchNormalization()
    # switch_norm = SwitchNormalization()
    activ = tf.keras.layers.Activation(activation)

    layer_list = [
        conv3D_T,
        batch_norm,
        # switch_norm,
        activ,
    ]

    layer_out = _combine_layers(input_layer, layer_list)

    if to_concatenate_layer is not None:
        if attention:
            to_concatenate_layer = _attention_gate(to_concatenate_layer, layer_out)
    
        layer_out = tf.keras.layers.concatenate([layer_out, to_concatenate_layer])

    return layer_out


def _attention_gate(queries, keys):
    assert queries.shape == keys.shape

    b, h, w, d, c = queries.shape
    
    q = tf.keras.layers.Reshape([h*w*d, c])(queries)
    k = tf.keras.layers.Reshape([h*w*d, c])(keys)
    
    weights = tf.keras.layers.Dot([1, 1])([q, k])
    weights = tf.keras.layers.Activation('softmax')(weights)

    att = tf.keras.layers.Dot([-1, -1])([q, weights])
    att = tf.keras.layers.Reshape([h, w, d, c])(att)

    outputs = queries + att
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



def ECAUNet(n_filters, k_size, k_stride):
    inputs = tf.keras.layers.Input((None, None, None, 1))

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

    return tf.keras.Model(inputs=inputs, outputs=inv_conv_block_4)


def SEUNet(n_filters, k_size, k_stride, squeeze_factor=2):
    inputs = tf.keras.layers.Input((None, None, None, 1))

    # Encoder
    conv_block_1 = _encoder_block(inputs, n_filters, k_size, k_stride)
    conv_block_1 = _squeeze_and_excite(conv_block_1, squeeze_factor)
    
    conv_block_2 = _encoder_block(conv_block_1, n_filters * 2, k_size, k_stride)
    conv_block_2 = _squeeze_and_excite(conv_block_2, squeeze_factor)
    
    conv_block_3 = _encoder_block(conv_block_2, n_filters * 4, k_size, (1, 1, 1))
    conv_block_3 = _squeeze_and_excite(conv_block_3, squeeze_factor)
    
    # Bottleneck
    conv_block_4 = _encoder_block(conv_block_3, n_filters * 8, k_size, (1, 1, 1))
    conv_block_4 = _squeeze_and_excite(conv_block_4, squeeze_factor)

    # Decoder
    inv_conv_block_1 = _decoder_block(conv_block_4, conv_block_3, n_filters * 4, k_size, (1, 1, 1), 'relu')
    inv_conv_block_1 = _squeeze_and_excite(inv_conv_block_1, squeeze_factor)
    
    inv_conv_block_2 = _decoder_block(inv_conv_block_1, conv_block_2, n_filters * 2, k_size, (1, 1, 1), 'relu')
    inv_conv_block_2 = _squeeze_and_excite(inv_conv_block_2, squeeze_factor)
    
    inv_conv_block_3 = _decoder_block(inv_conv_block_2, conv_block_1, n_filters, k_size, k_stride, 'relu')
    inv_conv_block_3 = _squeeze_and_excite(inv_conv_block_3, squeeze_factor)
    
    # Predictor
    inv_conv_block_4 = _decoder_block(inv_conv_block_3, None, 1, k_size, k_stride, 'linear')

    return tf.keras.Model(inputs=inputs, outputs=inv_conv_block_4)


def keep_top_k(tensor, k):
    top_k_values, _ = tf.math.top_k(tensor, k, sorted=True)
    mask = tensor >= top_k_values[..., -1][..., tf.newaxis]
    return tf.where(mask, tensor, -np.inf)


def GateNet(inputs, n_encoding_layers, n_filters, k_size, k_stride, n_experts, k):

    conv_block = _encoder_block(inputs, n_filters , k_size, k_stride)
    for i in range(1, n_encoding_layers):
        conv_block = _encoder_block(conv_block, n_filters * (2 ** i), k_size, k_stride)
    
    conv_block = tf.keras.layers.GlobalAveragePooling3D(keepdims=True)(conv_block)
    
    gate_weights = tf.keras.layers.Conv3D(n_filters * (2 ** i), kernel_size=1, kernel_initializer='zeros')(conv_block)
    gate_weights = tf.keras.layers.BatchNormalization()(gate_weights)
    gate_weights = tf.keras.layers.Activation('relu')(gate_weights)
    gate_weights = tf.keras.layers.Conv3D(n_experts, kernel_size=1, kernel_initializer='zeros')(conv_block)

    # gate_noise = tf.keras.layers.Conv3D(n_filters * (2 ** i), kerne_size=1, kernel_initializer='zeros')(conv_block)
    # gate_noise = tf.keras.layers.BatchNormalization()(gate_noise)
    # gate_noise = tf.keras.layers.Activation('relu')(gate_noise)
    # gate_noise = tf.keras.layers.Conv3D(n_experts, kernel_size=1, kernel_initializer='zeros')(conv_block)
    # gate_noise = tf.keras.layers.Activation('softplus')(gate_noise)

    # noisy_gate_weights = gate_weights + tf.random.normal(tf.shape(gate_noise)) * gate_noise

    sparse_gate_weights = keep_top_k(gate_weights, k)
    sparse_gate_weights = tf.keras.layers.Activation('softmax')(sparse_gate_weights)

    gatenet = tf.keras.Model(inputs=inputs, outputs=sparse_gate_weights)
    
    return gatenet


class MoUNetE(tf.keras.models.Model):
    def __init__(self, n_filters, k_size, k_stride, n_experts, top_k_experts, dropout=None, regularizer=None):
        super(MoUNetE, self).__init__()
        
        inputs = tf.keras.layers.Input((None, None, None, 1))
        self.gate = GateNet(inputs, 4, n_filters, k_size, k_stride, n_experts, top_k_experts)
        
        self.experts = []
        for _ in range(n_experts):
            unet = UNet(inputs, n_filters, k_size, k_stride, dropout, regularizer)
            self.experts.append(unet)
        
        self.imp_loss = ImportanceLoss(0.2)
        self.load_loss = LoadLoss(0.01)
        self.build((None, None, None, None, 1))

    def call(self, inputs, training=True):
        expert_weights = self.gate(inputs)

        expert_outputs = []
        for i in range(len(self.experts)):
            expert_outputs.append(self.experts[i](inputs))

        if training:
            return expert_weights, expert_outputs
        else:
            expert_outputs = tf.concat(expert_outputs, axis=-1)
            outputs = expert_outputs * expert_weights
            outputs = tf.reduce_sum(outputs, axis=-1)
            outputs = tf.expand_dims(outputs, -1)
            return outputs
    
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        weights, y_pred = y_pred
        
        i_loss = self.imp_loss(weights)
        l_loss = self.load_loss(weights)

        weights = tf.squeeze(weights)
        
        loss = tf.zeros_like(weights[:, 0])
        for i in range(len(y_pred)):
            exp_loss = self.compiled_loss(y, y_pred[i])
            loss += exp_loss * weights[:, i]
        return tf.reduce_mean(loss, axis=0) + l_loss


def UNetMoE(input_shape, n_filters, k_size, k_stride, n_experts, top_k_experts, dropout=None, regularizer=None):
    inputs = tf.keras.layers.Input(input_shape)

    # encoder block 1
    gate = GateNet(input_shape, 3, n_filters, k_size, k_stride, n_experts, top_k_experts)
    expert_weights = gate(inputs)

    outputs = []
    for _ in range(n_experts):
        expert_out = _encoder_block(inputs, n_filters, k_size, k_stride, regularizer)
        outputs.append(expert_out[..., tf.newaxis])
    outputs = tf.concat(outputs, axis=-1)
    conv_block_1 = tf.reduce_sum(outputs * expert_weights, axis=-1)

    # encoder block 2
    gate = GateNet(conv_block_1.shape[1:], 2, n_filters * 2, k_size, k_stride, n_experts, top_k_experts)
    expert_weights = gate(conv_block_1)

    outputs = []
    for _ in range(n_experts):
        expert_out = _encoder_block(conv_block_1, n_filters * 2, k_size, k_stride, regularizer)
        outputs.append(expert_out[..., tf.newaxis])
    outputs = tf.concat(outputs, axis=-1)
    conv_block_2 = tf.reduce_sum(outputs * expert_weights, axis=-1)
    
    # encoder block 3
    gate = GateNet(conv_block_2.shape[1:], 1, n_filters * 4, k_size, k_stride, n_experts, top_k_experts)
    expert_weights = gate(conv_block_2)

    outputs = []
    for _ in range(n_experts):
        expert_out = _encoder_block(conv_block_2, n_filters, k_size, (1, 1, 1), regularizer)
        outputs.append(expert_out[..., tf.newaxis])
    outputs = tf.concat(outputs, axis=-1)
    conv_block_3 = tf.reduce_sum(outputs * expert_weights, axis=-1)
    
    # bottleneck block
    gate = GateNet(conv_block_3.shape[1:], 1, n_filters * 8, k_size, k_stride, n_experts, top_k_experts)
    expert_weights = gate(conv_block_3)

    outputs = []
    for _ in range(n_experts):
        expert_out = _encoder_block(conv_block_3, n_filters * 8, k_size, (1, 1, 1), regularizer)
        outputs.append(expert_out[..., tf.newaxis])
    outputs = tf.concat(outputs, axis=-1)
    conv_block_4 = tf.reduce_sum(outputs * expert_weights, axis=-1)
    
    # decoder block 1
    gate = GateNet(conv_block_4.shape[1:], 1, n_filters * 4, k_size, k_stride, n_experts, top_k_experts)
    expert_weights = gate(conv_block_4)

    outputs = []
    for _ in range(n_experts):
        expert_out = _decoder_block(conv_block_4, conv_block_3, n_filters * 4, k_size, (1, 1, 1), 'relu')
        outputs.append(expert_out[..., tf.newaxis])
    outputs = tf.concat(outputs, axis=-1)
    inv_conv_block_1 = tf.reduce_sum(outputs * expert_weights, axis=-1)
    
    # decoder block 2
    gate = GateNet(inv_conv_block_1.shape[1:], 1, n_filters * 2, k_size, k_stride, n_experts, top_k_experts)
    expert_weights = gate(inv_conv_block_1)

    outputs = []
    for _ in range(n_experts):
        expert_out = _decoder_block(inv_conv_block_1, conv_block_2, n_filters * 2, k_size, (1, 1, 1), 'relu')
        outputs.append(expert_out[..., tf.newaxis])
    outputs = tf.concat(outputs, axis=-1)
    inv_conv_block_2 = tf.reduce_sum(outputs * expert_weights, axis=-1)
    
    # decoder block 3
    gate = GateNet(inv_conv_block_2.shape[1:], 1, n_filters * 2, k_size, k_stride, n_experts, top_k_experts)
    expert_weights = gate(inv_conv_block_2)

    outputs = []
    for _ in range(n_experts):
        expert_out = _decoder_block(inv_conv_block_2, conv_block_1, n_filters, k_size, k_stride, 'relu')
        outputs.append(expert_out[..., tf.newaxis])
    outputs = tf.concat(outputs, axis=-1)
    inv_conv_block_3 = tf.reduce_sum(outputs * expert_weights, axis=-1)
    
    # predictor
    gate = GateNet(inv_conv_block_3.shape[1:], 1, n_filters * 2, k_size, k_stride, n_experts, top_k_experts)
    expert_weights = gate(inv_conv_block_3)

    outputs = []
    for _ in range(n_experts):
        expert_out = _decoder_block(inv_conv_block_3, None, 1, k_size, k_stride, 'linear')
        outputs.append(expert_out[..., tf.newaxis])
    outputs = tf.concat(outputs, axis=-1)
    predictions = tf.reduce_sum(outputs * expert_weights, axis=-1)
    
    return tf.keras.Model(inputs=inputs, outputs=predictions)

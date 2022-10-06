import numpy as np
import tensorflow as tf

from bcfind.layers import EncoderBlock
from bcfind.models import UNet
from bcfind.losses import ImportanceLoss, LoadLoss


def _keep_top_k(tensor, k):
    top_k_values, _ = tf.math.top_k(tensor, k, sorted=True)
    mask = tensor >= top_k_values[..., -1][..., tf.newaxis]
    return tf.where(mask, tensor, -np.inf)


class GateNet(tf.keras.models.Model):
    def __init__(self, n_blocks, n_filters, k_size, k_stride, n_experts, keep_top_k=None, add_noise=False, **kwargs):
        super(GateNet, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.n_filters = n_filters 
        self.k_size = k_size 
        self.k_stride = k_stride 
        self.n_experts = n_experts
        self.keep_top_k = keep_top_k
        self.add_noise = add_noise

        self.encoder_blocks = []
        for i in range(self.n_blocks):
            encoder_block = EncoderBlock(self.n_filters * (2 ** i), k_size, k_stride)
            self.encoder_blocks.append(encoder_block)

        self.gap = tf.keras.layers.GlobalAveragePooling3D(keepdims=True)

        self.w_conv_1 = tf.keras.layers.Conv3D(n_filters * (2 ** i), kernel_size=1, kernel_initializer='zeros')
        self.w_bn = tf.keras.layers.BatchNormalization()
        self.w_relu = tf.keras.layers.Activation('relu')
        self.w_conv_2 = tf.keras.layers.Conv3D(n_experts, kernel_size=1, kernel_initializer='zeros')

        if self.add_noise:
            self.n_conv_1 = tf.keras.layers.Conv3D(n_filters * (2 ** i), kernel_size=1, kernel_initializer='zeros')
            self.n_bn = tf.keras.layers.BatchNormalization()
            self.n_relu = tf.keras.layers.Activation('relu')
            self.n_conv_2 = tf.keras.layers.Conv3D(n_experts, kernel_size=1, kernel_initializer='zeros')
        
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=None):
        h = self.encoder_blocks[0](inputs)
        for i in range(1, len(self.encoder_blocks)):
            h = self.encoder_blocks[i](h, training=training)
        
        h = self.gap(h)

        weights = self.w_conv_1(h)
        weights = self.w_bn(weights, training=training)
        weights = self.w_relu(weights)
        weights = self.w_conv_2(weights)

        if self.add_noise:
            noise = self.n_conv_1(h)
            noise = self.n_bn(noise, training=training)
            noise = self.n_relu(noise)
            noise = self.n_conv_2(noise)
            noise *= tf.random.normal(tf.shape(noise))

            weights += noise

        if self.keep_top_k is not None:
            weights = _keep_top_k(weights)
        
        weights = self.softmax(weights)
        return weights
    
    def get_config(self,):
        config = {
            'n_blocks': self.n_blocks,
            'n_filters': self.n_filters,
            'k_size': self.k_size,
            'k_stride': self.k_stride,
            'n_experts': self.n_experts,
            'keep_top_k': self.keep_top_k,
            'add_noise': self.add_noise,
        }
        base_config = super(GateNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MoUNets(tf.keras.models.Model):
    def __init__(self, n_blocks, n_filters, k_size, k_stride, n_experts, keep_top_k=None, dropout=None, regularizer=None, **kwargs):
        super(MoUNets, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.k_size = k_size 
        self.k_stride = k_stride 
        self.n_experts = n_experts 
        self.keep_top_k = keep_top_k 
        self.dropout = dropout
        self.regularizer = regularizer
        
        self.gate = GateNet(n_blocks, n_filters, k_size, k_stride, n_experts, keep_top_k)
        
        self.experts = []
        for _ in range(n_experts):
            unet = UNet(n_blocks, n_filters, k_size, k_stride, dropout, regularizer)
            self.experts.append(unet)
        
        self.imp_loss = ImportanceLoss(0.2)
        self.load_loss = LoadLoss(0.01)
    
    def build(self, input_shape):
        self.gate.build(input_shape)
        for exp in self.experts:
            exp.build(input_shape)
        self.built = True

    def call(self, inputs, training=None):
        expert_weights = self.gate(inputs, training=training)

        expert_outputs = []
        for i in range(len(self.experts)):
            expert_outputs.append(self.experts[i](inputs, training=training))

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
    
    def get_config(self,):
        config = {
            'n_blocks': self.n_blocks,
            'n_filters': self.n_filters,
            'k_size': self.k_size,
            'k_stride': self.k_stride,
            'n_experts': self.n_experts,
            'keep_top_k': self.keep_top_k,
            'dropout': self.dropout,
            'regularizer': self.regularizer,
        }
        base_config = super(MoUNets, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


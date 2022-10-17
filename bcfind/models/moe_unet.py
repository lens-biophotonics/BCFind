import numpy as np
import tensorflow as tf

from bcfind.layers import EncoderBlock
from bcfind.models import UNet
from bcfind.losses import ImportanceLoss, LoadLoss, MUMLRegularizer


def _keep_top_k(tensor, k):
    top_k_values, _ = tf.math.top_k(tensor, k, sorted=True)
    mask = tensor >= top_k_values[..., -1][..., tf.newaxis]
    return tf.where(mask, tensor, -np.inf)


class GateNet(tf.keras.Model):
    def __init__(
        self, 
        n_blocks, 
        n_filters, 
        k_size, 
        k_stride, 
        n_experts, 
        keep_top_k=None, 
        add_noise=False, 
        **kwargs
        ):
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
        self.w_conv_2 = tf.keras.layers.Conv3D(n_experts, kernel_size=1, kernel_initializer='zeros', kernel_regularizer=MUMLRegularizer(0.05))

        if self.add_noise:
            self.n_conv_1 = tf.keras.layers.Conv3D(n_filters * (2 ** i), kernel_size=1, kernel_initializer='zeros')
            self.n_bn = tf.keras.layers.BatchNormalization()
            self.n_relu = tf.keras.layers.Activation('relu')
            self.n_conv_2 = tf.keras.layers.Conv3D(n_experts, kernel_size=1, kernel_initializer='zeros', kernel_regularizer=MUMLRegularizer(0.05))
        
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

        if self.add_noise and training:
            noise = self.n_conv_1(h)
            noise = self.n_bn(noise, training=training)
            noise = self.n_relu(noise)
            noise = self.n_conv_2(noise)
            noise *= tf.random.normal(tf.shape(noise))

            weights += noise

        if self.keep_top_k:
            weights = _keep_top_k(weights, self.keep_top_k)
        
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


class MoUNets(tf.keras.Model):
    """Class for Mixture of UNets model. 
    In the mixture of experts model, many experts are trained to output the desired target, while a gate network weights the losses of each expert \
    for each input during training or directly weights each expert output during inference (the weights obviously sum to one).

    This implementation mainly refers to:
        - R. Jacobs et al. 'Adaptive Mixture of Local Experts  <https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf>'
        - Shazeer et al. 'Outrageously Large Neural Networks <https://www.cs.toronto.edu/~hinton/absps/Outrageously.pdf>' for the importance loss in
        the gate network
        - Fedus et al. 'Switch Transformers <https://arxiv.org/pdf/2101.03961.pdf>' for the load loss in the gate network
    """
    def __init__(
        self,
        n_blocks,
        n_filters,
        k_size,
        k_stride,
        n_experts,
        keep_top_k=None,
        add_noise=False,
        balance_loss=None,
        dropout=None,
        regularizer=None,
        **kwargs
        ):
        """Constructor method.

        Parameters
        ----------
        n_blocks : int
            depth of each UNet encoder and the Gate
        n_filters : int
            number of filters for the first layer. Consecutive layers increase esponentially their number of filters.
        k_size : int or tuple of ints
            size of the kernel for convolutional layers
        k_stride : int or tuple of ints
            stride for the convolutional layers. The last two encoding and the first two decoding layers will however have no stride.
        n_experts : int
            number of experts (UNets)
        keep_top_k : int, optional
            if different from 0, None, False, make a hard selection of the best k experts as stated by the gate net. 
            Must be an int between 1 and n_experts, by default None.
        add_noise : bool, optional
            whether to add gaussian noise to the expert weights or not, by default False. 
            Set to True can sometimes stabilize training, but also make it much slower and make it need more epochs for reaching convergence. 
            Noise will never be added during inference.
        balance_loss : string, optional
            one of ['importance', 'load'], by default None. Importance or Load loss will be added to overall loss to balance weights 
            and number of inputs received between experts.
        dropout : bool, optional
            whether or not to add dropout layer after each convolutional block, by default None.
        regularizer : string or tf.keras.regularizers, optional
            a regularization method for keras layers, by default None.
        """        
        
        super(MoUNets, self).__init__(**kwargs)
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.k_size = k_size 
        self.k_stride = k_stride 
        self.n_experts = n_experts 
        self.keep_top_k = keep_top_k
        self.add_noise = add_noise
        self.balance_loss = balance_loss
        self.dropout = dropout
        self.regularizer = regularizer
        
        self.gate = GateNet(self.n_blocks, self.n_filters, self.k_size, self.k_stride, self.n_experts, self.keep_top_k, self.add_noise)
        
        self.experts = []
        for _ in range(self.n_experts):
            unet = UNet(self.n_blocks, self.n_filters, self.k_size, self.k_stride, self.dropout, self.regularizer)
            self.experts.append(unet)
        
        if self.balance_loss == 'importance':
            self.gate_loss = ImportanceLoss(0.2)
        elif self.balance_loss == 'load':
            self.gate_loss = LoadLoss(0.01)
    
    def call(self, inputs, training=None):
        expert_weights = self.gate(inputs, training=training)

        expert_outputs = []
        for i in range(len(self.experts)):
            expert_outputs.append(self.experts[i](inputs, training=training))

        if training:
            return expert_weights, expert_outputs
        else:
            expert_outputs = tf.concat(expert_outputs, axis=-1)
            outputs = tf.reduce_sum(expert_outputs * expert_weights, axis=-1)
            return tf.expand_dims(outputs, -1)
    
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        weights, y_pred = y_pred

        if self.balance_loss in ['importance', 'load']:
            g_loss = self.gate_loss(weights)

        weights = tf.squeeze(weights)
        loss = tf.zeros_like(weights[:, 0])

        # weighted crossentropy loss
        # for i in range(self.n_experts):
        #     exp_loss = self.compiled_loss(y, y_pred[i])
        #     loss += exp_loss * weights[:, i]
        
        # weighted loss as Jacobs et al. (1991)
        for i in range(self.n_experts):
            exp_loss = self.compiled_loss(y, y_pred[i])
            exp_loss = tf.exp(- 0.5 * exp_loss)
            loss += exp_loss * weights[..., i]
        loss = - tf.math.log(loss)

        if self.balance_loss in ['importance', 'load']:
            return tf.reduce_mean(loss, axis=0) + g_loss
        else:
            return tf.reduce_mean(loss, axis=0)

    def get_config(self,):
        config = {
            'n_blocks': self.n_blocks,
            'n_filters': self.n_filters,
            'k_size': self.k_size,
            'k_stride': self.k_stride,
            'n_experts': self.n_experts,
            'keep_top_k': self.keep_top_k,
            'add_noise': self.add_noise,
            'balance_loss': self.balance_loss,
            'dropout': self.dropout,
            'regularizer': self.regularizer,
        }
        base_config = super(MoUNets, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))

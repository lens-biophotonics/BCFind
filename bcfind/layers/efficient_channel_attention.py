import numpy as np
import tensorflow as tf


def _get_eca_kernel_size(n_filters, gamma=2, b=1):
    t = int(abs( np.log2(n_filters) + b ) / gamma)
    k = t if t % 2 else t + 1
    return k


class EfficientChannelAttention(tf.keras.layers.Layer):
    def __init__(self, k_size, **kwargs):
        super(EfficientChannelAttention, self).__init__(**kwargs)
        self.k_size = k_size

        self.gap = tf.keras.layers.GlobalAveragePooling3D()
        self.conv1d_1 = tf.keras.layers.Conv1D(1, kernel_size=1, padding='same', activation='relu')
        self.conv1d_2 = tf.keras.layers.Conv1D(1, kernel_size=self.k_size, padding='same', use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
    
    def call(self, inputs):
        weights = self.gap(inputs)[..., tf.newaxis]
        weights = self.conv1d_1(weights)
        weights = self.conv1d_2(weights)

        weights = tf.transpose(weights, (0, 2, 1))[:, tf.newaxis, tf.newaxis, ...]
        weights = self.sigmoid(weights)
        return inputs * weights

    def get_config(self, ):
        config = {
            'k_size': self.k_size,
        }
        base_config = super(EfficientChannelAttention, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


tf.keras.utils.get_custom_objects().update({'EfficientChannelAttention': EfficientChannelAttention})
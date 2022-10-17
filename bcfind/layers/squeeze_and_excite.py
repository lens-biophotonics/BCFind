import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='BCFind', name='SqueezeAndExcite')
class SqueezeAndExcite(tf.keras.layers.Layer):
    def __init__(self, n_input_channels, squeeze_factor=2, **kwargs):
        super(SqueezeAndExcite, self).__init__(**kwargs)
        self.n_input_channels = n_input_channels
        self.squeeze_factor = squeeze_factor

        self.gap = tf.keras.layers.GlobalAveragePooling3D(keepdims=True)
        self.squeeze = tf.keras.layers.Conv3D(self.n_input_channels // self.squeeze_factor, 1, 1, activation='relu')
        self.excite = tf.keras.layers.Conv3D(self.n_input_channels, 1, 1, activation='sigmoid')

    def call(self, inputs):
        weights = self.gap(inputs)
        weights = self.squeeze(weights)
        weights = self.excite(weights)
        return inputs * weights
    
    def get_config(self,):
        config = {
            'n_input_channels': self.n_input_channels,
            'squeeze_factor': self.squeeze_factor,
        }
        base_config = super(SqueezeAndExcite, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))

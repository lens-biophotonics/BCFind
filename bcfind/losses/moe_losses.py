import tensorflow as tf


class ImportanceLoss():
    '''
    Importance Loss from "Outrageosly Large Neural Networks" by Shazeer et al. (2017)
    '''
    def __init__(self, alpha):
        self.alpha = alpha
    
    # @tf.function
    def __call__(self, expert_weights):
        importance = tf.reduce_sum(tf.squeeze(expert_weights), axis=0)
        return self.alpha * (tf.math.reduce_std(importance) / tf.reduce_mean(importance)) ** 2


class LoadLoss():
    '''
    Load loss from "Switch Transformers" by Fedus et al. (2022)
    '''
    def __init__(self, alpha):
        self.alpha = alpha
    
    # @tf.function
    def __call__(self, gate_weights):
        bs, _, _, _, n_exp = gate_weights.shape
        gate_weights = tf.squeeze(gate_weights)

        input_choice = tf.argmax(gate_weights, axis=-1)
        exp_importance = tf.reduce_mean(gate_weights, axis=0)

        load = 0
        for i in range(n_exp):
            is_best_exp = tf.cast(tf.equal(input_choice, i), tf.float32)
            best_exp_times = tf.reduce_sum(is_best_exp)
            exp_samples = best_exp_times / bs if bs is not None else best_exp_times
            load += exp_importance[i] * exp_samples        
        
        return self.alpha * n_exp * load


tf.keras.utils.get_custom_objects().update({'ImportanceLoss': ImportanceLoss, 'LoadLoss': LoadLoss})
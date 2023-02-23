import tensorflow as tf


class ImportanceLoss(tf.keras.losses.Loss):
    """
    Importance Loss from "Outrageosly Large Neural Networks" by Shazeer et al. (2017)
    """

    def __init__(self, alpha=0.2, **kwargs):
        super(ImportanceLoss, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, expert_weights):
        importance = tf.reduce_sum(tf.squeeze(expert_weights), axis=0)
        return (
            self.alpha
            * (tf.math.reduce_std(importance) / tf.reduce_mean(importance)) ** 2
        )

    def __call__(self, expert_weights):
        return self.call(expert_weights)

    def get_config(self):
        config = super(ImportanceLoss, self).get_config()
        config.update(
            {
                "alpha": self.alpha,
            }
        )
        return config


class LoadLoss(tf.keras.losses.Loss):
    """
    Load loss from "Switch Transformers" by Fedus et al. (2022)
    """

    def __init__(self, alpha=0.01, **kwargs):
        super(LoadLoss, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, gate_weights):
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

    def __call__(self, gate_weights):
        return self.call(gate_weights)

    def get_config(self):
        config = super(LoadLoss, self).get_config()
        config.update(
            {
                "alpha": self.alpha,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="BCFind", name="MUMLR")
class MUMLRegularizer(tf.keras.regularizers.Regularizer):
    """
    Correlation regularizer from "Maximal Uncorrelated Multinomial Logistic Regression" by Lei D. et al. (2019)
    """

    def __init__(self, alpha=0.001):
        self.alpha = alpha / 2

    def __call__(self, weights):
        n_classes = weights.shape[-1]
        w_norms = tf.norm(weights, axis=-2)

        dot = tf.squeeze(tf.tensordot(w_norms, w_norms, [0, 0]))
        dot = tf.linalg.set_diag(dot, tf.zeros(n_classes))

        reg = tf.reduce_sum(dot) / n_classes
        return self.alpha * reg

    def get_config(self):
        return {"alpha": float(self.alpha)}


tf.keras.utils.get_custom_objects().update(
    {
        "ImportanceLoss": ImportanceLoss,
        "LoadLoss": LoadLoss,
        "MUMLRegularizer": MUMLRegularizer,
    }
)

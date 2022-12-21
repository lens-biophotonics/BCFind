import tensorflow as tf

from bcfind.losses.utils import get_mask_fn


class Recall(tf.keras.metrics.Metric):
    def __init__(self, thresh, target_shape, border_size, from_logits=True, **kwargs):
        super(Recall, self).__init__(**kwargs)
        self.thresh = thresh
        self.target_shape = target_shape
        self.border_size = border_size
        self.from_logits = from_logits

        self.mask_fn = get_mask_fn(self.target_shape, self.border_size)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)

        y_true = tf.where(y_true >= self.thresh, 1, 0)
        y_pred = tf.where(y_pred >= self.thresh, 1, 0)

        tp = tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1))
        fn = tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0))

        tp = tf.map_fn(self.mask_fn, tp)
        fn = tf.map_fn(self.mask_fn, fn)

        tp = tf.cast(tp, self.dtype)
        fn = tf.cast(fn, self.dtype)

        self.tp.assign_add(tf.reduce_sum(tp))
        self.fn.assign_add(tf.reduce_sum(fn))

    def result(self,):
        rec = self.tp / (self.tp + self.fn + 1e-4)
        return rec

    def get_config(self,):
        config = {
            'border_size': self.border_size,
            'target_shape': self.target_shape,
            'from_logits': self.from_logits,
            'thresh': self.thresh,
        }
        base_config = super(Recall, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


class Precision(tf.keras.metrics.Metric):
    def __init__(self, thresh, target_shape, border_size, from_logits=True, **kwargs):
        super(Precision, self).__init__(**kwargs)
        self.thresh = thresh
        self.from_logits = from_logits
        self.target_shape = target_shape
        self.border_size = border_size

        self.mask_fn = get_mask_fn(self.target_shape, self.border_size)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)

        y_true = tf.where(y_true >= self.thresh, 1, 0)
        y_pred = tf.where(y_pred >= self.thresh, 1, 0)

        tp = tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1))
        fp = tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1))

        tp = tf.map_fn(self.mask_fn, tp)
        fp = tf.map_fn(self.mask_fn, fp)

        tp = tf.cast(tp, self.dtype)
        fp = tf.cast(fp, self.dtype)

        self.tp.assign_add(tf.reduce_sum(tp))
        self.fp.assign_add(tf.reduce_sum(fp))

    def result(self,):
        prec = self.tp / (self.tp + self.fp + 1e-4)
        return prec
    
    def get_config(self,):
        config = {
            'border_size': self.border_size,
            'target_shape': self.target_shape,
            'from_logits': self.from_logits,
            'thresh': self.thresh,
        }
        base_config = super(Precision, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


class F1(tf.keras.metrics.Metric):
    def __init__(self, thresh, target_shape, border_size, from_logits=True, **kwargs):
        super(F1, self).__init__(**kwargs)
        self.thresh = thresh
        self.target_shape = target_shape
        self.border_size = border_size
        self.from_logits = from_logits

        self.mask_fn = get_mask_fn(self.target_shape, self.border_size)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)

        y_true = tf.where(y_true >= self.thresh, 1, 0)
        y_pred = tf.where(y_pred >= self.thresh, 1, 0)

        tp = tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1))
        fp = tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1))
        fn = tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0))

        tp = tf.map_fn(self.mask_fn, tp)
        fp = tf.map_fn(self.mask_fn, fp)
        fn = tf.map_fn(self.mask_fn, fn)

        tp = tf.cast(tp, self.dtype)
        fp = tf.cast(fp, self.dtype)
        fn = tf.cast(fn, self.dtype)

        self.tp.assign_add(tf.reduce_sum(tp))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))

    def result(self,):
        f1 = self.tp / (self.tp + 0.5 * (self.fp + self.fn) + 1e-4)
        return f1
    
    def get_config(self,):
        config = {
            'border_size': self.border_size,
            'target_shape': self.target_shape,
            'from_logits': self.from_logits,
            'thresh': self.thresh,
        }
        base_config = super(F1, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


tf.keras.utils.get_custom_objects().update({'Recall': Recall, 'Precision': Precision, 'F1': F1})
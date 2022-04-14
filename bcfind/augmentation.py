import functools

import numpy as np
from scipy import ndimage

import tensorflow as tf

rng = np.random.default_rng()


def random_crop(x, y, zoom_range=None, target_shape=(50, 100, 100)):
    if zoom_range is not None:
        alpha = rng.uniform(zoom_range[0], zoom_range[1])

        rank = len(x.shape)

        zoom = np.ones(rank)
        zoom = alpha
        x = ndimage.zoom(x, zoom, order=0, prefilter=False)

    high = np.array(x.shape) - target_shape
    high[high == 0] = 1
    f = rng.integers(0, high)
    t = f + target_shape

    x = x[f[0]:t[0], f[1]:t[1], f[2]:t[2], np.newaxis]
    y = y[f[0]:t[0], f[1]:t[1], f[2]:t[2], np.newaxis]
    return x, y


@tf.function
def random_gamma_tf(x, param_range=(0.5, 1.8)):
    gamma = tf.random.uniform((1,), param_range[0], param_range[1])
    x_min = tf.math.reduce_min(x, axis=tuple(range(1, len(x.shape))), keepdims=True)
    x_max = tf.math.reduce_max(x, axis=tuple(range(1, len(x.shape))), keepdims=True)
    x_range = x_max - x_min
    return tf.math.pow((x - x_min) / x_range, gamma) * x_range + x_min


@tf.function
def random_noise_tf(x, param_range=(0.01, 0.05)):
    sigma = tf.random.uniform((1,), param_range[0], param_range[1])
    noise = tf.random.normal(x.shape, mean=0, stddev=sigma)
    return x + noise


@tf.function
def random_contrast_tf(x, param_range=(0, 2)):
    x_mean = tf.math.reduce_mean(x, axis=tuple(range(1, len(x.shape))), keepdims=True)
    alpha = tf.random.uniform(x_mean.shape, param_range[0], param_range[1])
    return (x - x_mean) * alpha + x_mean


@tf.function
def random_brightness_tf(x, param_range=(-50, 100)):
    return x + tf.random.uniform((x.shape[0],) + (1,) * (len(x.shape) - 1), param_range[0], param_range[1])


@tf.function
def augment(x, func_list, p=0.5):
    branch = tf.random.shuffle(tf.range(len(func_list)))  # shuffle order of transformations
    random_p = tf.random.uniform((len(func_list),), 0, 1)
    
    if isinstance(p, float) and 0 <= p <= 1:
        cond = tf.math.less(random_p, [p] * len(func_list))  # conditions based on p < probability
    
    elif isinstance(p, (list, tuple)) and len(p) == len(func_list):
        cond = tf.math.less(random_p, p)  # conditions based on p < probability
    
    else:
        raise ValueError('Augmentation probability must be a float between 0 and 1'\
            'or a list of floats between 0 and 1 whose lenght is equal to augmentation operations.')

    for i in range(len(func_list)):
        # the following line must be inside for loop, not outside, to make the partial function use the new x
        branch_fns = {j: functools.partial(func_list[j], x) for j in range(len(func_list))}
        x = tf.cond(cond[i],
                    true_fn=lambda: tf.switch_case(branch[i], branch_fns),  # apply transformation
                    false_fn=lambda: x)  # do not apply transformation
    return x

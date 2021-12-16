import functools

import numpy as np
from scipy import ndimage

import tensorflow as tf

rng = np.random.default_rng()


def random_crop(x, zoom_range=(0.8, 1.3), target_shape=(80, 240, 240)):
    if zoom_range is not None:
        alpha = rng.uniform(zoom_range[0], zoom_range[1])

        rank = len(x.shape)

        zoom = np.ones(rank)
        zoom[1:4] = alpha
        x = ndimage.zoom(x, zoom, order=0, prefilter=False)

    high = np.array(x.shape[1:4]) - target_shape
    high[high == 0] = 1
    f = rng.integers(0, high)
    t = f + target_shape

    x = x[:, f[0]:t[0], f[1]:t[1], f[2]:t[2]]
    return x


@tf.function
def random_gamma_tf(x, param_range=(0.5, 1.8)):
    gamma = tf.random.uniform((1,), param_range[0], param_range[1])
    x_min = tf.math.reduce_min(x, axis=tuple(range(1, len(x.shape))), keepdims=True)
    x_max = tf.math.reduce_max(x, axis=tuple(range(1, len(x.shape))), keepdims=True)
    x_range = x_max - x_min
    return tf.math.pow((x - x_min) / x_range, gamma) * x_range + x_min


@tf.function
def random_noise_tf(x, param_range=(0.05, 0.15)):
    sigma = tf.random.uniform((1,), param_range[0], param_range[1])
    noise = tf.random.normal(x.shape, mean=0, stddev=sigma)
    return x + noise


@tf.function
def random_contrast_tf(x):
    x_mean = tf.math.reduce_mean(x, axis=tuple(range(1, len(x.shape))), keepdims=True)
    alpha = tf.random.uniform(x_mean.shape, 0, 2)
    return (x - x_mean) * alpha + x_mean


@tf.function
def random_brightness_tf(x, alpha_range=(-50, 100)):
    return x + tf.random.uniform((x.shape[0],) + (1,) * (len(x.shape) - 1), alpha_range[0], alpha_range[1])


@tf.function
def augment(x):
    func_list = [
        random_gamma_tf,
        random_contrast_tf,
        random_brightness_tf,
        random_noise_tf,
    ]

    branch = tf.random.shuffle(tf.range(len(func_list)))  # shuffle order of transformations
    p = tf.random.uniform((len(func_list),), 0, 1)
    cond = tf.math.less(p, [0.5] * len(func_list))  # conditions based on p < probability

    for i in range(len(func_list)):
        # the following line must be inside for loop, not outside, to make the partial function use the new x
        branch_fns = {j: functools.partial(func_list[j], x) for j in range(len(func_list))}
        x = tf.cond(cond[i],
                    true_fn=lambda: tf.switch_case(branch[i], branch_fns),  # apply transformation
                    false_fn=lambda: x)  # do not apply transformation

    return x

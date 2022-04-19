import functools
import numpy as np
import tensorflow as tf

from scipy import ndimage

rng = np.random.default_rng()


@tf.function
def random_crop_tf(x, y, target_shape=(50, 100, 100)):
    def random_crop(x, y, target_shape):
        high = np.array(x.shape) - target_shape
        high[high == 0] = 1
        f = rng.integers(0, high)
        t = f + target_shape

        x = x[f[0]:t[0], f[1]:t[1], f[2]:t[2]]
        y = y[f[0]:t[0], f[1]:t[1], f[2]:t[2]]
        return x, y

    x, y = tf.numpy_function(random_crop, [x, y, target_shape], (tf.float32, tf.float32))
    return tf.ensure_shape(x, target_shape), tf.ensure_shape(y, target_shape)


@tf.function
def random_zoom_tf(x, y, param_range=(1.0, 1.3)):
    def scipy_zoom(x, y, param_range):
        zoom = rng.uniform(param_range[0], param_range[1])
        org_shape = x.shape

        x = ndimage.zoom(x, zoom, order=0, prefilter=False)
        y = ndimage.zoom(y, zoom, order=0, prefilter=False)

        x, y = random_crop_tf(x, y, org_shape)
        return x, y
    
    x_shape = x.shape
    y_shape = y.shape
    x, y = tf.numpy_function(scipy_zoom, [x, y, param_range], (tf.float32, tf.float32))
    return tf.ensure_shape(x,  x_shape), tf.ensure_shape(y, y_shape)


@tf.function
def random_rotation_tf(x, y, param_range=(-180, 180)):
    def scipy_rotate(x, y):
        angles = np.arange(param_range[0], param_range[1] + 1, 90)
        angle = rng.choice(angles)

        x = ndimage.rotate(x, angle, axes=(0, 1), reshape=False)
        y = ndimage.rotate(y, angle, axes=(0, 1), reshape=False)
        return x, y

    x_shape = x.shape
    y_shape = y.shape
    x, y =  tf.numpy_function(scipy_rotate, [x, y], (tf.float32, tf.float32))
    return tf.ensure_shape(x,  x_shape), tf.ensure_shape(y, y_shape)


@tf.function
def random_gamma_tf(x, y, param_range=(0.5, 1.8)):
    gamma = tf.random.uniform((1,), param_range[0], param_range[1])
    x_min = tf.math.reduce_min(x)
    x_max = tf.math.reduce_max(x)
    x_range = x_max - x_min
    return tf.math.pow((x - x_min) / x_range, gamma) * x_range + x_min, y


@tf.function
def random_noise_tf(x, y, param_range=(0.01, 0.05)):
    sigma = tf.random.uniform((1,), param_range[0], param_range[1])
    noise = tf.random.normal(tf.shape(x), mean=0, stddev=sigma)
    return x + noise, y


@tf.function
def random_contrast_tf(x, y, param_range=(0, 2)):
    x_mean = tf.math.reduce_mean(x)
    alpha = tf.random.uniform((1,), param_range[0], param_range[1])
    return (x - x_mean) * alpha + x_mean, y


@tf.function
def random_brightness_tf(x, y, param_range=(-50, 100)):
    return x + tf.random.uniform((1,), param_range[0], param_range[1]), y


@tf.function
def augment(x, y, func_list, p=0.5):
    branch = tf.random.shuffle(tf.range(len(func_list)))  # shuffle order of transformations
    random_p = tf.random.uniform((len(func_list),), 0, 1)
    
    if isinstance(p, float) and 0 <= p <= 1:
        cond = tf.math.less(random_p, [p] * len(func_list))  # conditions based on p < probability
    elif isinstance(p, (list, tuple)) and len(p) == len(func_list):
        cond = tf.math.less(random_p, p)  # conditions based on p < probability
    else:
        raise ValueError('Augmentation probability must be a float between 0 and 1 '\
            'or a list of floats of length equal to augmentation operations.')

    for i in range(len(func_list)):
        # the following line must be inside for loop, not outside, to make the partial function use the new x
        branch_fns = {j: functools.partial(func_list[j], x, y) for j in range(len(func_list))}
        x, y = tf.cond(cond[branch[i]],
                    true_fn=lambda: tf.switch_case(branch[i], branch_fns),  # apply transformation
                    false_fn=lambda: (x, y),  # do not apply transformation
        )
    return x, y

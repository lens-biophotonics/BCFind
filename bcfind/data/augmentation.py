import SimpleITK as sitk
import numpy as np
import tensorflow as tf

from scipy import ndimage


rng = np.random.default_rng()


def random_crop(x, target_shape):
    high = np.array(x.shape[-3:]) - target_shape
    high[high == 0] = 1
    f = rng.integers(0, high)
    t = f + target_shape

    output = x[..., f[0] : t[0], f[1] : t[1], f[2] : t[2]]
    return output


@tf.function(experimental_relax_shapes=True)
def random_crop_tf(input, target_shape=(50, 100, 100)):
    output = tf.numpy_function(random_crop, [input, target_shape], tf.float32)

    input_shape = input.get_shape().as_list()
    shape = (*input_shape[: len(input_shape) - 3], *target_shape)
    return tf.ensure_shape(output, shape)


@tf.function
def random_zoom_tf(input, param_range=(1.0, 1.1), order=1):
    def sitk_zoom(x, param_range=param_range, order=order):
        if order == 0:
            interpolator = sitk.sitkNearestNeighbor
        elif order == 1:
            interpolator = sitk.sitkLinear
        elif order == 2:
            interpolator = sitk.sitkBSplineResamplerOrder2
        elif order == 3:
            interpolator = sitk.sitkBSplineResamplerOrder3
        elif order == 4:
            interpolator = sitk.sitkBSplineResamplerOrder4
        elif order == 5:
            interpolator = sitk.sitkBSplineResamplerOrder5
        else:
            raise ValueError("Order of interpolation must be between 0 and 5")

        zoom_factor = rng.uniform(param_range[0], param_range[1])

        def resample(volume):
            img = sitk.GetImageFromArray(volume)
            img_size = img.GetSize()

            new_size = [
                int(round(img_size[i] * zoom_factor)) for i in range(len(img_size))
            ]

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetSize(new_size)
            resampler.SetOutputSpacing(np.divide(img_size, new_size))
            resampler.SetOutputOrigin(img.GetOrigin())
            resampler.SetOutputDirection(img.GetDirection())
            resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
            resampler.SetInterpolator(interpolator)
            resampler.SetDefaultPixelValue(0)

            new_volume = resampler.Execute(img)
            new_volume = sitk.GetArrayFromImage(new_volume)
            return new_volume

        if len(x.shape) > 3:
            outputs = []
            for i in range(x.shape[0]):
                outputs.append(resample(x[i])[np.newaxis, ...])
            output = np.concatenate(outputs, 0)

        else:
            output = resample(x)

        output = random_crop(output, x.shape[-3:])
        return output

    output = tf.numpy_function(sitk_zoom, [input], tf.float32)
    return tf.ensure_shape(output, input.shape)


@tf.function
def random_90rotation_tf(input, param_range=(0, 270), axes=(-2, -1), mode="constant"):
    def scipy_rotate(x, param_range=param_range, axes=axes):
        angles = np.arange(param_range[0], param_range[1] + 1, 90)
        angle = rng.choice(angles)

        if len(axes) > 2:
            axes = rng.choice(axes, size=2, replace=False)

        output = ndimage.rotate(x, angle, axes=axes, reshape=False, mode=mode)
        return output

    output = tf.numpy_function(scipy_rotate, [input], tf.float32)
    return tf.ensure_shape(output, input.shape)


@tf.function
def random_flip_tf(input, axes=(-2)):
    def numpy_flip(x, axes=axes):
        axis = rng.choice(axes)
        output = np.flip(x, axis=axis)
        return output

    output = tf.numpy_function(numpy_flip, [input], tf.float32)
    return tf.ensure_shape(output, input.shape)


@tf.function
def random_blur_tf(input, param_range=(0.01, 0.06)):
    def scipy_blur(x, param_range=param_range):
        sigma = [rng.uniform(param_range[0], param_range[1])] * len(x.shape)

        if len(sigma) > 3:
            sigma[:-3] = [
                0,
            ] * (len(x.shape) - 3)

        output = ndimage.gaussian_filter(x, sigma)
        return output

    output = tf.numpy_function(scipy_blur, [input], tf.float32)
    return tf.ensure_shape(output, input.shape)


@tf.function
def random_gamma_tf(input, param_range=(0.8, 1.2)):
    gamma = tf.random.uniform((1,), param_range[0], param_range[1])
    x_min = tf.math.reduce_min(input)
    x_max = tf.math.reduce_max(input)
    x_range = x_max - x_min

    output = tf.math.pow((input - x_min) / x_range, gamma) * x_range + x_min
    return output


@tf.function
def random_noise_tf(input, param_range=(0.001, 0.02)):
    sigma = tf.random.uniform((1,), param_range[0], param_range[1])
    noise = tf.random.normal(tf.shape(input), mean=0, stddev=sigma)
    return input + noise


@tf.function
def random_contrast_tf(input, param_range=(0, 2)):
    x_mean = tf.math.reduce_mean(input)
    alpha = tf.random.uniform((1,), param_range[0], param_range[1])
    return (input - x_mean) * alpha + x_mean


@tf.function
def random_brightness_tf(input, param_range=(-0.06, 0.06)):
    x_min = tf.reduce_min(input)
    x_max = tf.reduce_max(input)

    output = input + tf.random.uniform((1,), param_range[0], param_range[1])
    return clip_tf(output, x_min, x_max)


@tf.function
def clip_tf(input, vmin, vmax):
    output = tf.where(input < vmin, vmin, input)
    output = tf.where(input > vmax, vmax, output)
    return output


class Lambda(tf.keras.layers.Layer):
    def __init__(self, func, args=None, **kwargs):
        super(Lambda, self).__init__(**kwargs)
        self._func = func
        if args is not None:
            self.args = args
        else:
            self.args = {}

    def call(self, input):
        return self._func(input, **self.args)


@tf.function
def augment(xy, augmentations, p=0.3):
    geometric_ops = ["rotation", "flip", "zoom", "crop"]

    op_dict = get_op_dict(augmentations)
    op_names = list(op_dict.keys())
    op_index = np.arange(len(op_dict))
    np.random.shuffle(op_index)  # shuffle order of transformations

    random_p = np.random.uniform(0, 1, size=len(op_dict))

    # conditions based on p < probability
    if isinstance(p, float) and 0 <= p <= 1:
        cond = random_p < [
            p,
        ] * len(op_dict)
    elif isinstance(p, (list, tuple)) and len(p) == len(op_dict):
        cond = random_p < p
    else:
        raise ValueError(
            "Augmentation probability must be a float between 0 and 1 "
            "or a list of floats whith lenght equal to augmentation operations."
        )

    # apply transformations in random order
    for i in op_index:
        if cond[i]:
            if op_names[i] in geometric_ops:
                xy = op_dict[op_names[i]](xy)
            else:
                x = op_dict[op_names[i]](xy[0])
                xy = tf.concat((x[tf.newaxis], xy[1][tf.newaxis]), axis=0)
    return xy


def get_op_dict(augmentations):
    """Returns a dict of callable operations from a list or dictionary of default or custom augmentations.

    Args:
        augmentations (list, tuple, dict): list or tuple of strings/callables or both. String elements will use default values of implemented operations.
                                    Strings must be one of [\'brightness\', \'contrast\', \'gamma\', \'noise\'].
                                    Callable elements will be called, better if they are tensorflow.functions.
                                      dict of lists/callables or both. List values will be the parameter range of implemented operations.
                                    Keys of list values must be on of [\'brightness\', \'contrast\', \'gamma\', \'noise\'].
                                    Callable values will be called, better if they are tensorflow.operations.
                                    Keys of callable values must be different from implemented operation names.
                                    Callables must take, in either cases, the tensor of an input image and return its augmented version.

    Raises:
        ValueError: if args are bad specified.

    Returns:
        list: list of tensorflow callable operations.
    """
    implemented_ops = {
        "crop": random_crop_tf,
        "gamma": random_gamma_tf,
        "contrast": random_contrast_tf,
        "brightness": random_brightness_tf,
        "noise": random_noise_tf,
        "rotation": random_90rotation_tf,
        "flip": random_flip_tf,
        "zoom": random_zoom_tf,
        "blur": random_blur_tf,
    }

    ops_dict = {}
    if isinstance(augmentations, (list, tuple)):
        c = 0
        for op in augmentations:
            if isinstance(op, str):
                assert op in implemented_ops, f"{op} not in {implemented_ops}."
                ops_dict[op] = Lambda(implemented_ops[op], augmentations[op], name=op)

            elif callable(op):
                ops_dict[f"custom_op_{c}"] = op
                c += 1

            else:
                raise ValueError(f"{op} is neither a string nor a callable.")

    elif isinstance(augmentations, dict):
        for op in augmentations:
            if op in implemented_ops:
                ops_dict[op] = Lambda(implemented_ops[op], augmentations[op], name=op)

            elif callable(augmentations[op]):
                ops_dict[op] = augmentations[op]

            else:
                raise ValueError(
                    f"{op} value is neither a list of parameter range nor a callable."
                )
    else:
        raise ValueError("augmentations is neither a list nor a dictionary.")

    return ops_dict

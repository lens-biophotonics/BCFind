import numpy as np

from scipy import ndimage
from skimage import filters


def random_gamma(x_batch, gamma_range):
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])

    for i, x in enumerate(x_batch):
        x_min = x.min()
        x_range = x.max() - x_min
        if x_range == 0:
            continue
        x_batch[i] = np.power((x - x_min) / x_range, gamma) * x_range + x_min

    return x_batch


def random_contrast(x_batch, alpha_range):
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])

    for i, x in enumerate(x_batch):
        x_mean = x.mean()
        x_batch[i] = (x - x_mean) * alpha + x_mean

    return x_batch


def random_brightness(x_batch, alpha_range):
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    return x_batch + alpha


def random_zoom(x_batch, y_batch, alpha_range):
    if alpha_range[0] < 1:
        raise ValueError('Zoom factor must be > 1')

    original_shape = np.array(x_batch.shape)
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])

    rank = len(x_batch.shape)

    zoom = np.ones(rank)
    zoom[1:-1] = alpha
    x_batch = ndimage.zoom(x_batch, zoom, order=0, prefilter=False)
    y_batch = ndimage.zoom(y_batch, zoom, order=0, prefilter=False)

    # random crop
    high = x_batch.shape[1:-1] - original_shape[1:-1]
    high[high == 0] = 1
    f = np.random.randint(0, high)
    t = f + original_shape[1:-1]

    return x_batch[:, f[0]:t[0], f[1]:t[1], f[2]:t[2]],  y_batch[:, f[0]:t[0], f[1]:t[1], f[2]:t[2]]


def random_gauss_filter(x_batch, sigma_range):
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    for i, x in enumerate(x_batch):
        x_batch[i] = filters.gaussian(x, sigma=sigma)

    return x_batch


def random_noise(x_batch, sigma_range):
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    noise = np.random.normal(loc=0, scale=sigma, size=x_batch.shape)

    return x_batch + noise


def random_rotation(x_batch, y_batch, rotation_angles):
    angle = np.random.choice(rotation_angles)
    
    for i in range(x_batch.shape[0]):
        x_batch[i] = ndimage.interpolation.rotate(x_batch[i], angle, axes=(0, 1))
        y_batch[i] = ndimage.interpolation.rotate(y_batch[i], angle, axes=(0, 1))
    
    return x_batch, y_batch

def random_flip(x_batch, y_batch, axes):
    axis = np.random.choice(axes) + 1
    
    x_batch = np.flip(x_batch, axis=axis)
    y_batch = np.flip(y_batch, axis=axis)
    return x_batch, y_batch
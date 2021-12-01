import functools as ft

import numpy as np
from scipy import ndimage

from skimage import filters


def random_gamma(x_batch, gamma_range, p):
    if np.random.uniform(0, 1) > p:
        return x_batch

    gamma = np.random.uniform(gamma_range[0], gamma_range[1])

    for i, x in enumerate(x_batch):
        x_min = x.min()
        x_range = x.max() - x_min
        if x_range == 0:
            continue
        x_batch[i] = np.power((x - x_min) / x_range, gamma) * x_range + x_min

    return x_batch


def random_contrast(x_batch, alpha_range, p):
    if np.random.uniform(0, 1) > p:
        return x_batch

    alpha = np.random.uniform(alpha_range[0], alpha_range[1])

    for i, x in enumerate(x_batch):
        x_mean = x.mean()
        x_batch[i] = (x - x_mean) * alpha + x_mean

    return x_batch


def random_brightness(x_batch, alpha_range, p):
    if np.random.uniform(0, 1) > p:
        return x_batch

    alpha = np.random.uniform(alpha_range[0], alpha_range[1])

    return x_batch + alpha


def random_zoom(x_batch, alpha_range, p):
    if alpha_range[0] < 1:
        raise ValueError('Zoom factor must be > 1')

    if np.random.uniform(0, 1) > p:
        return x_batch

    original_shape = np.array(x_batch.shape)
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])

    rank = len(x_batch.shape)

    zoom = np.ones(rank)
    zoom[1:-1] = alpha
    x_batch = ndimage.zoom(x_batch, zoom, order=0, prefilter=False)

    # random crop
    high = x_batch.shape[1:-1] - original_shape[1:-1]
    high[high == 0] = 1
    f = np.random.randint(0, high)
    t = f + original_shape[1:-1]

    return x_batch[:, f[0]:t[0], f[1]:t[1], f[2]:t[2]]


def random_gauss_filter(x_batch, sigma_range, p):
    if np.random.uniform(0, 1) > p:
        return x_batch

    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    for i, x in enumerate(x_batch):
        x_batch[i] = filters.gaussian(x, sigma=sigma)

    return x_batch


def random_noise(x_batch, sigma_range, p):
    if np.random.uniform(0, 1) > p:
        return x_batch

    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    noise = np.random.normal(loc=0, scale=sigma, size=x_batch.shape)

    return x_batch + noise


class Augmentor:
    def __init__(self, data_gen):
        self.data_gen = data_gen
        self.operations = dict()
        self.it = 0

    def add_random_gamma(self, gamma_range, p=0.5):
        self.operations['gamma'] = ft.partial(random_gamma, gamma_range=gamma_range, p=p)

    def add_random_contrast(self, alpha_range, p=0.5):
        self.operations['contrast'] = ft.partial(random_contrast, alpha_range=alpha_range, p=p)

    def add_random_brightness(self, alpha_range, p=0.5):
        self.operations['brightness'] = ft.partial(random_brightness, alpha_range=alpha_range, p=p)

    def add_random_zoom(self, alpha_range, p=0.5):
        self.operations['zoom'] = ft.partial(random_zoom, alpha_range=alpha_range, p=p)

    def add_random_gauss_filter(self, sigma_range, p=0.5):
        self.operations['gauss_filter'] = ft.partial(random_gauss_filter, sigma_range=sigma_range, p=p)

    def add_random_noise(self, sigma_range, p=0.5):
        self.operations['noise'] = ft.partial(random_noise, sigma_range=sigma_range, p=p)

    def __len__(self):
        return len(self.data_gen)

    def __getitem__(self, idx):
        x, y = self.data_gen[idx]
        for op in self.operations:
            if op == 'zoom':
                x = self.operations[op](x)
                y = self.operations[op](y)
            else:
                x = self.operations[op](x)

        return x, y

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item

    def __next__(self):
        if self.it <= len(self):
            item = self[self.it]
            self.it += 1
            return item
        else:
            raise StopIteration

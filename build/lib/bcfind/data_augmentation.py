import cupy as cp
import numpy as np
import functools as ft

from cucim.skimage import filters as cm_skim_filt
from cucim.skimage import transform as cm_skim_trans

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


def random_gamma(x_batch, gamma_range, p):
    if cp.random.uniform(0, 1) < p:
        x_batch = cp.asarray(x_batch)
        gamma = cp.random.uniform(gamma_range[0], gamma_range[1])

        new_x_batch = cp.zeros_like(x_batch)
        for i, x in enumerate(x_batch):
            x_min = x.min()
            x_range = x.max() - x_min
            new_x_batch[i] = cp.power((x - x_min) / x_range, gamma) * x_range + x_min
        
        new_x_batch = cp.asnumpy(new_x_batch)

        del x_batch, gamma
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return new_x_batch
    else:
        return x_batch


def random_contrast(x_batch, alpha_range, p):
    if cp.random.uniform(0, 1) < p:
        alpha = cp.random.uniform(alpha_range[0], alpha_range[1])

        new_x_batch = cp.zeros_like(x_batch)
        for i, x in enumerate(x_batch):
            x = cp.asarray(x)
            x_mean = x.mean()
            new_x_batch[i] = (x - x_mean) * alpha + x_mean

        new_x_batch = cp.asnumpy(new_x_batch)

        del x, alpha
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return new_x_batch

    else:
        return x_batch


def random_brightness(x_batch, alpha_range, p):
    if cp.random.uniform(0, 1) < p:
        x_batch = cp.asarray(x_batch)
        alpha = cp.random.uniform(alpha_range[0], alpha_range[1])

        new_x_batch = x_batch + alpha
        new_x_batch = cp.asnumpy(new_x_batch)

        del x_batch, alpha
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return new_x_batch

    else:
        return x_batch


def random_zoom(x_batch, y_batch, alpha_range, p):
    if cp.random.uniform(0, 1) < p:
        alpha = np.random.uniform(alpha_range[0], alpha_range[1])

        new_shape = np.array(x_batch.shape)
        new_shape[1:-1] = np.round(new_shape[1:-1] * alpha).astype(int)
        new_x_batch = cp.zeros(new_shape)
        new_y_batch = cp.zeros(new_shape)
        for i, (x, y) in enumerate(zip(x_batch, y_batch)):
            x = cp.asarray(cp.squeeze(x))
            y = cp.asarray(cp.squeeze(y))

            new_x = cm_skim_trans.rescale(x, scale=alpha)
            new_y = cm_skim_trans.rescale(y, scale=alpha)
            new_x_batch[i] = new_x[..., cp.newaxis]
            new_y_batch[i] = new_y[..., cp.newaxis]
        
        s0 = cp.random.randint(0, max(1, new_shape[1] - x_batch.shape[1]))
        s1 = cp.random.randint(0, max(1, new_shape[2] - x_batch.shape[2]))
        s2 = cp.random.randint(0, max(1, new_shape[3] - x_batch.shape[3]))
        s = cp.array([s0, s1, s2])
        e = s + cp.array(x_batch.shape[1:-1])

        new_x_batch = new_x_batch[:, s[0]:e[0], s[1]:e[1], s[2]:e[2], :]
        new_y_batch = new_y_batch[:, s[0]:e[0], s[1]:e[1], s[2]:e[2], :]

        new_x_batch = cp.asnumpy(new_x_batch)
        new_y_batch = cp.asnumpy(new_y_batch)

        del x, y, new_x, new_y, s0, s1, s2, s, e
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return new_x_batch, new_y_batch

    else:
        return x_batch, y_batch


def random_gauss_filter(x_batch, sigma_range, p):
    if cp.random.uniform(0, 1) < p:
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])

        new_x_batch = cp.zeros_like(x_batch)
        for i, x in enumerate(x_batch):
            x = cp.asarray(x)
            new_x_batch[i] = cm_skim_filt.gaussian(x, sigma=sigma)
        
        new_x_batch = cp.asnumpy(new_x_batch)

        del x
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return new_x_batch
    else:
        return x_batch


def random_noise(x_batch, sigma_range, p):
    if cp.random.uniform(0, 1) < p:
        x_batch = cp.asarray(x_batch)
        sigma = cp.random.uniform(sigma_range[0], sigma_range[1])
        noise = cp.random.normal(loc=0, scale=sigma, size=x_batch.shape)

        new_x_batch = x_batch + noise

        del x_batch, sigma, noise
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return cp.asnumpy(new_x_batch)
    else:
        return x_batch


class Augmentor():
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
                x, y = self.operations[op](x, y)
            else:
                x = self.operations[op](x)

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
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

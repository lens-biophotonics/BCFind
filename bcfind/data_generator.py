import numpy as np
import functools as ft
import tensorflow as tf
import numpy.random as rs

import bcfind.data_augmentation as bcf_aug


class Scaler:
    def __init__(self, norm_method="input", stand_method="none"):
        """TODO describe function

        :param norm_method:
        :type norm_method:
        :param stand_method:
        :type stand_method:
        :returns:

        """
        admitted_methods = ["input", "data", "none"]
        if norm_method not in admitted_methods:
            raise ValueError(
                f"Normalization method must be one of {admitted_methods}."
                f" But received {norm_method} instead."
            )
        if stand_method not in admitted_methods:
            raise ValueError(
                f"Standardization method must be one of {admitted_methods}."
                f" But received {stand_method} instead."
            )

        self.norm_method = norm_method
        self.stand_method = stand_method
        self.X_min = None
        self.X_range = None
        self.X_mean = None
        self.X_std = None

    def _normalize(self, X):
        """TODO describe function

        :param X:
        :type X:
        :returns:

        """
        if self.norm_method == "input":
            for i, x in enumerate(X):
                x_min = x.min()
                x_range = x.max() - x_min
                X[i] = (x - x_min) / x_range

        elif self.norm_method == "data":
            if self.X_min is None or self.X_range is None:
                raise ValueError("Scaler object must be fitted first")
            X = (X - self.X_min) / self.X_range

        return X

    def _standardize(self, X):
        """TODO describe function

        :param X: a
        :type X:
        :returns:

        """
        if self.stand_method == "input":
            for i in range(X.shape[0]):
                X[i] = (X[i] - X[i].mean()) / X[i].std()

        elif self.stand_method == "data":
            if self.X_mean is None or self.X_std is None:
                raise ValueError("Scaler object must be fitted first")
            X = (X - self.X_mean) / self.X_std

        return X

    def fit(self, X):
        if self.norm_method != "none":
            if self.norm_method == "data":
                self.X_min = X.min()
                self.X_range = X.max() - self.X_min
            new_X = self._normalize(X)

        if self.stand_method != "none":
            if self.stand_method == "data":
                self.X_mean = X.mean()
                self.X_std = X.std()
            new_X = self._standardize(X)

        return new_X

    def transform(self, X):
        if self.norm_method != "none":
            X = self._normalize(X)
        if self.stand_method != "none":
            X = self._standardize(X)
        return X


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        X,
        Y,
        batch_size,
        output_shape,
        augment=False
    ):
        self.X = X
        self.Y = Y
        self.indices = list(range(self.X.shape[0]))
        self.batch_size = batch_size
        self.output_shape = output_shape
        self.augment = augment
        self.operations = dict()
        self.it = 0
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def on_epoch_end(self):
        print('Epoch_end!')
        self.it = 0
        rs.shuffle(self.indices)
        self.X = self.X[self.indices]
        self.Y = self.Y[self.indices]

    def add_random_gamma(self, gamma_range, p=0.5):
        self.operations['gamma'] = ft.partial(bcf_aug.random_gamma, gamma_range=gamma_range), p

    def add_random_contrast(self, alpha_range, p=0.5):
        self.operations['contrast'] = ft.partial(bcf_aug.random_contrast, alpha_range=alpha_range), p
    
    def add_random_brightness(self, alpha_range, p=0.5):
        self.operations['brightness'] = ft.partial(bcf_aug.random_brightness, alpha_range=alpha_range), p

    def add_random_zoom(self, alpha_range, p=0.5):
        self.operations['zoom'] = ft.partial(bcf_aug.random_zoom, alpha_range=alpha_range), p
    
    def add_random_gauss_filter(self, sigma_range, p=0.5):
        self.operations['gauss_filter'] = ft.partial(bcf_aug.random_gauss_filter, sigma_range=sigma_range), p
    
    def add_random_noise(self, sigma_range, p=0.5):
        self.operations['noise'] = ft.partial(bcf_aug.random_noise, sigma_range=sigma_range), p
    
    def add_random_rotation(self, rotation_angles, p=0.5):
        self.operations['rotate'] = ft.partial(bcf_aug.random_rotation, rotation_angles=rotation_angles), p

    def add_random_flip(self, axes, p=0.5):
        self.operations['flip'] = ft.partial(bcf_aug.random_flip, axes=axes), p
    
    def _random_crop(self, x_batch, y_batch):
        lshape = x_batch[0].shape
        assert lshape[2] >= self.output_shape[2]

        s0 = rs.randint(0, max(1, lshape[0] - self.output_shape[0]))
        s1 = rs.randint(0, max(1, lshape[1] - self.output_shape[1]))
        s2 = rs.randint(0, max(1, lshape[2] - self.output_shape[2]))

        s = np.array([s0, s1, s2])
        e = s + self.output_shape

        region_x = x_batch[:, s[0] : e[0], s[1] : e[1], s[2] : e[2]]
        region_y = y_batch[:, s[0] : e[0], s[1] : e[1], s[2] : e[2]]

        return region_x.astype("float32"), region_y.astype("float32")

    def _augment_data(self, x_batch, y_batch):
        items = list(self.operations.items())
        np.random.shuffle(items)
        
        for name, (op, p) in items:
            if np.random.uniform() < p:
                if name == 'zoom' or name == 'rotate' or name == 'flip':
                    x_batch, y_batch = op(x_batch, y_batch)
                else:
                    x_batch = op(x_batch)
        return x_batch, y_batch

    def __getitem__(self, idx):
        x_batch = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        y_batch = self.Y[idx * self.batch_size : (idx + 1) * self.batch_size]

        x_batch, y_batch = self._random_crop(x_batch, y_batch)
        
        if self.augment:
            x_batch, y_batch = self._augment_data(x_batch, y_batch)

        return x_batch[..., np.newaxis], y_batch[..., np.newaxis]

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
    
    def __next__(self):
        if self.it <= len(self):
            item = self[self.it]
            self.it += 1
            return item
        else:
            self.on_epoch_end()
            raise StopIteration


def get_train_val_idx(n, val_fold, seed=123):
    indices = np.arange(n)
    k_folds = int(val_fold.split("/")[-1])
    fold_idx = int(val_fold.split("/")[0])
    val_size = n // k_folds

    val = list(np.arange((fold_idx - 1) * val_size, fold_idx * val_size))
    train = list(set(indices) - set(val))

    if seed is None:
        rs.shuffle(indices)
        return indices[train], indices[val]
    else:
        rs.seed(seed)
        rs.shuffle(indices)
        return indices[train], indices[val]


def get_tf_data(
        X,
        Y,
        batch_size,
        output_shape,
        val_fold=None,
        val_seed=123,
        augment=False,
        brightness=[-1, 1],
        gamma=[0.5, 1.5],
        contrast=None,
        zoom=[1., 1.5],
        gauss_filter=[0.2, 3],
        noise=[0.2, 3],
):
    TRAIN_PREFETCH = 0
    VAL_PREFETCH = 0

    if val_fold is not None:
        train_idx, val_idx = get_train_val_idx(X.shape[0], val_fold, val_seed)

        if augment:
            train_gen = BatchGenerator(
                X[train_idx],
                Y[train_idx],
                batch_size,
                output_shape,
                augment=True,
            )
            if brightness is not None:
                train_gen.add_random_brightness(brightness)
            if gamma is not None:
                train_gen.add_random_gamma(gamma)
            if contrast is not None:
                train_gen.add_random_contrast(contrast)
            if zoom is not None:
                train_gen.add_random_zoom(zoom)
            if gauss_filter is not None:
                train_gen.add_random_gauss_filter(gauss_filter)
            if noise is not None:
                train_gen.add_random_noise(noise)
        
        else:
            train_gen = BatchGenerator(
                X[train_idx],
                Y[train_idx],
                batch_size,
                output_shape,
                augment=False
            )

        train = tf.data.Dataset.from_generator(
            lambda: train_gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, *output_shape, 1], [None, *output_shape, 1]),
        )


        val_gen = BatchGenerator(
            X[val_idx],
            Y[val_idx],
            batch_size,
            output_shape,
            augment=False
        )
        val = tf.data.Dataset.from_generator(
            lambda: val_gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, *output_shape, 1], [None, *output_shape, 1]),
        )

        train = train.prefetch(TRAIN_PREFETCH)
        val = val.prefetch(VAL_PREFETCH)
        return train, val

    else:
        if augment:
            train_gen = BatchGenerator(
                X,
                Y,
                batch_size,
                output_shape,
                augment=True,
            )
            if brightness is not None:
                train_gen.add_random_brightness(brightness)
            if gamma is not None:
                train_gen.add_random_gamma(gamma)
            if contrast is not None:
                train_gen.add_random_contrast(contrast)
            if zoom is not None:
                train_gen.add_random_zoom(zoom)
            if gauss_filter is not None:
                train_gen.add_random_gauss_filter(gauss_filter)
            if noise is not None:
                train_gen.add_random_noise(noise)
        else:
            train_gen = BatchGenerator(
                X,
                Y,
                batch_size,
                output_shape,
                augment=False,
            )

        train = tf.data.Dataset.from_generator(
            lambda: train_gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, *output_shape, 1], [None, *output_shape, 1]),
        )
        train = train.prefetch(TRAIN_PREFETCH)
        return train

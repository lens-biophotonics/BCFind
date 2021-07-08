import h5py
import numpy as np
import tensorflow as tf


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, Y, batch_size, output_shape):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.output_shape = output_shape
        self.indices = list(range(self.X.shape[0]))

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        self.X = self.X[self.indices]
        self.Y = self.Y[self.indices]

    def _random_crop(self, x_batch, y_batch):
        lshape = x_batch[0].shape
        assert lshape[2] >= self.output_shape[2]

        ri = np.random.randint
        s0 = ri(0, max(1, lshape[0] - self.output_shape[0]))
        s1 = ri(0, max(1, lshape[1] - self.output_shape[1]))
        s2 = ri(0, max(1, lshape[2] - self.output_shape[2]))

        s = np.array([s0, s1, s2])
        e = s + self.output_shape

        region_x = x_batch[:, s[0] : e[0], s[1] : e[1], s[2] : e[2]]
        region_y = y_batch[:, s[0] : e[0], s[1] : e[1], s[2] : e[2]]

        # add channel axis for conv3D layers
        region_x = region_x[..., tf.newaxis].astype("float32") / 255
        region_y = region_y[..., tf.newaxis].astype("float32") / 255
        return region_x, region_y

    def __getitem__(self, idx):
        x_batch = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        y_batch = self.Y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return self._random_crop(x_batch, y_batch)

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item


def get_train_val_idx(n, val_fold):
    k_folds = int(val_fold.split("/")[-1])
    fold_idx = int(val_fold.split("/")[0])
    val_size = n // k_folds

    val_idx = list(np.arange((fold_idx - 1) * val_size, fold_idx * val_size))
    train_idx = list(set(np.arange(n)) - set(val_idx))

    return train_idx, val_idx


def get_tf_data(x_file, y_file, batch_size, output_shape, val_fold=None):
    with h5py.File(x_file, "r") as fx:
        n = fx["x"].shape[0]
        X = fx["x"][()]

    with h5py.File(y_file, "r") as fy:
        assert n == fy["y"].shape[0], "Inputs and targets do not have the same length"
        Y = fy["y"][()]

    if val_fold is not None:
        train_idx, val_idx = get_train_val_idx(X.shape[0], val_fold)

        train_gen = BatchGenerator(
            X[train_idx, ...], Y[train_idx, ...], batch_size, output_shape
        )
        train = tf.data.Dataset.from_generator(
            lambda: train_gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, *output_shape, 1], [None, *output_shape, 1]),
        )
        train = train.prefetch(3)

        val_gen = BatchGenerator(
            X[val_idx, ...], Y[val_idx, ...], batch_size, output_shape
        )
        val = tf.data.Dataset.from_generator(
            lambda: val_gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, *output_shape, 1], [None, *output_shape, 1]),
        )
        val = val.prefetch(3)
        return train, val

    else:
        batch_gen = BatchGenerator(X, Y, batch_size, output_shape)
        data = tf.data.Dataset.from_generator(
            lambda: batch_gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, *output_shape, 1], [None, *output_shape, 1]),
        )
        data = data.prefetch(3)
        return data

import h5py
import numpy as np
import tensorflow as tf


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_file, y_file, batch_size, output_shape):
        with h5py.File(x_file, "r") as fx:
            n = fx["x"].shape[0]
            self.x_train = fx["x"][()]

        with h5py.File(y_file, "r") as fy:
            assert (
                n == fy["y"].shape[0]
            ), "Inputs and targets do not have the same length"
            self.y_train = fy["y"][()]

        self.batch_size = batch_size
        self.output_shape = output_shape
        self.indices = list(range(self.x_train.shape[0]))

    def __len__(self):
        return int(np.ceil(self.x_train.shape[0] / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        self.x_train = self.x_train[self.indices]
        self.y_train = self.y_train[self.indices]

    def _random_crop(
        self, x_batch, y_batch, min_x_mean=20 / 255.0, min_y_mean=0.05 / 255.0
    ):
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
        region_x = region_x[..., np.newaxis].astype("float32") / 255.0
        region_y = region_y[..., np.newaxis].astype("float32") / 255.0
        return region_x, region_y

    def __getitem__(self, idx):
        x_batch = self.x_train[idx * self.batch_size : (idx + 1) * self.batch_size]
        y_batch = self.y_train[idx * self.batch_size : (idx + 1) * self.batch_size]
        return self._random_crop(x_batch, y_batch)

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item


def get_tf_data(x_file, y_file, batch_size, output_shape):
    batch_gen = BatchGenerator(x_file, y_file, batch_size, output_shape)

    data = tf.data.Dataset.from_generator(
        lambda: batch_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, *output_shape, 1], [None, *output_shape, 1]),
    )
    data = data.prefetch(3)
    return data

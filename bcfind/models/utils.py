import itertools
import tensorflow as tf


def predict(input, model):
    if tf.rank(input) == 3:
        input = input[tf.newaxis, ..., tf.newaxis]

    I, J = 4, 4
    for i, j in itertools.product(range(I), range(J)):
        if i == 0  and j == 0:
            pad_x = tf.identity(input)
            continue
        try:
            print('Input shape =', pad_x.shape)
            pred = model(pad_x, training=False)
            break
        except (tf.errors.InvalidArgumentError, ValueError) as e:
            print('Invalid input shape for concat layer. Try padding')
            paddings = tf.constant([[0, 0], [0, j], [0, i], [0, i], [0, 0]])
            pad_x = tf.pad(input, paddings)

            if i == I-1 and j == J-1:
                raise e

    pred = tf.sigmoid(tf.squeeze(pred)).numpy()
    return pred
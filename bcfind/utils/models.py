import itertools
import tensorflow as tf


def predict(input, model):
    if tf.rank(input) == 3:
        input = input[tf.newaxis, ..., tf.newaxis]

    I, J = 8, 8
    for i, j in itertools.product(range(I), range(J)):
        if i == 0 and j == 0:
            x = tf.identity(input)
            continue
        try:
            print("Input shape =", x.shape)
            pred = model(x, training=False)
            break
        except (tf.errors.InvalidArgumentError, ValueError) as e:
            print("Invalid input shape for concat layer. Extracting slice")
            x = tf.slice(
                input,
                [0, 0, 0, 0, 0],
                size=[
                    input.shape[0],
                    input.shape[1] - j,
                    input.shape[2] - i,
                    input.shape[3] - i,
                    input.shape[4],
                ],
            )

            if i == I - 1 and j == J - 1:
                raise e

    pred = tf.sigmoid(tf.squeeze(pred))
    return pred

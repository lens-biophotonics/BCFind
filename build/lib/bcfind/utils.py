import numpy as np
import skimage.transform as sk_trans


def iround(val):
    return int(round(val))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def metrics(df):
    # Precision
    den = np.sum(df.TP + df.FP)
    if den != 0.0:
        prec = np.sum(df.TP) / den
    else:
        prec = 1.0

    # Recall
    den = np.sum(df.TP + df.FN)
    if den != 0.0:
        rec = np.sum(df.TP) / den
    else:
        rec = 1.0

    # F1
    if prec + rec != 0.0:
        f1 = 2.0 * prec * rec / (prec + rec)
    else:
        f1 = 0.0

    # Accuracy
    den = np.sum(df.TP + df.FP + df.FN)
    if den != 0.0:
        acc = np.sum(df.TP) / den
    else:
        acc = 1.0
    return {"prec": prec, "rec": rec, "f1": f1, "acc": acc}


def pad(x, output_shape):
    padded_x = np.zeros(output_shape)
    insert_here = tuple(slice(0, x.shape[dim]) for dim in range(len(x.shape)))
    padded_x[insert_here] = x
    return padded_x


def preprocessing(
    x,
    transpose=None,
    flip_axis=None,
    clip_threshold=None,
    gamma_correction=None,
    downscale=None,
    pad_output_shape=None,
):
    if transpose is not None:
        x = np.transpose(x, transpose)

    if flip_axis is not None:
        x = np.flip(x, axis=flip_axis)

    if clip_threshold is not None:
        x[np.where(x > clip_threshold)] = clip_threshold

    if gamma_correction is not None:
        x_min = x.min()
        x_range = x.max() - x_min
        x = np.power((x - x_min) / x_range, gamma_correction) * x_range + x_min

    if downscale is not None:
        x = sk_trans.rescale(x, downscale, anti_aliasing=False)

    if pad_output_shape is not None:
        if any([inp != out for inp, out in zip(x.shape, pad_output_shape)]):
            x = pad(x, pad_output_shape)

    return x


def remove_border_points_from_df(df, df_columns, data_shape, border_size):
    df = df.drop(df[df[df_columns[0]] <= border_size[0]].index)
    df = df.drop(df[df[df_columns[0]] >= data_shape[0] - border_size[0]].index)
    df = df.drop(df[df[df_columns[1]] <= border_size[1]].index)
    df = df.drop(df[df[df_columns[1]] >= data_shape[1] - border_size[1]].index)
    df = df.drop(df[df[df_columns[2]] <= border_size[2]].index)
    df = df.drop(df[df[df_columns[2]] >= data_shape[2] - border_size[2]].index)
    return df


def remove_border_points_from_array(array, data_shape, border_size):
    frame_centers = np.where(
        (array[:, 0] <= border_size[0])
        + (array[:, 1] <= border_size[1])
        + (array[:, 2] <= border_size[2])
        + (array[:, 0] >= data_shape[0] - border_size[0])
        + (array[:, 1] >= data_shape[1] - border_size[1])
        + (array[:, 2] >= data_shape[2] - border_size[2])
    )
    array = np.delete(array, frame_centers, axis=0)
    return array

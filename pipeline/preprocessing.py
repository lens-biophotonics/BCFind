import numpy as np
from sklearn.neighbors import LocalOutlierFactor



def auto_clip(x: np.ndarray) -> np.ndarray:
    """
    Automatically detect and clip outlier values using LocalOutlierFactor.

    Parameters:
      x:    ND array

    Returns:
      A float32 array with values above the maximum inlier capped.
    """
    try:
        # Prepare data for LOF (expects 2D)
        x_flat = x.reshape(-1, 1)
        # Fit LOF model to identify inliers (label 1)
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.2)
        preds = lof.fit_predict(x_flat)
        inlier_mask = preds == 1

        # Determine highest inlier value to use as clip threshold
        max_inlier = x_flat[inlier_mask].max()
        # Clip values exceeding the inlier maximum
        clipped = np.minimum(x, max_inlier)

        return clipped.astype(np.float32)
    except Exception as exc:
        raise Exception(
            f"^ Error occured when auto clipping during the preprocessing over a path.\n{exc}"
        )



def get_preprocessing_func(
        clip = "bit",
        clip_value = 15,
        center = None,
        center_value = 0,
        scale = "bit",
        scale_value  = 15,
    ):
    """
    Build a preprocessing function.
    Steps: clip values, center data, then scale values.

    Parameters:
      clip        : method for clipping values
      clip_value  : threshold or quantile for clipping
      center      : method for centering data
      center_value: constant offset if center='constant'
      scale       : method for scaling data
      scale_value : divisor if scale='constant' or 'bit'

    Returns:
      A function that applies clip, center, then scale to an np.ndarray.
    """
    try:
        # Validate choices
        valid = {
            "clip":   {"constant", "bit", "quantile", "auto", "none", None},
            "center": {"constant", "min", "mean", "none", None},
            "scale":  {"constant", "bit", "max", "std", "none", None},
        }
        if clip not in valid["clip"]:
            raise ValueError(f"Invalid clip: {clip!r}")
        if center not in valid["center"]:
            raise ValueError(f"Invalid center: {center!r}")
        if scale not in valid["scale"]:
            raise ValueError(f"Invalid scale: {scale!r}")

        # Dispatch tables
        clip_fns = {
            "constant": lambda x: np.minimum(x, clip_value),
            "bit":      lambda x: np.minimum(x, 2**clip_value),
            "quantile": lambda x: np.minimum(x, np.quantile(x, clip_value)),
            "auto":     auto_clip, 
            None:        lambda x: x,
        }

        center_fns = {
            "constant": lambda x: x - center_value,
            "min":      lambda x: x - x.min(), 
            "mean":     lambda x: x - x.mean(),
            None:        lambda x: x,
        }

        scale_fns = {
            "constant": lambda x: x / scale_value,
            "bit":      lambda x: x / 2**scale_value,
            "max":      lambda x: x / x.max(),
            "std":      lambda x: x / x.std(),
            "none":     lambda x: x,
            None:        lambda x: x,
        }

        # Select the functions based on user choice
        clip_fun   = clip_fns[clip]
        center_fun = center_fns[center]
        scale_fun  = scale_fns[scale]

        # preprocess pipeline
        def preprocess(x: np.ndarray) -> np.ndarray:
            # apply clip, then center, then scale in order
            for fn in (clip_fun, center_fun, scale_fun):
                x = fn(x)

            return x

        return preprocess
    except Exception as exc:
        raise Exception(
            f"^ Error occured when getting the preprrocessing function for the patches.\n{exc}"
        )

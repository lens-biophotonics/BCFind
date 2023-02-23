import os
import json
import lmdb
import pickle
import cupy as cp
import numpy as np
import pandas as pd
import hyperopt as ho
import functools as ft
import concurrent.futures as cf

from cupyx.scipy import ndimage as cpx_img
from skimage.feature.blob import _prune_blobs

from bcfind.localizers import bipartite_match
from bcfind.localizers.utils import get_counts_from_bm_eval
from bcfind.utils import (
    metrics,
    remove_border_points_from_array,
    remove_border_points_from_df,
)

# # Disable memory pool for device memory (GPU)
# cp.cuda.set_allocator(None)
# # Disable memory pool for pinned memory (CPU).
# cp.cuda.set_pinned_memory_allocator(None)


def cp_peak_local_max(image, threshold_rel=0.0, footprint=None):
    if footprint is None:
        footprint = cp.ones((3,) * image.ndim)

    maxima = image == cpx_img.maximum_filter(
        image, footprint=footprint, mode="constant", cval=0.0
    )

    thresh = cp.max(image) * threshold_rel
    maxima = cp.logical_and(maxima, image >= thresh)

    peaks = cp.transpose(cp.array(cp.where(maxima)))
    return peaks


def blob_dog(image, min_sigma, max_sigma, sigma_ratio, threshold_rel, overlap):
    r"""Equivalent function to skimage.feature.blob_dog which runs on the GPU, making computation 10 times faster.

    Finds blobs in the given grayscale image.
    Blobs are found using the Difference of Gaussian (DoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.
    Parameters
    ----------
    image : 2D or 3D ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        The minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    sigma_ratio : float, optional
        The ratio between the standard deviation of Gaussian Kernels used for
        computing the Difference of Gaussians
    threshold_rel : float, optional.
        A value between 0 and 1. The relative lower bound for scale space maxima.
        Local maxima smaller than max(image) * thresh are ignored.
        Reduce this to detect blobs with less intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.

    Returns
    -------
    A : (n, image.ndim + sigma) ndarray
        A 2d array with each row representing 2 coordinate values for a 2D
        image, and 3 coordinate values for a 3D image, plus the sigma(s) used.
        When a single sigma is passed, outputs are:
        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
        deviation of the Gaussian kernel which detected the blob. When an
        anisotropic gaussian is used (sigmas per dimension), the detected sigma
        is returned for each dimension.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach

    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """
    scalar_sigma = np.isscalar(max_sigma) and np.isscalar(min_sigma)

    if np.isscalar(min_sigma):
        min_sigma = np.array((min_sigma,) * image.ndim)
    else:
        min_sigma = np.array(min_sigma)
    if np.isscalar(max_sigma):
        max_sigma = np.array((max_sigma,) * image.ndim)
    else:
        max_sigma = np.array(max_sigma)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio**i) for i in range(k + 1)])
    gpu_image = cp.asarray(image)

    gpu_detected_blobs = []
    for i in range(k):
        low = cpx_img.gaussian_filter(gpu_image, sigma_list[i])
        high = cpx_img.gaussian_filter(gpu_image, sigma_list[i + 1])
        dog_image = (low - high) * cp.mean(sigma_list[i])

        lm = cp_peak_local_max(
            dog_image,
            threshold_rel=threshold_rel,
            footprint=cp.ones((3,) * (image.ndim)),
        )
        lm = cp.c_[lm, cp.ones(lm.shape[0]) * i]
        gpu_detected_blobs.append(lm)

    gpu_detected_blobs = cp.concatenate(gpu_detected_blobs)
    detected_blobs = cp.asnumpy(gpu_detected_blobs).astype("float32")

    # Catch no peaks
    if detected_blobs.size == 0:
        return np.empty((0, image.ndim * 2))

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[detected_blobs[:, -1].astype(int)]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    detected_blobs = np.hstack([detected_blobs[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]
    detected_blobs = _prune_blobs(detected_blobs, overlap, sigma_dim=sigma_dim)

    return detected_blobs


class BlobDoG:
    def __init__(self, n_dim=2, dim_resolution=1, exclude_border=None):
        if np.isscalar(dim_resolution):
            dim_resolution = (dim_resolution,) * n_dim
        elif dim_resolution is None:
            dim_resolution = (1,) * n_dim
        elif len(dim_resolution) != n_dim:
            raise ValueError(
                f"Length of dim_resolution must be = {n_dim}. Found {dim_resolution}"
            )

        if np.isscalar(exclude_border):
            exclude_border = (exclude_border,) * n_dim
        elif exclude_border is not None and len(exclude_border) != n_dim:
            raise ValueError(
                f"Length of exclude_border must be = {n_dim}. Found {exclude_border}"
            )

        self.D = n_dim
        self.dim_resolution = np.array(dim_resolution)
        self.exclude_border = np.array(exclude_border)
        self.train_step = 0

        # Default parameter settings
        self.min_rad = 7
        self.max_rad = 15
        self.sigma_ratio = 1.4
        self.overlap = 0.8
        self.threshold = 0.05

    def get_parameters(self):
        par = {
            "min_rad": self.min_rad,
            "max_rad": self.max_rad,
            "sigma_ratio": self.sigma_ratio,
            "overlap": self.overlap,
            "threshold": self.threshold,
        }
        return par

    def set_parameters(self, parameters):
        self.min_rad = parameters["min_rad"]
        self.max_rad = parameters["max_rad"]
        self.sigma_ratio = parameters["sigma_ratio"]
        self.overlap = parameters["overlap"]
        self.threshold = parameters["threshold"]

    def predict(self, x, parameters=None, exclude_border="default"):
        """Predicts blob locations from 2 or 3 dimensional image. Blobs are considered white on black.
        If exclude_border has been specified, detected blobs inside the borders of x will be deleted.

        Args:
            x (ndarray): Image array. Can be 2 or 3 dimensional.
            parameters (dict, optional): Dictionary of blob detection parameters.
                Expected keys are: [`min_rad`, `max_rad`, `sigma_ratio`, `overlap`, `threshold`].
                Defaults to None will assign default or previously setted parameters.

        Returns:
            [ndarray]: 2 dimensional array with shape [n_blobs, (n_dim + len(dim_resolution))].
                First `n_dim` columns are the coordinates of each detected blob, last columns are the standard deviations
                which detected the blob. For isotropic images (len(dim_resolution)=1) a single standard deviation is returned,
                for anysotropic images (len(dim_resolution)==n_dim) the standard deviation of each axis is returned.
        """
        if type(x) == bytes:
            x = pickle.loads(x)

        x = x.astype("float32")

        if parameters is None:
            min_sigma = (self.min_rad / self.dim_resolution) / np.sqrt(self.D)
            max_sigma = (self.max_rad / self.dim_resolution) / np.sqrt(self.D)

            with cp.cuda.Device(cp.cuda.runtime.getDeviceCount() - 1):
                with cp.cuda.Stream():
                    centers = blob_dog(
                        x,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        sigma_ratio=self.sigma_ratio,
                        overlap=self.overlap,
                        threshold_rel=self.threshold,
                    )
        else:
            min_sigma = (parameters["min_rad"] / self.dim_resolution) / np.sqrt(self.D)
            max_sigma = (parameters["max_rad"] / self.dim_resolution) / np.sqrt(self.D)

            with cp.cuda.Device(cp.cuda.runtime.getDeviceCount() - 1):
                with cp.cuda.Stream():
                    centers = blob_dog(
                        x,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        sigma_ratio=parameters["sigma_ratio"],
                        overlap=parameters["overlap"],
                        threshold_rel=parameters["threshold"],
                    )

        if isinstance(exclude_border, (list, tuple, int, float)):
            if np.isscalar(exclude_border):
                exclude_border = (exclude_border,) * self.n_dim

            centers = remove_border_points_from_array(centers, x.shape, exclude_border)
        elif exclude_border == "default":
            if self.exclude_border is not None:
                centers = remove_border_points_from_array(
                    centers, x.shape, self.exclude_border
                )
        return centers

    def evaluate(self, y_pred, y_true, max_match_dist, evaluation_type="complete"):
        """Evaluates blob detection prediction with bipartite matching algorithm.
        Note: no border exclusion will be made at this step!

        Args:
            y_pred (ndarray): 2 dimensional array of shape [n_blobs, n_dim] of predicted blobs
            y_true (ndarray): 2 dimensional array of shape [n_blobs, n_dim] of true blobs
            max_match_dist (scalar): maximum distance between predicted and true blobs for a correct prediction.
                It must be in the same scale as dim_resolution.
            evaluation_type (str, optional): One of ["complete", "counts", "f1", "acc", "prec", "rec"].
                "complete" returns every centroid labelled as TP, FP, or FN.
                "counts" returns only the counts of TP, FP, FN plus the total number of predicted blobs
                and the total number of true blobs.
                "f1", "acc", "prec", "rec" returns only the requested metric evaluation.
                Defaults to "complete".

        Returns:
            [pandas.DataFrame or scalar]: if evaluation_type = "complete" returns a pandas.DataFrame with every centroid
                labelled as TP, FP, or FN. If evaluation_type = "counts" returns a pandas.DataFrame with the counts of TP, FP, FN,
                the total number of predicted blobs and the total number of true blobs.
                if evaluation_type is one of ["f1", "acc", "prec", "rec"] returns the scalar of requested metric.
        """
        admitted_types = ["complete", "counts", "f1", "acc", "prec", "rec"]
        assert (
            evaluation_type in admitted_types
        ), f"Wrong evaluation_type provided. {evaluation_type} not in {admitted_types}."

        labeled_centers = bipartite_match(
            y_true, y_pred, max_match_dist, self.dim_resolution
        )

        if evaluation_type == "complete":
            return labeled_centers
        else:
            eval_counts = get_counts_from_bm_eval(labeled_centers)

        if evaluation_type == "counts":
            return eval_counts
        else:
            return metrics(eval_counts)[evaluation_type]

    def predict_and_evaluate(
        self,
        x,
        y,
        max_match_dist,
        evaluation_type="complete",
        parameters=None,
    ):
        """Predicts blob coordinates from x and evaluates the result with the true coordinates in y.
        If exclude_border has been specified, both predicted and true blobs inside the borders of x will be removed.

        Args:
            x (ndarray): array of n_dim dimensions
            y (ndarray): 2 dimensional array with shape [n_blobs, n_dim] of true blobs coordinates
            max_match_dist (scalar): maximum distance between predicted and true blobs for a correct prediction.
                It must be in the same scale as dim_resolution.
            evaluation_type (str, optional): One of ["complete", "counts", "f1", "acc", "prec", "rec"].
                "complete" returns every centroid labelled as TP, FP, or FN.
                "counts" returns only the counts of TP, FP, FN plus the total number of predicted blobs
                and the total number of true blobs.
                "f1", "acc", "prec", "rec" returns only the requested metric evaluation.
                Defaults to "complete".
            parameters (dict, optional): Dictionary of blob detection parameters.
                Expected keys are: [`min_rad`, `max_rad`, `sigma_ratio`, `overlap`, `threshold`].
                Defaults to None will assign default or previously setted parameters.

        Returns:
            [type]: [description]
        """
        if type(x) == bytes:
            x = pickle.loads(x)

        x = x.astype("float32")

        y_pred = self.predict(x, parameters, exclude_border=None)
        # if self.exclude_border is not None:
        #     y_pred = remove_border_points_from_array(y_pred, x.shape, self.exclude_border / 2)
        #     y = remove_border_points_from_array(y, x.shape, self.exclude_border / 2)

        labeled_centers = self.evaluate(y_pred[:, :3], y, max_match_dist=max_match_dist)

        if self.exclude_border is not None:
            labeled_centers = remove_border_points_from_df(
                labeled_centers, ["x", "y", "z"], x.shape, self.exclude_border
            )

        if evaluation_type == "complete":
            return labeled_centers
        else:
            eval_counts = get_counts_from_bm_eval(labeled_centers)
            if evaluation_type == "counts":
                return eval_counts
            else:
                return metrics(eval_counts)[evaluation_type]

    def _objective(
        self,
        parameters,
        X,
        Y,
        max_match_dist,
        checkpoint_dir=None,
        n_cpu=1,
    ):
        if isinstance(X, lmdb.Cursor):
            X = X.iternext(keys=False)

        self.train_step += 1
        parameters["max_rad"] = parameters["min_rad"] + parameters["min_max_rad_diff"]

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "BlobDoG_parameters.json")
            if os.path.isfile(checkpoint_file):
                with open(checkpoint_file, "r") as f:
                    state = json.load(f)

        if n_cpu == 1:
            res = [
                self.predict_and_evaluate(x, y, max_match_dist, "counts", parameters)
                for x, y, in zip(X, Y)
            ]
        else:
            with cf.ThreadPoolExecutor(n_cpu) as pool:
                futures = [
                    pool.submit(
                        self.predict_and_evaluate,
                        x,
                        y,
                        max_match_dist,
                        "counts",
                        parameters,
                    )
                    for x, y in zip(X, Y)
                ]
                res = [future.result() for future in cf.as_completed(futures)]

        res = pd.concat(res)
        f1 = metrics(res)["f1"]

        if checkpoint_dir is not None:
            if self.train_step == 1:
                state = parameters
                state["f1"] = f1
                state["step"] = self.train_step
                with open(checkpoint_file, "w") as f:
                    json.dump(state, f)
            else:
                if f1 > state["f1"]:
                    state = parameters
                    state["f1"] = f1
                    state["step"] = self.train_step
                    with open(checkpoint_file, "w") as f:
                        json.dump(state, f)
        return {"loss": -f1, "status": ho.STATUS_OK}

    def fit(
        self,
        X,
        Y,
        max_match_dist,
        n_iter=60,
        checkpoint_dir=None,
        n_cpu=1,
    ):
        """Method to fit BlobDoG model learning best parameters with Tree Parzen Estimator (TPE).
        Wrapper around hyperopt.fmin().

        Args:
            X (iterable): Iterable of n_dim dimensional images. Length of X must be equal to lenght of Y.
            Y (iterable): Iterable of ndarrays of true blob coordinates. Length of Y must be equal to lenght of X.
            max_match_dist (scalar): Maximum distance between predicted and true blobs for a correct prediction.
                It must be in the same scale as dim_resolution.
            n_iter (int, optional): Number of TPE iterations to perform. Defaults to 60.
            checkpoint_dir (str, optional): Path to the directory where saving the parameters during training. Defaults to None.
            n_cpu (int, optional): Number of parallel threads to use during training. Defaults to 1.
        """
        obj_wrapper = ft.partial(
            self._objective,
            X=X,
            Y=Y,
            max_match_dist=max_match_dist,
            checkpoint_dir=checkpoint_dir,
            n_cpu=n_cpu,
        )

        # Search space
        search_space = {
            "min_rad": ho.hp.uniform("min_rad", 4.0, 15.0),
            "min_max_rad_diff": ho.hp.uniform("min_max_rad_diff", 1.0, 10.0),
            "sigma_ratio": ho.hp.uniform("sigma_ratio", 1.1, 2.0),
            "overlap": ho.hp.uniform("overlap", 0.05, 0.5),
            "threshold": ho.hp.uniform("threshold", 0.05, 0.5),
        }

        best_par = ho.fmin(
            fn=obj_wrapper,
            space=search_space,
            algo=ho.tpe.suggest,
            max_evals=n_iter,
        )

        best_par["max_rad"] = best_par["min_rad"] + best_par["min_max_rad_diff"]
        self.set_parameters(best_par)

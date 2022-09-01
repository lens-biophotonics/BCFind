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
import cucim.skimage.filters as cm_skim_filt
import cucim.skimage.feature as cm_skim_feat

from scipy import spatial

from bcfind.bipartite_match import bipartite_match
from bcfind.utils import metrics, remove_border_points_from_array, remove_border_points_from_df


mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


def _compute_disk_overlap(d, r1, r2):
    """
    Compute fraction of surface overlap between two disks of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.
    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first disk.
    r2 : float
        Radius of the second disk.
    Returns
    -------
    fraction: float
        Fraction of area of the overlap between the two disks.
    """

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = np.arcsin(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = np.arcsin(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 -
            0.5 * np.sqrt(abs(a * b * c * d)))
    return area / (np.pi * (min(r1, r2) ** 2))


def _compute_sphere_overlap(d, r1, r2):
    """
    Compute volume overlap fraction between two spheres of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.
    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first sphere.
    r2 : float
        Radius of the second sphere.
    Returns
    -------
    fraction: float
        Fraction of volume of the overlap between the two spheres.
    Notes
    -----
    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    for more details.
    """
    vol = (np.pi / (12 * d) * (r1 + r2 - d)**2 *
           (d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2))
    return vol / (4./3 * np.pi * min(r1, r2) ** 3)


def _blob_overlap(blob1, blob2, *, sigma_dim=1):
    """Finds the overlapping area fraction between two blobs.
    Returns a float representing fraction of overlapped area. Note that 0.0
    is *always* returned for dimension greater than 3.
    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    sigma_dim : int, optional
        The dimensionality of the sigma value. Can be 1 or the same as the
        dimensionality of the blob space (2 or 3).
    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).
    """
    ndim = len(blob1) - sigma_dim
    if ndim > 3:
        return 0.0
    root_ndim = np.sqrt(ndim)

    # we divide coordinates by sigma * sqrt(ndim) to rescale space to isotropy,
    # giving spheres of radius = 1 or < 1.
    if blob1[-1] == blob2[-1] == 0:
        return 0.0
    elif blob1[-1] > blob2[-1]:
        max_sigma = blob1[-sigma_dim:]
        r1 = 1
        r2 = blob2[-1] / blob1[-1]
    else:
        max_sigma = blob2[-sigma_dim:]
        r2 = 1
        r1 = blob1[-1] / blob2[-1]
    pos1 = blob1[:ndim] / (max_sigma * root_ndim)
    pos2 = blob2[:ndim] / (max_sigma * root_ndim)

    d = np.sqrt(np.sum((pos2 - pos1)**2))
    if d > r1 + r2:  # centers farther than sum of radii, so no overlap
        return 0.0

    # one blob is inside the other
    if d <= abs(r1 - r2):
        return 1.0

    if ndim == 2:
        return _compute_disk_overlap(d, r1, r2)

    else:  # ndim=3 http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        return _compute_sphere_overlap(d, r1, r2)



def _prune_blobs(blobs_array, overlap, *, sigma_dim=1):
    """Eliminates blobs with area overlap.
    
    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 (or 4) values,
        ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
        where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
        and ``sigma`` is the standard deviation of the Gaussian kernel which
        detected the blob.
        This array must not have a dimension of size 0.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    sigma_dim : int, optional
        The number of columns in ``blobs_array`` corresponding to sigmas rather
        than positions.
    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """
    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * np.sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if _blob_overlap(blob1, blob2, sigma_dim=sigma_dim) > overlap:
                # note: this test works even in the anisotropic case because
                # all sigmas increase together.
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.stack([b for b in blobs_array if b[-1] > 0])

def blob_dog(image, min_sigma, max_sigma, sigma_ratio, threshold_rel, overlap):
    r""" Equivalent function to skimage.feature.blob_dog which runs on the GPU, making computation 10 times faster.
    
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
        min_sigma = np.array([min_sigma] * image.ndim)
    else:
        min_sigma = np.array(min_sigma)
    if np.isscalar(max_sigma):
        max_sigma = np.array([max_sigma] * image.ndim)
    else:
        max_sigma = np.array(max_sigma)
    
    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])
    image = cp.asarray(image)

    detected_blobs = []
    for i in range(k):
        low = cm_skim_filt.gaussian(image, sigma_list[i])
        high = cm_skim_filt.gaussian(image, sigma_list[i + 1])
        dog_image = (low - high) * cp.mean(sigma_list[i])
        
        lm = cm_skim_feat.peak_local_max(
            dog_image,
            threshold_rel=threshold_rel,
            footprint=np.ones((3,) * (image.ndim)),
        )
        lm = cp.c_[lm, cp.ones(lm.shape[0]) * i]
        detected_blobs.append(lm)
    
    detected_blobs = cp.concatenate(detected_blobs)
    detected_blobs = cp.asnumpy(detected_blobs).astype("float32")

    # free GPU memory
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # Catch no peaks
    if detected_blobs.size == 0:
        return np.empty((0, 3))

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
            dim_resolution = (dim_resolution, ) * n_dim

        self.D = n_dim
        self.dim_resolution = np.array(dim_resolution)
        self.exclude_border = exclude_border
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

    def predict(self, x, parameters=None, exclude_border='default'):
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
        
        x = x.astype('float32')

        if parameters is None:
            min_sigma = (self.min_rad / self.dim_resolution) / np.sqrt(self.D)
            max_sigma = (self.max_rad / self.dim_resolution) / np.sqrt(self.D)

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
                centers = blob_dog(
                    x,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    sigma_ratio=parameters["sigma_ratio"],
                    overlap=parameters["overlap"],
                    threshold_rel=parameters["threshold"],
                )

        if exclude_border == 'default':
            centers = remove_border_points_from_array(
                centers, x.shape, self.exclude_border
            )
        elif isinstance(exclude_border, (list, int, float)):
            centers = remove_border_points_from_array(
                centers, x.shape, exclude_border
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
            TP = np.sum(labeled_centers.name == "TP")
            FP = np.sum(labeled_centers.name == "FP")
            FN = np.sum(labeled_centers.name == "FN")

            eval_counts = pd.DataFrame([TP, FP, FN, y_pred.shape[0], y_true.shape[0]]).T
            eval_counts.columns = ["TP", "FP", "FN", "tot_pred", "tot_true"]
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
        
        x = x.astype('float32')
        
        y_pred = self.predict(x, parameters, exclude_border=None)
        
        labeled_centers = self.evaluate(y_pred, y, max_match_dist=max_match_dist)

        if self.exclude_border is not None:
            labeled_centers = remove_border_points_from_df(labeled_centers, ['x', 'y', 'z'], x.shape, self.exclude_border)
        
        if evaluation_type == 'complete':
            return labeled_centers
        else:
            TP = np.sum(labeled_centers.name == "TP")
            FP = np.sum(labeled_centers.name == "FP")
            FN = np.sum(labeled_centers.name == "FN")

            eval_counts = pd.DataFrame([TP, FP, FN, y_pred.shape[0], y.shape[0]]).T
            eval_counts.columns = ["TP", "FP", "FN", "tot_pred", "tot_true"]
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
            res = [self.predict_and_evaluate(x, y, max_match_dist, 'counts', parameters) for x, y, in zip(X, Y)]
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
        return {'loss': -f1, 'status': ho.STATUS_OK}

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
            "min_rad": ho.hp.uniform("min_rad", 2.0, 15.0),
            "min_max_rad_diff": ho.hp.uniform("min_max_rad_diff", 1.0, 10.0),
            "sigma_ratio": ho.hp.uniform("sigma_ratio", 1.0, 2.0),
            "overlap": ho.hp.uniform("overlap", 0.0, 1.0),
            "threshold": ho.hp.uniform("threshold", 0.0, 1.0),
        }

        best_par = ho.fmin(
            fn=obj_wrapper, 
            space=search_space, 
            algo=ho.tpe.suggest, 
            max_evals=n_iter,
        )

        best_par["max_rad"] = best_par["min_rad"] + best_par["min_max_rad_diff"]
        self.set_parameters(best_par)

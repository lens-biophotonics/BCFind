import joblib
import pickle
import numpy as np
import pandas as pd
import functools as ft
import scipy.ndimage as sp_img 
import scipy.spatial as sp_spt
import sklearn.neighbors as sk_ngbr
from bcfind.localizers.utils import get_counts_from_bm_eval

from pythreshold.global_th.entropy import kapur_multithreshold

from bcfind.localizers import bipartite_match
from bcfind.utils import metrics, remove_border_points_from_df


class SpatialMeanShift():
    def __init__(self, n_dim=2, dim_resolution=1, exclude_border=None):
        self.n_dim = n_dim
        self.dim_resolution = dim_resolution
        self.exclude_border = exclude_border

        if isinstance(dim_resolution, (int, float)):
            self.dim_resolution = np.array((dim_resolution, ) * n_dim)
        
        if isinstance(exclude_border, (int, float)):
            self.exclude_border = np.array((exclude_border, ) * n_dim)

    def get_seeds(self, x, radius=1, threshold='auto'):
        k_size = 2 * (radius / self.dim_resolution) + 1

        maxima = (x == sp_img.maximum_filter(x, size=k_size.astype(int), mode='constant', cval=0.0))
        
        if threshold == 'auto':
            print('Adopting Kapur thresholding')
            t1, t2 = kapur_multithreshold(x, 2)
            print(f'Removing {np.sum((maxima) & (x<t1))} seeds below {t1}')
            maxima[x < t1] = False
    
        elif isinstance(threshold, (int, float)):
            print('Adopting user-defined thresholding')
            print(f'Removing {np.sum((maxima) & (x<t1))} seeds below {threshold}')
            maxima[x < threshold] = False
        
        seeds = np.array(np.where(maxima)).transpose()

        if self.exclude_border:
            x, y, z = x.shape
            seeds = seeds[(seeds[:, 0] >= self.exclude_border[0] // 2) & (seeds[:, 0] < x - self.exclude_border[0] // 2)]
            seeds = seeds[(seeds[:, 1] >= self.exclude_border[1] // 2) & (seeds[:, 1] < y - self.exclude_border[1] // 2)]
            seeds = seeds[(seeds[:, 2] >= self.exclude_border[2] // 2) & (seeds[:, 2] < z - self.exclude_border[2] // 2)]
        return seeds

    @staticmethod
    def get_coordinates(x_shape):
        return np.indices(x_shape).transpose(1, 2, 3, 0).reshape(-1, 3)
    
    @staticmethod
    def _climb_grad(seed, kdtree, bandwidth, coord, intensities, stop_thresh, max_iter):
        iter = 0
        while True:
            i_nbrs = kdtree.query_ball_point(seed, r=bandwidth)
            points_within = coord[i_nbrs]
            
            # Depending on seeding strategy this condition may occur
            if len(points_within) == 0:
                break

            old_seed = seed  # save the old mean
            seed = np.average(points_within, axis=0, weights=intensities[i_nbrs])
            
            # If converged or at max_iterations, add the cluster
            if np.linalg.norm(seed - old_seed) < stop_thresh or iter == max_iter:
                return tuple(seed), len(points_within), sum(intensities[i_nbrs])
            iter += 1
    
    def _remove_duplicates(self, center_mass_dict, radius):
        print('Removing duplicates')
        
        sorted_by_intensity = sorted(center_mass_dict.items(), key=lambda tup: tup[1], reverse=True)
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        
        nn = sk_ngbr.NearestNeighbors(radius=radius).fit(sorted_centers)

        unique = np.ones(len(sorted_centers), dtype=bool)
        for i, center in enumerate(sorted_centers):
            if unique[i]:
                neighbor_idxs = nn.radius_neighbors([center], return_distance=False)[0]

                unique[neighbor_idxs] = False
                unique[i] = True  # leave the current point as unique

        return sorted_centers[unique]

    def predict(self, x, kernel_radius, peaks_dist=1, threshold='auto', seeds=None, n_jobs=-1, max_iterations=300):
        """
        Spatial Mean shift algorithm to find local peaks in images.

        Implementation taken from scikit-learn with two minor variants:
            - Use (by default) scipy KD-trees, which are faster in our case
            - weigthed version of mean-shift using pixel intensities as
            weights (i.e., we compute centers of mass rather than means)

        Parameters
        ----------

        x : array-like, len(shape) = n_dim
            Input signal.

        kernel_radius : float
            Kernel bandwidth.
        
        peaks_dist : float
            Peaks within this distance will be reduced to one peak. Default to 1.

        seeds : array-like, shape=[n_seeds, n_dim], optional
            Point used as initial kernel locations. If None, a maximum filter is applied and all local maxima are considered seeds.

        Returns
        -------

        peaks_coord : array, shape=[n_clusters, n_dim]
            Coordinates of peaks.

        """
        if seeds is None:
            seeds = self.get_seeds(x, radius=1, threshold=threshold)
        
        print(f'Here I have {len(seeds)} seeds')

        coord = self.get_coordinates(x.shape)
        intensities = x.reshape(-1)

        stop_thresh = 1e-3 * kernel_radius  # when mean has converged
        kdtree = sp_spt.KDTree(coord)

        climb = ft.partial(
            self._climb_grad, 
            kdtree=kdtree, 
            bandwidth=kernel_radius, 
            coord=coord, 
            intensities=intensities, 
            stop_thresh=stop_thresh, 
            max_iterations=max_iterations
            )
        RET = joblib.Parallel(n_jobs=n_jobs, prefer='threads')(joblib.delayed(climb)(m) for m in seeds)
        RET = [a for a in RET if a is not None]  # In case some did not converge...
        
        center_mass_dict = {}
        for ctr, vlm, mss in RET:
            center_mass_dict[ctr] = mss

        peaks = self._remove_duplicates(center_mass_dict, peaks_dist)
        return peaks

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
        kernel_radius,
        peaks_dist,
        max_match_dist,
        seeds=None,
        threshold='auto',
        n_jobs=-1,
        max_iterations=300,
        evaluation_type="complete", 
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
        y_pred = self.predict(x, kernel_radius, peaks_dist, threshold, seeds, n_jobs, max_iterations)
        
        labeled_centers = self.evaluate(y_pred, y, max_match_dist=max_match_dist)

        if self.exclude_border is not None:
            labeled_centers = remove_border_points_from_df(labeled_centers, ['x', 'y', 'z'], x.shape, self.exclude_border)
        
        if evaluation_type == 'complete':
            return labeled_centers
        else:
            eval_counts = get_counts_from_bm_eval(labeled_centers)
            if evaluation_type == "counts":
                return eval_counts
            else:
                return metrics(eval_counts)[evaluation_type]
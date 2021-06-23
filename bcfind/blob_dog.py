import os
import json
import numpy as np
import pandas as pd
import functools as ft
import concurrent.futures as cf
import skimage.feature as sk_feat
import hyperopt as ho

from bcfind.bipartite_match import bipartite_match
from bcfind.utils import metrics


class BlobDoG:
    def __init__(self, n_dim=2, dim_resolution=[1.0, 1.0]):
        self.D = n_dim
        self.dim_resolution = np.array(dim_resolution)
        self.min_rad = 7
        self.max_rad = 15
        self.sigma_ratio = 1.4
        self.overlap = 0.8
        self.threshold = 10

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

    def predict(self, x, parameters=None):
        if parameters is None:
            min_sigma = (self.min_rad / self.dim_resolution) / np.sqrt(self.D)
            max_sigma = (self.max_rad / self.dim_resolution) / np.sqrt(self.D)

            centers = sk_feat.blob_dog(
                x,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                sigma_ratio=self.sigma_ratio,
                overlap=self.overlap,
                threshold=self.threshold,
            )
        else:
            min_sigma = (parameters["min_rad"] / self.dim_resolution) / np.sqrt(self.D)
            max_sigma = (parameters["max_rad"] / self.dim_resolution) / np.sqrt(self.D)

            centers = sk_feat.blob_dog(
                x,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                sigma_ratio=parameters["sigma_ratio"],
                overlap=parameters["overlap"],
                threshold=parameters["threshold"],
            )
        return centers

    def evaluate(self, y_pred, y, max_match_dist, evaluation_type="complete"):
        """6 possible evaluation types are admitted: /'complete/' for single centroid
        labelling, /'counts/' for counts of TP, FP, FN, total predicted and total true,
        /'f1/', /'acc/', /'prec/' or /'rec/' for specific metric evaluation.
        """
        admitted_types = ["complete", "counts", "f1", "acc", "prec", "rec"]
        assert (
            evaluation_type in admitted_types
        ), f"Wrong evaluation_type provided. {evaluation_type} not in {admitted_types}."

        labeled_centers = bipartite_match(
            y, y_pred, max_match_dist, self.dim_resolution
        )

        if evaluation_type == "complete":
            return labeled_centers
        else:
            TP = np.sum(labeled_centers.label == "TP")
            FP = np.sum(labeled_centers.label == "FP")
            FN = np.sum(labeled_centers.label == "FN")

            eval_counts = pd.DataFrame([TP, FP, FN, y_pred.shape[0], y.shape[0]]).T
            eval_counts.columns = ["TP", "FP", "FN", "tot_pred", "tot_true"]
            if evaluation_type == "counts":
                return eval_counts
            else:
                return metrics(eval_counts)[evaluation_type]

    def predict_and_evaluate(
        self, x, y, max_match_dist, evaluation_type="complete", parameters=None
    ):
        x = x.astype(np.float32)
        centers = self.predict(x, parameters)
        evaluation = self.evaluate(
            centers, y, max_match_dist=max_match_dist, evaluation_type=evaluation_type
        )
        return evaluation

    def _objective(
        self, parameters, X, Y, max_match_dist, checkpoint_dir=None, n_cpu=1
    ):
        parameters["max_rad"] = parameters["min_rad"] + parameters["min_max_rad_diff"]

        step = 0
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "parameters.json")
            if os.path.isfile(checkpoint_file):
                with open(checkpoint_file, "r") as f:
                    state = json.load(f)
                step = state["step"] + 1

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
            if step == 0:
                state = parameters
                state["f1"] = f1
                state["step"] = step
                with open(checkpoint_file, "w") as f:
                    json.dump(state, f)
            else:
                if f1 > state["f1"]:
                    state = parameters
                    state["f1"] = f1
                    state["step"] = step
                    with open(checkpoint_file, "w") as f:
                        json.dump(state, f)
        return -f1

    def fit(
        self,
        X,
        Y,
        max_match_dist,
        n_iter=30,
        logs_dir=None,
        checkpoint_dir=None,
        n_cpu=10,
        n_gpu=1,
        verbose=0,
    ):
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
            "sigma_ratio": ho.hp.uniform("sigma_ratio", 1.0, 2.5),
            "overlap": ho.hp.uniform("overlap", 0.01, 1.0),
            "threshold": ho.hp.uniform("threshold", 0.0, 150.0),
        }

        best_par = ho.fmin(
            fn=obj_wrapper, space=search_space, algo=ho.tpe.suggest, max_evals=n_iter
        )

        best_par["max_rad"] = best_par["min_rad"] + best_par["min_max_rad_diff"]
        self.set_parameters(best_par)

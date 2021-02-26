import os
import ray
import json
import numpy as np
import pandas as pd
import concurrent.futures as cf
import skimage.feature as sk_feat
import ray.tune.suggest.hyperopt as ray_ho

from ray import tune

from bipartite_match import bipartite_match
from utils import metrics


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

    def evaluate(self, y_pred, y, max_match_dist=10):
        _, _, TP, FP, FN = bipartite_match(
            y, y_pred, max_match_dist, self.dim_resolution
        )

        pred_eval = pd.DataFrame([TP, FP, FN, y_pred.shape[0], y.shape[0]]).T
        pred_eval.columns = ["TP", "FP", "FN", "tot_pred", "tot_true"]
        return pred_eval

    def predict_and_evaluate(self, x, y, return_centers=True, parameters=None):
        centers = self.predict(x, parameters)
        evaluation = self.evaluate(centers, y)
        if return_centers:
            return centers, evaluation
        if not return_centers:
            return evaluation

    def _objective(self, parameters, X, Y, checkpoint_dir=None, n_cpu=1):
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
                pool.submit(self.predict_and_evaluate, x, y, False, parameters)
                for x, y in zip(X, Y)
            ]
            res = [future.result() for future in cf.as_completed(futures)]

        res = pd.concat(res)

        f1 = metrics(res)["f1"]
        tune.report(score=f1)

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

    def fit(
        self,
        X,
        Y,
        n_iter=30,
        logs_dir=None,
        checkpoint_dir=None,
        n_cpu=10,
        n_gpu=1,
        verbose=0,
    ):
        ray.init()

        init = [
            {
                "min_rad": self.min_rad,
                "min_max_rad_diff": self.max_rad - self.min_rad,
                "sigma_ratio": self.sigma_ratio,
                "overlap": self.overlap,
                "threshold": self.threshold,
            }
        ]

        # Tree-Parzen estimator
        algo = ray_ho.HyperOptSearch(points_to_evaluate=init)
        # Gaussian-process estimator
        # algo = ray_ho.BayesOptSearch(points_to_evaluate=init)
        algo = tune.suggest.ConcurrencyLimiter(algo, max_concurrent=3)

        scheduler = tune.schedulers.AsyncHyperBandScheduler()

        optim = tune.run(
            tune.with_parameters(
                self._objective,
                X=X,
                Y=Y,
                checkpoint_dir=checkpoint_dir,
                n_cpu=n_cpu,
            ),
            num_samples=n_iter,
            search_alg=algo,
            config={
                "min_rad": tune.uniform(5.0, 15.0),
                "min_max_rad_diff": tune.uniform(1.0, 10.0),
                "sigma_ratio": tune.uniform(1.0, 2.0),
                "overlap": tune.uniform(0.1, 1.0),
                "threshold": tune.uniform(0.0, 50.0),
            },
            metric="score",
            mode="max",
            scheduler=scheduler,
            local_dir=logs_dir,
            resources_per_trial={"cpu": 1, "gpu": n_gpu},
            verbose=verbose,
        )

        best_par = optim.get_best_config()
        best_par["max_rad"] = best_par["min_rad"] + best_par["min_max_rad_diff"]
        self.set_parameters(best_par)

        ray.shutdown()

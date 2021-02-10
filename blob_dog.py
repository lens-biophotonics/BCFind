import os
import ray
import json
import numpy as np
import pandas as pd
import skimage.feature as sk_feat
import ray.tune.suggest.hyperopt as ray_ho

from ray import tune

from bipartite_match import bipartite_match


def metrics(df):
    try:
        prec = np.sum(df.TP) / np.sum(df.TP + df.FP)
    except ZeroDivisionError:
        prec = 1
    try:
        rec = np.sum(df.TP) / np.sum(df.TP + df.FN)
    except ZeroDivisionError:
        rec = 1
    try:
        f1 = 2 * prec * rec / (prec + rec)
    except ZeroDivisionError:
        f1 = 0
    return {"prec": prec, "rec": rec, "f1": f1}


class BlobDoG:
    def __init__(self, n_dim=2, dim_resolution=[1.0, 1.0]):
        self.D = n_dim
        self.dim_resolution = np.array(dim_resolution)
        self.min_rad = 7
        self.max_rad = 15
        self.sigma_ratio = 1.6
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
        if not parameters:
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

    def evaluate(self, x, y, parameters=None):
        pred_centers = self.predict(x, parameters)

        _, _, TP, FP, FN = bipartite_match(y, pred_centers, 10, self.dim_resolution)

        pred_eval = pd.DataFrame([TP, FP, FN, pred_centers.shape[0], y.shape[0]]).T
        pred_eval.columns = ["TP", "FP", "FN", "tot_pred", "tot_true"]
        return pred_eval

    def _objective(self, parameters, checkpoint_dir=None, X=None, Y=None):
        parameters["max_rad"] = parameters["min_rad"] + parameters["min_max_rad_diff"]

        step = 0
        if checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
                state = json.load(f.read())
                step = state["step"] + 1

        res = [self.evaluate(x, y, parameters) for x, y in zip(X, Y)]
        res = pd.concat(res)

        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            with open(path, "w") as f:
                f.write(json.dumps({"step": step}))

        f1 = metrics(res)["f1"]
        tune.report(score=f1)

    def fit(self, X, Y, n_iter=50, n_cpu=10, n_gpu=1, outdir=None, verbose=0):
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
        algo = tune.suggest.ConcurrencyLimiter(algo, max_concurrent=4)

        scheduler = tune.schedulers.AsyncHyperBandScheduler()

        optim = tune.run(
            tune.with_parameters(self._objective, X=X, Y=Y),
            num_samples=n_iter,
            search_alg=algo,
            config={
                "min_rad": tune.uniform(5.0, 15.0),
                "min_max_rad_diff": tune.uniform(1.0, 10.0),
                "sigma_ratio": tune.uniform(1.0, 2.0),
                "overlap": tune.uniform(0.5, 1.0),
                "threshold": tune.uniform(0.0, 20.0),
            },
            metric="score",
            mode="max",
            scheduler=scheduler,
            resources_per_trial={"gpu": n_gpu, "cpu": n_cpu},
            local_dir=outdir,
            verbose=verbose,
        )

        best_par = optim.get_best_config()
        best_par["max_rad"] = best_par["min_rad"] + best_par["min_max_rad_diff"]
        self.set_parameters(best_par)

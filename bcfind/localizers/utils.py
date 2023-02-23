import numpy as np
import pandas as pd


def get_counts_from_bm_eval(bm_eval):
    TP = np.sum(bm_eval.name == "TP")
    FP = np.sum(bm_eval.name == "FP")
    FN = np.sum(bm_eval.name == "FN")

    eval_counts = pd.DataFrame(
        [[TP, FP, FN, TP + FP, TP + FN]],
        columns=["TP", "FP", "FN", "tot_pred", "tot_true"],
    )
    return eval_counts

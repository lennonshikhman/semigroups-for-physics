import numpy as np
from scipy.stats import spearmanr


def rel_l2(a, b, axis=-1):
    num = np.linalg.norm(a - b, axis=axis)
    den = np.linalg.norm(b, axis=axis) + 1e-8
    return num / den


def rollout_metrics(pred, true):
    errs = rel_l2(pred, true, axis=2)
    auc = errs.mean(axis=1)
    final = errs[:, -1]
    return errs, auc, final


def spearman_safe(x, y):
    r, _ = spearmanr(x, y)
    return float(0.0 if np.isnan(r) else r)

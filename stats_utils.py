import numpy as np
from scipy.stats import spearmanr, wilcoxon


def bootstrap_mean_ci(x, n_resamples=2000, alpha=0.05, rng=None):
    rng = np.random.default_rng(rng)
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def bootstrap_spearman_ci(x, y, n_resamples=2000, alpha=0.05, rng=None):
    rng = np.random.default_rng(rng)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(x[idx], y[idx])
        vals.append(0.0 if np.isnan(r) else r)
    vals = np.asarray(vals)
    r, _ = spearmanr(x, y)
    return float(0.0 if np.isnan(r) else r), [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]


def paired_bootstrap_diff_ci(a, b, n_resamples=2000, alpha=0.05, rng=None):
    rng = np.random.default_rng(rng)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def wilcoxon_pvalue(a, b):
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return None


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    return float(d.mean() / (d.std() + 1e-8))

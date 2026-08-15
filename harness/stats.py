"""Agreement statistics and confidence intervals.

Everything resamples items, not judgments, because judgments within an item are
not independent: the same pair judged in two orders is one draw from the world,
not two.
"""

import numpy as np

N_BOOT = 10000
SEED = 20260815


def _resample(n_items, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_items, size=(n_boot, n_items))


def ratio(hits, include):
    """Point estimate of sum(hits) / sum(include), or nan if nothing is included.

    `hits` is the numerator contributed by an item and `include` the denominator.
    For a simple accuracy both are 0 or 1. For the human ceiling they are counts
    of agreeing annotator pairs and of annotator pairs. Callers must not put a
    hit on an item they excluded.
    """
    hits = np.asarray(hits, dtype=float)
    include = np.asarray(include, dtype=float)
    denom = include.sum()
    return float("nan") if denom == 0 else float(hits.sum() / denom)


def bootstrap_ci(hits, include=None, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI for a ratio of item level counts.

    Returns (point, lo, hi, n_included).
    """
    hits = np.asarray(hits, dtype=float)
    include = np.ones_like(hits) if include is None else np.asarray(include, dtype=float)
    idx = _resample(len(hits), n_boot, seed)
    num = hits[idx].sum(axis=1)
    den = include[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.where(den > 0, num / den, np.nan)
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return ratio(hits, include), float(lo), float(hi), int(include.sum())


def bootstrap_diff_ci(hits_a, inc_a, hits_b, inc_b, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """CI for (ratio_a - ratio_b) using the same resampled items for both, which
    is the whole point: the two judges saw the same pairs."""
    hits_a, inc_a = np.asarray(hits_a, float), np.asarray(inc_a, float)
    hits_b, inc_b = np.asarray(hits_b, float), np.asarray(inc_b, float)
    idx = _resample(len(hits_a), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        ra = hits_a[idx].sum(axis=1) / inc_a[idx].sum(axis=1)
        rb = hits_b[idx].sum(axis=1) / inc_b[idx].sum(axis=1)
    draws = ra - rb
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return ratio(hits_a, inc_a) - ratio(hits_b, inc_b), float(lo), float(hi)


def cohens_kappa(a, b, labels=None):
    """Cohen's kappa for two raters over nominal labels.

    Written out rather than imported so the hand computed test in tests.py is
    testing this code and not sklearn's.
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("rater sequences must be the same length")
    if not a:
        return float("nan")
    labels = sorted(set(a) | set(b)) if labels is None else list(labels)
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def krippendorff_alpha(reliability_matrix, labels):
    """Nominal alpha over a raters x items matrix of labels, np.nan for missing."""
    import krippendorff

    codes = {label: i for i, label in enumerate(labels)}
    matrix = [[np.nan if v is None else codes[v] for v in row] for row in reliability_matrix]
    return float(
        krippendorff.alpha(reliability_data=np.array(matrix, dtype=float), level_of_measurement="nominal")
    )

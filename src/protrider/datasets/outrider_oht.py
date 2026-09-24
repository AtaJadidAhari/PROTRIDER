from functools import lru_cache

import numpy as np
from scipy.integrate import quad


@lru_cache(maxsize=128)
def _oht_coefficient(beta: float) -> float:
    """Match R's Marchenko-Pastur median search, including its 0.001 tolerance."""
    lower_edge = (1 - np.sqrt(beta)) ** 2
    upper_edge = (1 + np.sqrt(beta)) ** 2

    def density(x):
        if x <= lower_edge or x >= upper_edge:
            return 0.0
        return np.sqrt((upper_edge - x) * (x - lower_edge)) / (2 * np.pi * beta * x)

    low, high = lower_edge, upper_edge
    while high - low > 0.001:
        points = np.linspace(low, high, 10)
        cumulative = np.array([quad(density, lower_edge, x)[0] for x in points])
        if np.any(cumulative < 0.5):
            low = np.max(points[cumulative < 0.5])
        if np.any(cumulative > 0.5):
            high = np.min(points[cumulative > 0.5])
    mp_median = (low + high) / 2
    lambda_beta = np.sqrt(
        2 * (beta + 1) + 8 * beta / (beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1))
    )
    return float(lambda_beta / np.sqrt(mp_median))


def outrider_oht(counts, size_factors, pseudocount=1.0):
    """Return R-compatible OHT diagnostics for a samples-by-genes count matrix.

    R uses double precision and sample standard deviations (n - 1). Constant
    genes yield undefined Z-scores and fail in R; do not silently remove them
    or change the matrix's aspect ratio. PCA initialization uses a different
    transform and must not reuse this decomposition.
    """
    counts = np.asarray(counts, dtype=np.float64)
    size_factors = np.asarray(size_factors, dtype=np.float64).reshape(-1, 1)
    if counts.ndim != 2 or counts.shape[0] < 2 or counts.shape[1] == 0:
        raise ValueError("OUTRIDER OHT requires at least two samples and one gene.")
    if size_factors.shape[0] != counts.shape[0]:
        raise ValueError("OUTRIDER OHT requires one size factor per sample.")
    if not np.isfinite(size_factors).all() or (size_factors <= 0).any():
        raise ValueError("OUTRIDER OHT size factors must be finite and positive.")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("OUTRIDER OHT counts must be finite and non-negative.")

    controlled = counts / size_factors
    log_controlled = np.log2(
        (controlled + pseudocount) / (controlled.mean(axis=0) + pseudocount)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = (log_controlled - log_controlled.mean(axis=0)) / log_controlled.std(axis=0, ddof=1)
    if not np.isfinite(z_scores).all():
        raise ValueError(
            "OUTRIDER OHT gene-wise standardization produced non-finite Z-scores; "
            "check for genes with zero variance after size-factor normalization."
        )

    # As in R, use the orientation with at least as many rows as columns.
    if z_scores.shape[1] > z_scores.shape[0]:
        z_scores = z_scores.T
    singular_values = np.linalg.svd(z_scores, compute_uv=False)
    beta = z_scores.shape[1] / z_scores.shape[0]
    threshold = float(_oht_coefficient(beta) * np.median(singular_values))
    raw_q = int(np.count_nonzero(singular_values > threshold))
    # R permits q=1 and falls back to 2 only when no singular value passes.
    return {
        "singular_values": singular_values,
        "threshold": threshold,
        "q": raw_q if raw_q else 2,
        "raw_q": raw_q,
    }

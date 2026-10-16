"""Scoring metrics.

The headline being scored against is itself noisy, so RMSE has a floor near
``sigma_h`` and the interval scores (CRPS, coverage) are the ones that tell you
whether the posterior is calibrated.  Always report ``n`` next to the number.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _clean(*arrays, positive: tuple[int, ...] = ()):
    """Drop rows where any array is non-finite (and, optionally, non-positive)."""
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a)
    for i in positive:
        ok &= arrays[i] > 0
    return tuple(a[ok] for a in arrays)


def rmse(y, mu) -> float:
    y, mu = _clean(y, mu)
    if not len(y):
        return float('nan')
    return float(np.sqrt(np.mean((y - mu) ** 2)))


def mae(y, mu) -> float:
    y, mu = _clean(y, mu)
    return float(np.mean(np.abs(y - mu))) if len(y) else float('nan')


def bias(y, mu) -> float:
    y, mu = _clean(y, mu)
    return float(np.mean(mu - y)) if len(y) else float('nan')


def crps_normal(y, mu, sd) -> np.ndarray:
    """Closed-form CRPS of a normal forecast against a point observation."""
    y, mu, sd = _clean(y, mu, sd, positive=(2,))
    if not len(y):
        return np.array([])
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def coverage(y, mu, sd, level: float = 0.80) -> float:
    y, mu, sd = _clean(y, mu, sd, positive=(2,))
    if not len(y):
        return float('nan')
    z = norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(y - mu) <= z * sd))


def interval_score(y, lo, hi) -> np.ndarray:
    """Winkler / interval score - a proper scoring rule for interval forecasts."""
    y, lo, hi = _clean(y, lo, hi)
    if not len(y):
        return np.array([])
    width = np.maximum(hi - lo, 1e-9)
    below = 2 / width * (lo - y) * (y < lo)
    above = 2 / width * (y - hi) * (y > hi)
    return width + below + above


def pinball(y, mu, q: float) -> np.ndarray:
    """Quantile (pinball) loss - what a hedger actually cares about."""
    y, mu = _clean(y, mu)
    d = y - mu
    return np.maximum(q * d, (q - 1) * d)

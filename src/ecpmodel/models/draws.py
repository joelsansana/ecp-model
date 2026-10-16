"""Posterior draws of the state, conditional on parameter uncertainty.

Bands from the smoother alone ignore parameter uncertainty.  A cheap fix,
well ahead of full MCMC: draw parameters from their asymptotic normal and run
the simulation smoother for each draw.  For monthly data with ~10 parameters
this is usually good enough.

``res.cov_params()`` is the inverse Hessian of the log-likelihood and is
computed in the *unconstrained* optimiser space; ``res.params`` is in model
space.  Both live in model space after ``cov_params_model()`` below, which
applies the delta method - do not sample ``res.params`` with ``cov_params()``
directly, the two are not on the same scale.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _nearest_pd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Clip a slightly indefinite numerical covariance back onto the cone."""
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    w, v = np.linalg.eigh((cov + cov.T) / 2.0)
    if w.min() < eps:
        warnings.warn(f'cov_params was not positive definite (min eig {w.min():.3g}); '
                      'clipping', stacklevel=2)
        w = np.clip(w, eps, None)
    return (v * w) @ v.T


def cov_params_model(res, robust: bool = False) -> np.ndarray:
    """Parameter covariance in *model* space (the space ``res.params`` lives in)."""
    if robust:
        try:
            return _nearest_pd(res.cov_params(method='robust'))
        except Exception:  # noqa: BLE001
            pass
    try:
        return _nearest_pd(res.cov_params())
    except Exception:  # noqa: BLE001
        return _nearest_pd(res.cov_params(method='robust'))


def param_draws(res, n: int = 300, seed: int = 0) -> np.ndarray:
    """``(n, k)`` draws of the model-space parameter vector."""
    rng = np.random.default_rng(seed)
    mod = res.model
    mu = np.asarray(res.params, dtype=float)
    cov = cov_params_model(res)
    try:
        draw = rng.multivariate_normal(mu, cov, size=n, method='eigh')
    except Exception:  # noqa: BLE001 - fall back to diagonal + jitter
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        draw = mu + rng.normal(size=(n, len(mu))) * sd

    k = len(mu)
    lo = np.full(k, -np.inf)
    hi = np.full(k, np.inf)
    for name, (a, b) in {'gamma': (-0.995, -1e-4), 'theta': (1e-4, 0.999)}.items():
        if name in mod.layout.idx:
            lo = np.maximum(lo, a)
            hi = np.minimum(hi, b)
    for j in mod.layout.log_idx:
        lo[j] = max(lo[j], 1e-6)
    keep = (draw > lo[None, :]).all(axis=1) & (draw < hi[None, :]).all(axis=1)
    if not keep.all():
        if keep.sum() < max(10, n // 10):
            warnings.warn('too few parameter draws fell inside the constraints '
                          '(gamma or theta at a boundary); falling back to the MLE '
                          'point repeated - bands will understate uncertainty',
                          stacklevel=2)
            return np.tile(mu, (n, 1))
        draw = draw[keep]
        reps = int(np.ceil(n / len(draw)))
        draw = np.tile(draw, (reps, 1))[:n]
    return draw


def state_draws(mod, params, seed: int = 0, results=None) -> np.ndarray:
    """``(n_draws, nobs)`` sampled paths of ``p_t`` in k EUR/t.

    Uses the simulation smoother: each path is drawn from the full joint
    posterior of the state given that draw's parameters and the data, so it
    carries both state and parameter uncertainty.
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    paths = []
    for i, par in enumerate(params):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = results if results is not None else mod.smooth(par)
            sim = mod.simulation_smoother(simulator=res)
            sim.simulate(rng=np.random.default_rng(seed + i),
                         initial_state_variates=np.asarray(res.smoothed_state)[:, 0],
                         pretransformed_initial_state_variates=True)
            paths.append(np.array(sim.simulated_state[0], dtype=float))
    return np.array(paths)


def posterior_state_paths(res, n: int | None = None, seed: int = 0,
                          index=None) -> pd.DataFrame:
    """Parameter draws x simulation smoother -> ``(nobs, n)`` DataFrame in EUR/t."""
    from ecpmodel.io import load_cfg

    n = int(n if n is not None else load_cfg('model')['draws']['n'])
    draws = param_draws(res, n=n, seed=seed)
    paths = state_draws(res.model, draws, seed=seed, results=res)
    idx = pd.Index(index) if index is not None else res.model.data.row_labels
    return pd.DataFrame(paths.T * 1000.0, index=idx)

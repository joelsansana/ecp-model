"""Real (deflated) price views of the latent ECP.

Two definitions, and they answer different questions:

``real_price``          purchasing-power view: ECP deflated by the chemicals PPI.
``naphtha_equiv_ratio`` margin view: ECP relative to crude cost per tonne.

Both take *draws*, not point estimates, so the reported bands carry parameter
and state uncertainty.  For forecast months the deflator is unknown: hold it at
its last value (the default) or pass a scenario.  The bands then ignore
deflator uncertainty, which is small next to ethylene's own volatility but not
zero.

Score the backtest in nominal terms - the real price is the nominal one divided
by a known deflator, so errors simply carry over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BBL_PER_T_NAPHTHA = 7.3     # ~7.3 bbl of naphtha per tonne of ethylene


def real_price(ecp_eur_t, ppi_c20, base_month) -> pd.Series:
    """Deflate by the chemicals PPI, rebased to 1.0 at ``base_month``."""
    base = ppi_c20.loc[base_month]
    if not np.isfinite(base) or base == 0:
        raise ValueError(f'PPI base month {base_month!r} is missing or zero')
    out = ecp_eur_t / (ppi_c20 / base)
    return pd.Series(out, index=ecp_eur_t.index if hasattr(ecp_eur_t, 'index') else None)


def naphtha_equiv_ratio(ecp_eur_t, brent_eur_bbl, bbl_per_t: float = BBL_PER_T_NAPHTHA):
    """ECP divided by crude cost per tonne.  An indicator, not a measurement:
    naphtha trades at a variable premium/discount to crude."""
    return ecp_eur_t / (brent_eur_bbl * bbl_per_t)


def deflator(ppi_c20, idx, base_month) -> np.ndarray:
    """PPI deflator aligned to ``idx``, forward-filled (forecast months held flat)."""
    s = pd.Series(ppi_c20).reindex(pd.DatetimeIndex(idx)).ffill()
    base = pd.Series(ppi_c20).loc[base_month]
    if not np.isfinite(base) or base == 0:
        raise ValueError(f'PPI base month {base_month!r} is missing or zero')
    return (s / base).to_numpy(float)


def real_draws(p_paths_k, idx, ppi_c20, base_month) -> np.ndarray:
    """``p_paths_k`` is ``(n_draws, nobs)`` in k EUR/t -> real EUR/t draws."""
    p_paths_k = np.atleast_2d(np.asarray(p_paths_k, dtype=float))
    return p_paths_k * 1000.0 / deflator(ppi_c20, idx, base_month)[None, :]


def real_bands(paths_eur_t, levels=(0.10, 0.5, 0.90)) -> pd.DataFrame:
    """Quantiles of ``(n_draws, nobs)`` EUR/t paths -> tidy band DataFrame."""
    q = np.quantile(np.atleast_2d(paths_eur_t), levels, axis=0)
    lo = int(round(100 * levels[0]))
    mid = int(round(100 * levels[1]))
    hi = int(round(100 * levels[2]))
    return pd.DataFrame({f'lo{lo}': q[0], f'p{int(mid)}': q[1], f'hi{hi}': q[2]})


def margin_ratio(ecp_eur_t, brent_eur_bbl, bbl_per_t: float = BBL_PER_T_NAPHTHA):
    """Implied cracker margin over feedstock: ECP minus crude cost per tonne."""
    return ecp_eur_t - brent_eur_bbl * bbl_per_t

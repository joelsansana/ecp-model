"""Ethylene contract price: error-correction state-space model.

Notation (all prices inside the model are in **k EUR/t** for conditioning):

    p_t = p_{t-1} + gamma*(p_{t-1} - c_{t-1}) + theta*(c_t - c_{t-1}) + eta_t
    eta_t ~ N(0, sigma_eta^2 * (1 + kappa * r_t)),      -1 < gamma < 0, 0 <= theta <= 1

``c_t = beta' x_t`` is the cost anchor: Brent, gas and EUA, each through a
distributed lag, in EUR.  ``beta`` absorbs the naphtha/crude ratio, cracker
yields and co-product credits - do not try to read it as a physical pass-through.

Observations (any series may be missing in any month; the Kalman filter skips
missing entries natively):

    (1) headline:  y1_t = p_t                       + e1_t,  Var = s_h^2 * w(rel) * repl
    (2) Comext UV: y2_t = a_c + p_{t-1}             + e2_t,  Var = s_c^2
    (3) PPI (std): y3_t = a_p + b_p*mean(p_{t-3:t})+ e3_t,  Var = s_p^2

Only (1) pins the level of ``p_t``.  Without enough headlines the model
degenerates into "cost anchor plus smoothed trade data", which is useful but is
*not* an estimate of the contract price.  Aim for 25-30+ headline observations.

Everything is linear-Gaussian given the parameter vector, so the exact
likelihood comes from the Kalman filter and the parameters are fitted by
maximum likelihood.  Optimisation happens in unconstrained space
(``transform_params``), which is what keeps gamma, theta and every variance
valid; ``start_params`` given to ``fit`` is in *model* space, as statsmodels
expects.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.mlemodel import MLEModel

from ecpmodel.features.build import REL_MULT, build_panel
from ecpmodel.io import PROCESSED, load_cfg, save_processed

OBS_LOADING = {
    # obs name -> (state index it loads onto, intercept parameter)
    'headline': (0, None),      # p_t
    'comext': (1, 'a_c'),       # p_{t-1}
    'ppi': None,                # equal weights on the three most recent states
}
PPI_WINDOW = 3


@dataclass
class Layout:
    """Parameter vector layout, so the model adapts to which observations exist."""

    obs: tuple[str, ...]
    k_states: int
    names: list[str]
    log_idx: list[int] = field(default_factory=list)
    gamma_idx: int = -1
    theta_idx: int = -1
    idx: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.names)


def make_layout(obs: tuple[str, ...], k_beta: int, comext_lag: int = 1) -> Layout:
    obs = tuple(o for o in obs if o in OBS_LOADING)
    # the state must be deep enough for the most lagged observation
    depth = 0
    if 'comext' in obs:
        depth = max(depth, comext_lag)
    if 'ppi' in obs:
        depth = max(depth, PPI_WINDOW - 1)
    k_states = 1 + depth

    names = ['s_eta', 's_h']
    if 'comext' in obs:
        names += ['s_c']
    if 'ppi' in obs:
        names += ['s_p']
    names += ['gamma', 'theta', 'kappa']
    if 'comext' in obs:
        names += ['a_c']
    if 'ppi' in obs:
        names += ['a_p', 'b_p']
    names += [f'beta{i}' for i in range(k_beta)]

    idx = {n: i for i, n in enumerate(names)}
    log_idx = [idx[n] for n in ('s_eta', 's_h') if n in idx]
    for n in ('s_c', 's_p', 'kappa'):
        if n in idx:
            log_idx.append(idx[n])
    return Layout(obs=obs, k_states=k_states, names=names, log_idx=sorted(log_idx),
                  gamma_idx=idx['gamma'], theta_idx=idx['theta'], idx=idx)


class ECPModel(MLEModel):
    """Kalman-filter ECM for a single latent price with noisy proxies."""

    def __init__(self, endog, X, regime, hvar, obs=None, comext_lag: int = 1,
                 initialization: str = 'approximate_diffuse'):
        endog = pd.DataFrame(endog)
        obs = tuple(obs) if obs is not None else tuple(endog.columns)
        obs = tuple(o for o in obs if o in endog.columns)

        self.X = np.asarray(X, dtype=float)
        self.regime = np.asarray(regime, dtype=float)
        self.hvar = np.asarray(hvar, dtype=float)
        self.obs_names = list(obs)
        self.layout = make_layout(obs, self.X.shape[1], comext_lag=comext_lag)

        super().__init__(endog[list(obs)], k_states=self.layout.k_states, k_posdef=1,
                         initialization=initialization)
        self['selection'] = np.array([[1.0]] + [[0.0]] * (self.layout.k_states - 1))
        self['state_intercept'] = np.zeros((self.layout.k_states, self.endog.shape[0]))
        self['state_cov'] = np.zeros((1, 1, self.endog.shape[0]))
        self['obs_cov'] = np.zeros((len(obs), len(obs), self.endog.shape[0]))

    # -- parameter plumbing --------------------------------------------------
    @property
    def param_names(self) -> list[str]:
        return self.layout.names

    def _logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def transform_params(self, u):
        p = np.array(u, dtype=float).ravel()
        li = self.layout.log_idx
        if li:
            p[li] = np.exp(p[li])
        p[self.layout.gamma_idx] = -self._logistic(p[self.layout.gamma_idx])
        p[self.layout.theta_idx] = self._logistic(p[self.layout.theta_idx])
        return p

    def untransform_params(self, p):
        p = np.asarray(p, dtype=float).ravel()
        u = np.array(p, dtype=float)
        li = self.layout.log_idx
        if li:
            u[li] = np.log(np.clip(u[li], 1e-12, None))
        g, th = p[self.layout.gamma_idx], p[self.layout.theta_idx]
        u[self.layout.gamma_idx] = np.log(-g / (1.0 + g))
        u[self.layout.theta_idx] = np.log(th / (1.0 - th))
        return u

    # -- system matrices -----------------------------------------------------
    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        L = self.layout
        p = np.asarray(params)
        # complex-step covariance evaluation passes complex parameters; keep the
        # representation dtype in step or the imaginary parts are silently lost
        dtype = np.complex128 if np.iscomplexobj(p) else np.float64
        k, kE = L.k_states, len(self.obs_names)
        nobs = self.endog.shape[0]
        if self['state_intercept'].dtype != dtype:
            self['state_intercept'] = np.zeros((k, nobs), dtype)
            self['state_cov'] = np.zeros((1, 1, nobs), dtype)
            self['obs_cov'] = np.zeros((kE, kE, nobs), dtype)

        beta = p[[L.idx[f'beta{i}'] for i in range(self.X.shape[1])]]
        c = self.X @ beta
        c_lag = np.concatenate([c[:1], c[:-1]])
        d = -p[L.gamma_idx] * c_lag + p[L.theta_idx] * (c - c_lag)

        design = np.zeros((kE, k), dtype)
        obs_int = np.zeros((kE, 1), dtype)
        for j, name in enumerate(self.obs_names):
            if name == 'ppi':
                design[j, :PPI_WINDOW] = p[L.idx['b_p']] / PPI_WINDOW
                obs_int[j, 0] = p[L.idx['a_p']]
            else:
                design[j, OBS_LOADING[name][0]] = 1.0
                par = OBS_LOADING[name][1]
                if par is not None:
                    obs_int[j, 0] = p[L.idx[par]]

        trans = np.zeros((k, k), dtype)
        trans[0, 0] = 1.0 + p[L.gamma_idx]
        for i in range(1, k):
            trans[i, i - 1] = 1.0

        self['design'] = design
        self['obs_intercept'] = obs_int
        self['transition'] = trans
        self['state_intercept'][0, :] = d
        self['state_cov'][0, 0, :] = (p[L.idx['s_eta']] ** 2
                                      * (1.0 + p[L.idx['kappa']] * self.regime))
        for j, name in enumerate(self.obs_names):
            if name == 'headline':
                self['obs_cov'][j, j, :] = p[L.idx['s_h']] ** 2 * self.hvar
            elif name == 'comext':
                self['obs_cov'][j, j, :] = p[L.idx['s_c']] ** 2
            elif name == 'ppi':
                self['obs_cov'][j, j, :] = p[L.idx['s_p']] ** 2
        return params


# ---------------------------------------------------------------------------
# fitting and outputs
# ---------------------------------------------------------------------------
def make_model(panel: dict, obs=None, comext_lag: int = 1) -> ECPModel:
    endog = panel['endog']
    if obs is None:
        obs = load_cfg('model').get('observations', endog.columns)
    obs = [o for o in obs if o in endog.columns]
    for name in obs:
        if not endog[name].notna().any():
            warnings.warn(f'observation {name!r} has no data in this panel - its '
                          'noise parameter is unidentified', stacklevel=2)
    return ECPModel(endog, panel['X'], panel['regime'], panel['hvar'], obs=obs,
                    comext_lag=comext_lag)


def start_vector(panel: dict, mod: ECPModel | None = None) -> np.ndarray:
    """OLS on the visible headlines for beta; fixed defaults for the rest."""
    cfg = load_cfg('model')['start']
    if mod is None:
        k_beta = panel['X'].shape[1]
        obs = tuple(panel['endog'].columns)
        lay = make_layout(obs, k_beta)
        names = lay.names
    else:
        names = mod.param_names

    h = panel['endog']['headline'].dropna()
    if len(h) >= panel['X'].shape[1] + 2:
        Xh = panel['X'].loc[h.index]
        beta0 = np.linalg.lstsq(Xh.to_numpy(float), h.to_numpy(float), rcond=None)[0]
    else:  # too few headlines to fit beta: start at "constant = observed level"
        beta0 = np.zeros(panel['X'].shape[1])
        beta0[0] = float(h.mean()) if len(h) else 1.0

    out = np.empty(len(names))
    for i, n in enumerate(names):
        if n.startswith('beta'):
            out[i] = beta0[int(n[4:])]
        else:
            out[i] = float(cfg.get(n, 0.0))
    return out


def _jitter(s0: np.ndarray, lay: Layout, rng, i: int) -> np.ndarray:
    s = s0.copy()
    if i == 0:
        return s
    li = lay.log_idx
    if li:
        s[li] = np.clip(s[li] * np.exp(rng.normal(0, 0.6, len(li))), 1e-4, None)
    s[lay.gamma_idx] = float(np.clip(s[lay.gamma_idx] + rng.normal(0, 0.15), -0.9, -0.02))
    s[lay.theta_idx] = float(np.clip(s[lay.theta_idx] + rng.normal(0, 0.15), 0.02, 0.95))
    return s


def fit(panel: dict, n_starts: int | None = None, seed: int = 0, obs=None,
        comext_lag: int = 1, maxiter: int = 3000, quiet: bool = True):
    """Multi-start maximum likelihood.  Returns ``(model, results)``."""
    cfg = load_cfg('model')
    n_starts = int(n_starts if n_starts is not None else cfg['multi_starts'])
    mod = make_model(panel, obs=obs, comext_lag=comext_lag)
    s0 = start_vector(panel, mod)
    lay = mod.layout
    rng = np.random.default_rng(seed)

    best, best_llf, best_i = None, -np.inf, -1
    for i in range(max(n_starts, 1)):
        s = _jitter(s0, lay, rng, i)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = mod.fit(start_params=s, method='lbfgs', maxiter=maxiter, disp=False)
        except Exception:  # noqa: BLE001 - a failed start is normal, just skip it
            continue
        llf = float(res.llf)
        if np.isfinite(llf) and llf > best_llf:
            best, best_llf, best_i = res, llf, i

    if best is None:
        raise RuntimeError('every optimisation start failed; check the panel for NaNs')
    if not quiet:
        print(f'[model] best start #{best_i}, llf={best_llf:.2f}')
    return mod, best


def summarize(res, panel: dict, probs=(0.10, 0.05), burn_in: int | None = None):
    """Latent ECP with bands, the cost anchor and the implied margin."""
    idx = panel['endog'].index
    burn = int(burn_in if burn_in is not None else res.model.layout.k_states - 1)

    mu = np.asarray(res.smoothed_state)[0] * 1000.0
    sd = np.sqrt(np.asarray(res.smoothed_state_cov)[0, 0]) * 1000.0
    out = pd.DataFrame({'ecp_mean': mu, 'ecp_sd': sd}, index=idx)
    for a in probs:
        z = norm.ppf(1 - a)
        tag = int(round(100 - 2 * a * 100))
        out[f'lo{tag}'] = mu - z * sd
        out[f'hi{tag}'] = mu + z * sd
    out['burn_in'] = False
    out.iloc[:burn, out.columns.get_loc('burn_in')] = True

    L = res.model.layout
    beta = np.asarray(res.params)[[L.idx[f'beta{i}'] for i in range(panel['X'].shape[1])]]
    out['cost_anchor'] = panel['X'].to_numpy(float) @ beta * 1000.0
    out['margin'] = out['ecp_mean'] - out['cost_anchor']
    ppi_raw = panel.get('ppi_raw')
    if isinstance(ppi_raw, pd.Series):
        out['ppi'] = ppi_raw.reindex(idx)
    elif ppi_raw is not None:
        out['ppi'] = pd.Series(np.asarray(ppi_raw, dtype=float), index=idx)
    return out


def headline_observation(res, rel: int, repl: int = 1) -> float:
    """Predictive sd for scoring a *future* headline against a smoothed state."""
    p = np.asarray(res.params)
    return float(p[res.model.layout.idx['s_h']] * 1000.0
                 * np.sqrt(REL_MULT.get(int(rel), 2.0) * repl))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Fit the ECP state-space model.')
    ap.add_argument('--horizon', type=int, default=None, help='forecast months to append')
    ap.add_argument('--obs', nargs='+', default=None, choices=['headline', 'comext', 'ppi'],
                    help='observation equations to use (default: configs/model.yaml)')
    ap.add_argument('--no-comext', action='store_true')
    ap.add_argument('--no-ppi', action='store_true')
    ap.add_argument('--starts', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    mcfg = load_cfg('model')
    horizon = int(args.horizon if args.horizon is not None else mcfg['forecast_horizon'])
    obs = list(args.obs if args.obs is not None else mcfg['observations'])
    obs = [o for o in obs
           if not (args.no_comext and o == 'comext') and not (args.no_ppi and o == 'ppi')]

    panel = build_panel(horizon=horizon)
    mod, res = fit(panel, n_starts=args.starts, seed=args.seed, obs=obs, quiet=False)
    print(res.summary())

    out = summarize(res, panel)
    save_processed(out, 'ecp_latent')
    save_processed(pd.DataFrame(np.asarray(res.params).reshape(1, -1),
                                columns=mod.param_names,
                                index=[pd.Timestamp.now().normalize()]), 'ecp_params')
    print(f'[model] wrote {PROCESSED / "ecp_latent.parquet"} ({len(out)} months)')
    tail = out[~out['burn_in']].tail(6)
    print(tail[['ecp_mean', 'lo80', 'hi80', 'cost_anchor', 'margin']].round(1).to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

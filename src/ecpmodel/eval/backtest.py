"""Rolling-origin backtest with vintage discipline.

For each month-end ``t0`` the panel is rebuilt *as of* ``t0`` (publication lags
enforced), the model is refitted, and the nowcast (``h=0``) plus forecasts
``h=1..H`` are scored against headlines that were **not** in the information
set (``published_on > t0``).

Benchmarks:

* ``naive``  - last visible headline;
* ``ols``    - cost-plus OLS: regress visible headlines on the cost block and
  carry the latest costs forward;
* ``ssm_core`` - this model with Comext and PPI switched off, i.e. headlines and
  costs only.  The state-space model only earns its complexity if it beats the
  cheap benchmarks at h=0 and h=1 *and* its intervals are calibrated.

Honest limitation: you are scoring against noisy headlines, so RMSE has a floor
near ``sigma_h``.  Better scoring comes from a second independent source: fit on
source A, score on source B.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from ecpmodel.eval import metrics as M
from ecpmodel.features.build import build_panel
from ecpmodel.io import PROCESSED, load_cfg, save_processed
from ecpmodel.models.ecm_ssm import fit, headline_observation

BENCHMARKS = ['naive', 'ols', 'ssm_core']


def _h_month(t0: pd.Timestamp, m: pd.Timestamp) -> int:
    return (m.year - t0.year) * 12 + (m.month - t0.month)


def _ols_predict(panel: dict) -> pd.Series:
    """Cost-plus OLS on visible headlines, extrapolated along the cost path."""
    X = panel['X']
    vh = panel['endog']['headline'].dropna() * 1000.0
    if len(vh) < X.shape[1] + 2:
        return pd.Series(np.nan, index=X.index)
    beta = np.linalg.lstsq(X.loc[vh.index].to_numpy(float),
                           vh.to_numpy(float), rcond=None)[0]
    return pd.Series(X.to_numpy(float) @ beta, index=X.index)


def _fit_core(panel: dict, obs_default) -> tuple:
    core_obs = [o for o in obs_default if o == 'headline'] or ['headline']
    try:
        return fit(panel, n_starts=2, obs=core_obs, quiet=True)
    except Exception:  # noqa: BLE001
        return None, None


def run(origins, horizon: int = 3, obs=None, starts: int = 3, core: bool = True,
        verbose: bool = True) -> pd.DataFrame:
    from ecpmodel.ingest.headlines import load as load_headlines

    heads = load_headlines()
    mcfg = load_cfg('model')
    obs = list(obs if obs is not None else mcfg['observations'])
    rows: list[dict] = []

    for t0 in pd.DatetimeIndex(origins):
        t0 = pd.Timestamp(t0)
        t_start = time.time()
        panel = build_panel(t0=t0, horizon=horizon, headlines=heads)
        try:
            mod, res = fit(panel, n_starts=starts, obs=obs)
        except Exception as exc:  # noqa: BLE001
            print(f'[backtest] {t0.date()} fit failed: {exc}', flush=True)
            continue

        idx = panel['endog'].index
        mu = pd.Series(np.asarray(res.smoothed_state)[0] * 1000.0, index=idx)
        var_p = pd.Series(np.asarray(res.smoothed_state_cov)[0, 0] * 1e6, index=idx)

        truth = heads[(heads['available_on'] > t0)
                      & (heads['month'] >= t0.to_period('M').to_timestamp())
                      & (heads['month'] <= t0 + pd.DateOffset(months=horizon))]
        vis = heads[heads['available_on'] <= t0].sort_values('month')
        naive = float(vis['y'].iloc[-1]) if len(vis) else np.nan
        ols = _ols_predict(panel)

        res_c = None
        core_mu = pd.Series(np.nan, index=idx)
        core_var = pd.Series(np.nan, index=idx)
        if core:
            _, res_c = _fit_core(panel, obs)
            if res_c is not None:
                core_mu = pd.Series(np.asarray(res_c.smoothed_state)[0] * 1000.0, index=idx)
                core_var = pd.Series(np.asarray(res_c.smoothed_state_cov)[0, 0] * 1e6,
                                     index=idx)

        for _, r in truth.iterrows():
            m = pd.Timestamp(r['month'])
            if m not in mu.index:
                continue
            # the headline is p_t + noise: the predictive sd must carry both
            # the state uncertainty and the observation noise for this shape
            obs_sd = headline_observation(res, rel=int(r['rel']), repl=int(r['repl']))
            sd = float(np.sqrt(var_p[m] + obs_sd ** 2))
            row = {
                't0': t0, 'month': m, 'h': _h_month(t0, m), 'y': float(r['y']),
                'rel': int(r['rel']), 'repl': int(r['repl']),
                'ssm_mu': float(mu[m]), 'ssm_sd': sd,
                'naive': naive, 'ols': float(ols.get(m, np.nan)),
            }
            if res_c is not None:
                core_obs_sd = headline_observation(res_c, rel=int(r['rel']),
                                                  repl=int(r['repl']))
                row['core_sd'] = float(np.sqrt(core_var[m] + core_obs_sd ** 2))
                row['ssm_core_mu'] = float(core_mu[m])
            rows.append(row)

        if verbose:
            print(f'[backtest] {t0.date()} ({len(truth)} targets, {time.time() - t_start:.1f}s)',
                  flush=True)

    return pd.DataFrame(rows)


def report(df: pd.DataFrame, stress_window=None) -> pd.DataFrame:
    """Metrics by forecast horizon, plus a 2021-22 stress split."""
    if not len(df):
        return pd.DataFrame()
    stress_window = stress_window or load_cfg('model')['backtest']['stress_window']
    s0, s1 = pd.Timestamp(stress_window[0]), pd.Timestamp(stress_window[1])

    def agg(x: pd.DataFrame) -> pd.Series:
        out = {
            'n': len(x),
            'rmse_ssm': M.rmse(x['y'], x['ssm_mu']),
            'crps_ssm': float(np.mean(M.crps_normal(x['y'], x['ssm_mu'], x['ssm_sd']))),
            'cov80': M.coverage(x['y'], x['ssm_mu'], x['ssm_sd'], 0.80),
            'bias_ssm': M.bias(x['y'], x['ssm_mu']),
        }
        for b in BENCHMARKS:
            col = f'{b}_mu' if f'{b}_mu' in x.columns else b
            if col in x.columns:
                out[f'rmse_{b}'] = M.rmse(x['y'], x[col])
        if 'core_sd' in x and x['core_sd'].notna().any():
            out['crps_core'] = float(np.mean(M.crps_normal(x['y'], x['ssm_core_mu'],
                                                           x['core_sd'])))
        return pd.Series(out)

    labels, parts = [], []
    for h, g in df.groupby('h'):
        labels.append(f'h={h}')
        parts.append(agg(g))
    labels.append('all')
    parts.append(agg(df))

    calm = df[~df['month'].between(s0, s1)]
    stress = df[df['month'].between(s0, s1)]
    for name, part in (('stress', stress), ('calm', calm)):
        if len(part):
            labels.append(name)
            parts.append(agg(part))

    return pd.DataFrame(dict(zip(labels, parts, strict=True))).T.rename_axis('slice')


def main(argv: list[str] | None = None) -> int:
    bcfg = load_cfg('model')['backtest']
    ap = argparse.ArgumentParser(description='Rolling-origin backtest.')
    ap.add_argument('--start', default=bcfg['start'])
    ap.add_argument('--end', default=bcfg['end'])
    ap.add_argument('--step', type=int, default=bcfg['step_months'])
    ap.add_argument('--horizon', type=int, default=load_cfg('model')['forecast_horizon'])
    ap.add_argument('--starts', type=int, default=3)
    ap.add_argument('--no-core', action='store_true', help='skip the core benchmark')
    ap.add_argument('--out', default='backtest_raw')
    args = ap.parse_args(argv)

    origins = pd.date_range(args.start, args.end, freq='ME')[::args.step]
    df = run(origins, horizon=args.horizon, starts=args.starts, core=not args.no_core)
    if not len(df):
        print('[backtest] nothing to score - check the headline file and panel')
        return 1
    save_processed(df, args.out)
    rep = report(df)
    print(rep.round(3).to_string())
    save_processed(rep.reset_index(), f'{args.out}_report')
    print(f'[backtest] wrote {PROCESSED / (args.out + ".parquet")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

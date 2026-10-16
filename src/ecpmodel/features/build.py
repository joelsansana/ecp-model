"""As-of panel construction: units, distributed lags, vintage discipline.

Everything the model sees is assembled here, once, so that the fit and the
backtest cannot drift apart.  Two rules:

* **Availability is applied here, never in ingestion.**  A series is filtered
  by ``configs/data.yaml:availability_lag_days`` against the forecast date.
* **Every standardised column is standardised on the visible sample only.**
  Standardising on the full sample is a silent look-ahead and it is the easiest
  way to overstate backtest accuracy.

Caveat to document in any report built on this: the as-of filter models
*publication lag*, not *revision*.  Comext values pulled today are final unless
you snapshot monthly (``io.save_raw`` writes dated files, so just keep running
``make data`` on a schedule).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ecpmodel.io import INTERIM, ROOT, load_cfg, manual_path

MMBTU_TO_MWH = 0.29307
REL_MULT = {1: 1.0, 2: 2.0, 3: 4.0}

# interim file -> availability_lag_days key.  The PPI is discovered separately
# (`_discover_ppi`) because its stem depends on the NACE/country that worked.
SERIES_FILES = {
    'fx': 'eurusd',
    'brent': 'brent_usd_bbl',
    'gas': 'gas_usd_mmbtu',
    'eua': 'eua_eur_t',
    'uv': 'uv_ethylene',
    'uvmed': 'uvmed_ethylene',
    'uv_propylene': 'uv_propylene',
    'uv_butadiene': 'uv_butadiene',
    'uv_naphtha': 'uv_naphtha',
}


def _ld(name: str) -> pd.Series:
    return pd.read_parquet(INTERIM / f'{name}.parquet').iloc[:, 0]


def _discover_ppi() -> str | None:
    """Interim file stem of the chemicals PPI, whatever NACE/country it used."""
    hits = sorted(p.stem for p in INTERIM.glob('ppi_*.parquet'))
    return hits[0] if hits else None


def as_of(series: pd.Series, lag_days: int, t0: pd.Timestamp) -> pd.Series:
    """Rows that would have been visible at ``t0``.

    The observation for month ``m`` is assumed to appear at ``month_end(m) +
    lag_days``.  Month-end, not month-start: a monthly average is a complete
    month of data.
    """
    if not len(series):
        return series
    month_end = pd.DatetimeIndex(series.index) + pd.offsets.MonthEnd(0)
    visible = month_end + pd.Timedelta(days=int(lag_days)) <= pd.Timestamp(t0)
    return series[visible]


def headlines_as_of(h: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    if not len(h):
        return h
    return h[h['available_on'] <= pd.Timestamp(t0)]


def dist_lag(s: pd.Series, w) -> pd.Series:
    """Distributed lag with fixed weights (must sum to 1 to keep units)."""
    out = sum(float(wk) * s.shift(k) for k, wk in enumerate(w))
    return out


def load_regimes(idx: pd.DatetimeIndex, path: str | None = None) -> pd.Series:
    """0/1 disruption indicator; an absent file means "no known disruptions"."""
    r = pd.Series(0.0, index=idx, name='regime')
    p = path or str(manual_path('regimes'))
    try:
        d = pd.read_csv(p, parse_dates=['start', 'end'], comment='#')
    except FileNotFoundError:
        fallback = ROOT / 'data/manual/regimes.example.csv'
        if fallback.exists():
            d = pd.read_csv(fallback, parse_dates=['start', 'end'], comment='#')
        else:
            return r
    for _, row in d.iterrows():
        r[(r.index >= row['start']) & (r.index <= row['end'])] = 1.0
    return r


def _standardise(s: pd.Series) -> pd.Series:
    """z-score using the observed sample only (never the full series)."""
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index, name=s.name)
    return (s - mu) / sd


def build_panel(t0=None, horizon: int = 0, lag_w=(0.3, 0.5, 0.2), headlines=None,
                ppi_name: str | None = None):
    """Model-ready panel as it would have looked at ``t0``.

    Returns a dict with ``endog, X, regime, hvar, cost, ppi_raw, horizon`` and
    ``cost_cols`` (the actual regressor columns that survived).
    """
    cfg = load_cfg('data')
    lagd = cfg['availability_lag_days']

    raw: dict[str, pd.Series] = {}
    for key, fname in SERIES_FILES.items():
        try:
            raw[key] = _ld(fname)
        except FileNotFoundError:
            continue

    ppi_file = ppi_name or _discover_ppi()
    if ppi_file:
        try:
            raw['ppi'] = _ld(ppi_file)
        except FileNotFoundError:
            pass

    if t0 is not None:
        t0 = pd.Timestamp(t0)
        for k, s in raw.items():
            raw[k] = as_of(s, lagd.get(k, 35), t0)

    start = pd.Timestamp(cfg['start']).to_period('M').to_timestamp()
    end = (t0.to_period('M').to_timestamp() if t0 is not None
           else max(s.index.max() for s in raw.values())) + pd.DateOffset(months=horizon)
    idx = pd.date_range(start, end, freq='MS')
    df = pd.DataFrame({k: s.reindex(idx) for k, s in raw.items()}, index=idx)

    # --- units.  USD -> EUR via USD-per-EUR; USD/MMBtu -> EUR/MWh -------------
    if 'fx' in df:
        fx = df['fx']
    else:
        fx = 1.0
        warnings.warn('no EUR/USD series: cost drivers stay in USD, so the anchor '
                      'and the level of p_t are not comparable', stacklevel=2)

    cost = pd.DataFrame(index=idx)
    if 'brent' in df:
        cost['brent_eur'] = df['brent'] / fx
    if 'gas' in df:
        cost['gas_eur_mwh'] = df['gas'] / fx / MMBTU_TO_MWH
    if 'eua' in df:
        cost['eua'] = df['eua']

    # The not-yet-published tail is forward-filled: documented, and the reason to
    # prefer a delayed daily feed if you have one.
    cost = cost.ffill()
    cost_l = cost.apply(lambda c: dist_lag(c, lag_w))

    # a cost series that is entirely absent (e.g. no manual EUA file) is dropped
    # from the regressor block rather than faked with zeros - beta re-indexes
    # itself to whatever survives
    use = [c for c in cost_l.columns if cost_l[c].notna().any()]
    if not use:
        warnings.warn('no cost series survived the availability filter - the anchor '
                      'is unidentified and beta cannot be estimated', stacklevel=2)
    cost_l = cost_l[use]

    X = pd.concat([pd.Series(1.0, index=idx, name='const'), cost_l / 100.0], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)

    # --- observations --------------------------------------------------------
    if headlines is None:
        from ecpmodel.ingest.headlines import load as load_headlines

        headlines = load_headlines()
    h = headlines if t0 is None else headlines_as_of(headlines, t0)
    h = h.assign(mult=h['rel'].map(REL_MULT) * h['repl'])
    hm = h.groupby('month').agg(y=('y', 'mean'), mult=('mult', 'min'))

    endog = pd.DataFrame(index=idx, columns=['headline', 'comext', 'ppi'], dtype=float)
    endog['headline'] = (hm['y'] / 1000.0).reindex(idx)
    if 'uv' in df:
        endog['comext'] = df['uv'] / 1000.0
    ppi = df['ppi'] if 'ppi' in df else pd.Series(np.nan, index=idx, name='ppi')
    endog['ppi'] = _standardise(ppi)     # visible-sample mean/std only

    regime = load_regimes(idx)
    hvar = hm['mult'].reindex(idx).fillna(1.0)

    keep = X.dropna().index                       # drops the distributed-lag warm-up
    panel = {
        'endog': endog.loc[keep],
        'X': X.loc[keep],
        'regime': regime.loc[keep],
        'hvar': hvar.loc[keep],
        'cost': cost.loc[keep],
        'ppi_raw': ppi.loc[keep],
        'cost_cols': [c for c in X.columns if c != 'const'],
        'horizon': horizon,
    }
    return panel


if __name__ == '__main__':
    p = build_panel()
    for k, v in p.items():
        print(f'{k:10s}', getattr(v, "shape", v))
    print(p['X'].tail())
    print(p['endog'].tail())

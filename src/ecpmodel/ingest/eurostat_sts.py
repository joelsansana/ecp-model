"""Eurostat Short-term Business Statistics: basic-chemicals PPI.

The ``eurostat`` package is used with *named* filters, which is robust to
dimension order (the raw SDMX path is not).  NACE C20 is "manufacture of
chemicals and chemical products"; C201 is the ethylene sub-class, which is not
published for most countries - the fallbacks in ``configs/data.yaml`` handle
that.

This series is weak evidence about ethylene specifically.  Run the model with
PPI switched off and compare out of sample before keeping it.
"""

from __future__ import annotations

import warnings

import pandas as pd

from ecpmodel.io import save_interim, save_raw


def _tidy(df: pd.DataFrame) -> pd.Series:
    """STS comes back wide: one row, one column per YYYY-MM period."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    time_cols = [c for c in df.columns if c[:2].isdigit() and len(c) == 7]
    if not time_cols:
        raise ValueError(f'no time columns in STS response: {list(df.columns)[:10]}')
    s = df[time_cols].iloc[0]
    s.index = pd.to_datetime(s.index, format='%Y-%m')
    return s.astype(float).dropna()


def ppi(geo: str = 'DE', nace: str = 'C20', dataset: str = 'sts_inpp_m') -> pd.Series:
    """Producer price index for chemicals, NSA, I21, PRC_PRR, not seasonally adj.

    The ``eurostat`` package takes *named* filters, which is robust to the
    dimension order of the SDMX key (the raw URL path is not).
    """
    import eurostat

    filters = {'geo': [geo], 'nace_r2': [nace], 'unit': ['I21'],
               's_adj': ['NSA'], 'indic_bt': ['PRC_PRR']}
    try:
        df = eurostat.get_data_df(dataset, filter_pars=filters)
    except Exception as exc:  # noqa: BLE001 - eurostat raises bare assertion errors
        raise ValueError(f'{dataset} pull failed for {geo}/{nace}: {exc}') from exc
    if df is None or df.empty:
        raise ValueError(f'no {dataset} rows for geo={geo} nace_r2={nace}')

    save_raw(df, f'sts_ppi_{geo}_{nace}')
    s = _tidy(df).rename(f'ppi_{nace}_{geo}')
    save_interim(s, s.name)
    return s


def pull(cfg: dict | None = None) -> pd.Series | None:
    """Pull the configured PPI, walking the fallback list on failure."""
    from ecpmodel.io import load_cfg

    cfg = cfg if cfg is not None else load_cfg('data')
    e = cfg['eurostat']
    candidates = [{'geo': e['geo'], 'nace': e['nace']}, *e.get('fallbacks', [])]
    for cand in candidates:
        try:
            return ppi(cand['geo'], cand['nace'], e['dataset'])
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f'PPI pull failed for {cand}: {exc}', stacklevel=2)
    warnings.warn('no usable PPI series - the PPI observation equation will be dropped',
                  stacklevel=2)
    return None


if __name__ == '__main__':
    s = pull()
    print(None if s is None else s.tail())

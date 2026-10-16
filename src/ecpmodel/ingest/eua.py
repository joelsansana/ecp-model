"""EUA carbon price - hand-curated, because there is no stable free API.

Create ``data/manual/eua_monthly.csv``::

    date,eua_eur_t,source
    2024-01-01,71.23,Ember carbon price tracker (monthly avg of settles)

Sources that are free to redistribute: Ember's carbon price tracker download,
EEX primary auction result tables, or a delayed exchange feed.  If the file is
absent the series is dropped from the cost anchor rather than silently faked.
"""

from __future__ import annotations

import warnings

import pandas as pd

from ecpmodel.io import load_cfg, manual_path, save_interim, save_raw


def pull(path: str | None = None, cfg: dict | None = None) -> pd.Series:
    cfg = cfg if cfg is not None else load_cfg('data')
    path = path or str(manual_path('eua', cfg))

    try:
        df = pd.read_csv(path, parse_dates=['date'], comment='#')
    except FileNotFoundError:
        warnings.warn(
            f'{path} not found - the EUA carbon term will be dropped from the cost anchor',
            stacklevel=2,
        )
        return pd.Series(dtype=float, name='eua_eur_t')

    save_raw(df, 'eua_manual')
    s = (df.set_index('date')['eua_eur_t'].astype(float)
           .resample('MS').mean().dropna())
    if s.empty:
        warnings.warn(f'{path} has no EUA rows yet - the carbon term will be '
                      'dropped from the cost anchor', stacklevel=2)
        return pd.Series(dtype=float, name='eua_eur_t')
    save_interim(s, 'eua_eur_t')
    return s


if __name__ == '__main__':
    print(pull().tail())

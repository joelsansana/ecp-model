"""FRED pulls (Brent crude, IMF EU natural gas price).

``fredgraph.csv`` needs no API key.  The date column has been called ``DATE``,
``observation_date`` and ``DATE`` over time, so columns are assigned
positionally rather than by name.
"""

from __future__ import annotations

import pandas as pd

from ecpmodel.ingest._http import cached, get, read_csv_response
from ecpmodel.io import load_cfg, save_interim, save_raw


def fetch_fred(series_id: str) -> pd.Series:
    """Daily FRED series as a time-indexed float series."""
    def _pull() -> pd.DataFrame:
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
        r = get(url, timeout=60)
        return read_csv_response(r.text)

    df = cached(f'fred_{series_id}', _pull).copy()
    df.columns = ['date', 'value'][: len(df.columns)]
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    save_raw(df, f'fred_{series_id}')
    return df.dropna().set_index('date')['value']


def monthly(s: pd.Series) -> pd.Series:
    return s.resample('MS').mean()


def pull(cfg: dict | None = None) -> tuple[pd.Series, pd.Series]:
    cfg = cfg if cfg is not None else load_cfg('data')
    brent = monthly(fetch_fred(cfg['fred']['brent']))    # USD/bbl
    gas = monthly(fetch_fred(cfg['fred']['eu_gas']))     # USD/MMBtu

    save_interim(brent, 'brent_usd_bbl')
    save_interim(gas, 'gas_usd_mmbtu')
    return brent, gas


if __name__ == '__main__':
    print(pull()[0].tail())

"""ECB FX reference rates (SDMX 2.1 data API).

Convention used everywhere in this repo: **USD per 1 EUR**.  Divide USD-denominated
prices by this series to get EUR.
"""

from __future__ import annotations

import pandas as pd

from ecpmodel.ingest._http import cached, get, read_csv_response
from ecpmodel.io import save_interim, save_raw, to_month_start

BASE = 'https://data-api.ecb.europa.eu/service/data'


def _series(series: str, freq: str, start: str) -> pd.DataFrame:
    # the ECB SDMX path is <base>/<dataset>/<key>, e.g. EXR/M.USD.EUR.SP00.A
    url = f'{BASE}/{series}/{freq}.USD.EUR.SP00.A'
    r = get(url, params={'format': 'csvdata', 'startPeriod': start})
    return read_csv_response(r.text)


def fetch_eurusd(start: str = '2012-01') -> pd.Series:
    """Monthly average USD per EUR, indexed by month start."""
    df = cached('ecb_eurusd', _series, 'EXR', 'M', start)

    if 'TIME_PERIOD' not in df.columns:  # SDMX-CSV fallback shape
        time_col = next(c for c in df.columns if c.lower().startswith('time'))
        df = df.rename(columns={time_col: 'TIME_PERIOD'})
    obs = 'OBS_VALUE' if 'OBS_VALUE' in df.columns else df.columns[-1]

    out = pd.to_numeric(df[obs], errors='coerce').to_numpy()
    out = pd.Series(out, index=to_month_start(df['TIME_PERIOD']), name='eurusd').dropna()

    save_raw(df, 'ecb_eurusd')
    save_interim(out, 'eurusd')
    return out


if __name__ == '__main__':
    print(fetch_eurusd().tail())

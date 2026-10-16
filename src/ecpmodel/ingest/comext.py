"""Eurostat Comext: ethylene (HS 290121) and co-product unit values.

Dataset ``ds-045409`` via the Comext SDMX 2.1 API.

**Verify before you trust this module.** Two things change without notice:

1. the dimension order in the data key (this module assumes
   ``frequency.reporter.partner.product.flow.indicator``);
2. the indicator codes.

Print ``df.columns`` and ``df.head()`` from the first call and compare against
the dataset structure definition on the Eurostat Comext documentation page.

The observation the model actually uses is a **volume-weighted unit value**
(``VALUE_IN_EUROS / tonnes``) aggregated over reporters, with a thin-cell
filter: ethylene is mostly pipeline-traded, so many reporter-partner cells are
too small to mean anything and are dominated by mix and rounding.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecpmodel.ingest._http import get, read_csv_response
from ecpmodel.io import RAW, load_cfg, offline, save_interim, save_raw

BASE = 'https://ec.europa.eu/eurostat/api/comext/dissemination/sdmx/2.1/data/ds-045409'
CELLS = RAW / '_cells'
KG_PER_100KG_TO_TONNE = 10.0     # QUANTITY_IN_100KG counts 100 kg units

DEFAULT_INDICATORS = ['VALUE_IN_EUROS', 'QUANTITY_IN_100KG']


def _cell_path(key: str) -> Path:
    return CELLS / (key.replace('.', '_') + '.parquet')


def fetch_cell(reporter: str, partner: str, product: str, flow: int, indicator: str,
               start: str, sleep: float = 0.3, refresh: bool = False) -> pd.DataFrame:
    """One reporter/partner/product/flow/indicator series.  Empty frame = no data."""
    key = f'M.{reporter}.{partner}.{product}.{flow}.{indicator}'
    path = _cell_path(key)
    if os.environ.get('ECP_REFRESH', '').lower() in {'1', 'true', 'yes'}:
        refresh = True

    if path.exists() and not refresh and not offline():
        return pd.read_parquet(path)

    if offline() and not path.exists():
        return pd.DataFrame()

    r = get(f'{BASE}/{key}', params={'format': 'SDMX-CSV', 'startPeriod': start},
            timeout=180, allow_404=True)
    if r is None:
        df = pd.DataFrame()
    else:
        df = read_csv_response(r.text)
        if len(df):
            df['indicator'] = indicator
    CELLS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)               # cache the cell so re-runs stay polite
    time.sleep(sleep)
    return df


def _value_column(df: pd.DataFrame) -> str:
    for cand in ('OBS_VALUE', 'value', 'Value'):
        if cand in df.columns:
            return cand
    return df.columns[-1]


def pull_product(cfg: dict, name: str, refresh: bool = False) -> pd.DataFrame:
    """All cells for one product code, as a tidy frame."""
    c = cfg['comext']
    code = c['products'][name]
    indicators = c.get('indicators', DEFAULT_INDICATORS)

    frames = []
    for rep in c['reporters']:
        for part in c['partners']:
            for flow_name, flow in c['flows'].items():
                for ind in indicators:
                    d = fetch_cell(rep, part, code, flow, ind, cfg['start'], refresh=refresh)
                    if len(d):
                        d = d.copy()
                        d['indicator'] = ind
                        d['reporter'], d['partner'] = rep, part
                        d['flow'] = flow_name
                        d['product'] = name
                        frames.append(d)

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(raw):
        raw = raw.rename(columns={_value_column(raw): 'OBS_VALUE'})
        raw['OBS_VALUE'] = pd.to_numeric(raw['OBS_VALUE'], errors='coerce')
    save_raw(raw, f'comext_{name}')
    return raw


def unit_values(raw: pd.DataFrame, min_tonnes: float = 200) -> pd.DataFrame:
    """Collapse cells to (month, reporter, partner, flow) unit values in EUR/t."""
    if not len(raw):
        return pd.DataFrame(columns=['month', 'reporter', 'partner', 'flow', 'uv',
                                     'uv_med', 'tonnes', 'value_eur'])

    df = raw.copy()
    df['month'] = (pd.to_datetime(df['TIME_PERIOD'])
                     .dt.to_period('M').dt.to_timestamp())

    keys = ['month', 'reporter', 'partner', 'flow']
    wide = (df.pivot_table(index=keys, columns='indicator',
                           values='OBS_VALUE', aggfunc='sum')
              .reset_index())
    for col in DEFAULT_INDICATORS:
        if col not in wide.columns:
            wide[col] = np.nan

    wide['tonnes'] = wide['QUANTITY_IN_100KG'] / KG_PER_100KG_TO_TONNE
    wide = wide[wide['tonnes'] >= min_tonnes].copy()
    wide['value_eur'] = wide['VALUE_IN_EUROS']
    wide['uv'] = wide['value_eur'] / wide['tonnes']        # EUR per tonne
    return wide.sort_values(keys)


def aggregate(wide: pd.DataFrame) -> pd.DataFrame:
    """Reporter/partner/flow cells -> one row per month."""
    if not len(wide):
        return pd.DataFrame(columns=['uv_w', 'uv_med', 'tonnes', 'n_cells'])

    def agg(g):
        w = g['tonnes'].to_numpy(float)
        return pd.Series({
            'uv_w': float(np.average(g['uv'].to_numpy(float), weights=w)),
            'uv_med': float(g['uv'].median()),
            'tonnes': float(w.sum()),
            'n_cells': float(len(g)),
        })

    return wide.groupby('month').apply(agg, include_groups=False)


def pull(cfg: dict | None = None, refresh: bool = False) -> dict[str, pd.DataFrame]:
    cfg = cfg if cfg is not None else load_cfg('data')
    out: dict[str, pd.DataFrame] = {}
    for name in cfg['comext']['products']:
        raw = pull_product(cfg, name, refresh=refresh)
        agg = aggregate(unit_values(raw, cfg['comext']['min_tonnes']))
        if len(agg):
            save_interim(agg['uv_w'], f'uv_{name}')
            save_interim(agg['uv_med'], f'uvmed_{name}')
            save_interim(agg['tonnes'], f'tonnes_{name}')
        out[name] = agg
    return out


if __name__ == '__main__':
    res = pull()
    for k, v in res.items():
        print(k, v.tail(3))

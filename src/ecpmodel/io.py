"""Shared paths, config loading and parquet helpers.

Every cleaned series is a monthly ``pandas.Series`` indexed by month-start
timestamps (``freq='MS'``).  Availability is *never* applied during ingestion -
that happens once, in :mod:`ecpmodel.features.build`, so that a single as-of
function governs both the fit and the backtest.
"""

from __future__ import annotations

import datetime as dt
import functools
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RAW = DATA / 'raw'
INTERIM = DATA / 'interim'
PROCESSED = DATA / 'processed'
MANUAL = DATA / 'manual'
CONFIGS = ROOT / 'configs'

MS = 'MS'


@functools.cache
def load_cfg(name: str = 'data') -> dict:
    """Load ``configs/<name>.yaml`` (cached)."""
    with open(CONFIGS / f'{name}.yaml') as fh:
        return yaml.safe_load(fh)


def manual_path(kind: str, cfg: dict | None = None) -> Path:
    """Resolve a hand-curated file, honouring the env var override."""
    env = os.environ.get(f'ECP_{kind.upper()}')
    if env:
        return Path(env)
    cfg = cfg if cfg is not None else load_cfg('data')
    return ROOT / cfg['manual'][kind]


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def to_month_start(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).to_period('M').to_timestamp()


def save_raw(df: pd.DataFrame, name: str) -> Path:
    """Store an untouched pull twice: a dated snapshot (vintage archive) and a
    ``__latest`` pointer used by everything downstream."""
    ensure(RAW)
    df.to_parquet(RAW / f'{name}_latest.parquet')
    dated = RAW / f'{name}__{dt.date.today().isoformat()}.parquet'
    df.to_parquet(dated)
    return dated


def load_raw(name: str) -> pd.DataFrame:
    return pd.read_parquet(RAW / f'{name}__latest.parquet')


def save_interim(s: pd.Series, name: str) -> Path:
    ensure(INTERIM)
    s = pd.Series(s).astype(float).rename(name)
    s.index = to_month_start(s.index)
    s = s[~s.index.duplicated(keep='last')].sort_index()
    path = INTERIM / f'{name}.parquet'
    s.to_frame().to_parquet(path)
    return path


def load_interim(name: str) -> pd.Series:
    """Read a cleaned monthly series; missing file -> empty series."""
    path = INTERIM / f'{name}.parquet'
    if not path.exists():
        return pd.Series(dtype=float, name=name)
    return pd.read_parquet(path).iloc[:, 0].rename(name)


def has_interim(name: str) -> bool:
    return (INTERIM / f'{name}.parquet').exists()


def save_processed(df: pd.DataFrame, name: str) -> Path:
    ensure(PROCESSED)
    path = PROCESSED / f'{name}.parquet'
    df.to_parquet(path)
    return path


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f'{name}.parquet')


def offline() -> bool:
    """``ECP_OFFLINE=1`` makes ingestion reuse cached raw pulls only."""
    return os.environ.get('ECP_OFFLINE', '').lower() in {'1', 'true', 'yes'}

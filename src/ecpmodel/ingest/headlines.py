"""The ground-truth file: free-headline ECP settlements.

Nothing else in this repo matters as much as ``data/manual/ecp_headlines.csv``.
The model cannot observe the contract price; this file is the only direct
evidence of its level, and it is sparse, lagged and noisy.

Record only the data point you are legally allowed to quote - never article
text.  Field rules:

``month``          the month the price applies to (first month for a quarterly deal)
``published_on``   the date you could *first* have seen it; used by the backtest.
                   Never backfill it with the month itself.
``type``           ``monthly`` or ``quarterly``.  A quarterly contract becomes three
                   monthly observations, each carrying the full information (hence
                   the variance multiplier ``repl``).
``reliability``    1 producer statement, 2 reputable trade press, 3 hearsay.
``low``/``high``   optional published range; keep it for interval diagnostics.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:  # pandera >= 0.25 wants the pandas backend named explicitly
    import pandera.pandas as pa
except ImportError:  # pragma: no cover - older pandera
    import pandera as pa

from ecpmodel.io import manual_path, save_raw

EXAMPLE = 'data/manual/ecp_headlines.example.csv'

schema = pa.DataFrameSchema({
    'month': pa.Column('datetime64[ns]', coerce=True, required=True),
    'settlement_eur_t': pa.Column(float, pa.Check.in_range(200, 4000), coerce=True),
    'type': pa.Column(str, pa.Check.isin(['monthly', 'quarterly'])),
    'published_on': pa.Column('datetime64[ns]', coerce=True, required=True),
    'reliability': pa.Column(int, pa.Check.isin([1, 2, 3])),
}, coerce=True, strict=False)


def _resolve(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    return manual_path('headlines')


def validate(df: pd.DataFrame) -> pd.DataFrame:
    return schema.validate(df)


def load(path: str | Path | None = None, allow_example: bool = True) -> pd.DataFrame:
    """Return the long-format headline frame used by the model.

    Columns: ``month, y (EUR/t), available_on, rel, repl, low, high``.
    """
    p = _resolve(path)
    if not p.exists() and allow_example:
        fallback = Path(__file__).resolve().parents[3] / EXAMPLE
        if fallback.exists():
            warnings.warn(
                f'{p} not found - falling back to the ILLUSTRATIVE {fallback.name}. '
                'Numbers from that file are placeholders; backfill the real thing.',
                stacklevel=2,
            )
            p = fallback
    if not p.exists():
        raise FileNotFoundError(
            f'no headline file at {p}. Copy data/manual/ecp_headlines.example.csv, '
            'fill in settlements you can legally quote, and back at least 3 years.'
        )

    df = pd.read_csv(p, parse_dates=['month', 'published_on'], comment='#')
    df = validate(df)
    save_raw(df, 'headlines_ecp')

    for col, default in (('low', np.nan), ('high', np.nan)):
        if col not in df.columns:
            df[col] = default

    rows = []
    for _, r in df.iterrows():
        n = 3 if r['type'] == 'quarterly' else 1
        for k in range(n):
            rows.append({
                'month': r['month'] + pd.DateOffset(months=k),
                'y': float(r['settlement_eur_t']),
                'available_on': r['published_on'],
                'rel': int(r['reliability']),
                'repl': n,
                'low': float(r['low']) if pd.notna(r['low']) else np.nan,
                'high': float(r['high']) if pd.notna(r['high']) else np.nan,
            })
    out = (pd.DataFrame(rows)
           .sort_values(['month', 'available_on'])
           .drop_duplicates(subset=['month'], keep='last')
           .reset_index(drop=True))
    return out

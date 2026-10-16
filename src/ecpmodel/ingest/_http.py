"""Tiny retry/offline wrapper around ``requests`` shared by the ingest modules.

Comext and the ECB SDW endpoint are both rate-sensitive; a plain
``requests.get`` with a 180s timeout is not enough for a monthly cron job.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from ecpmodel.io import offline

_UA = {'User-Agent': 'ecpmodel/0.1 (+https://example.invalid; research use)'}


def get(url: str, params: dict | None = None, timeout: int = 120, tries: int = 4,
        allow_404: bool = False) -> requests.Response | None:
    """GET with exponential backoff.  Returns ``None`` when ``allow_404`` and
    the server answers 404 (i.e. "this cell legitimately has no data")."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=_UA)
            if r.status_code == 404 and allow_404:
                return None
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001 - retry on anything transport-ish
            last = exc
            if attempt < tries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f'GET failed after {tries} tries: {url} ({last})')


def cached(name: str, loader, *args, **kwargs):
    """Run ``loader`` unless offline and a raw snapshot already exists."""
    from ecpmodel.io import load_raw

    if offline():
        try:
            return load_raw(name)
        except FileNotFoundError:
            raise RuntimeError(
                f'ECP_OFFLINE=1 but no cached pull for {name!r}; run `make data` online first'
            ) from None
    return loader(*args, **kwargs)


def read_csv_response(text: str, **kwargs) -> pd.DataFrame:
    import io as _io

    return pd.read_csv(_io.StringIO(text), **kwargs)

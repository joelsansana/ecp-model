"""``make data`` - pull every public series, then record the vintage.

Phase A (ECB FX, FRED Brent/gas, manual EUA), B (Eurostat STS PPI), C (Comext).
Comext is the fragile one: start with a single reporter and a single product,
check the unit values look sane against your headline range, then widen.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

from ecpmodel.ingest import comext, ecb, eua, eurostat_sts, fred, headlines
from ecpmodel.io import load_cfg

STEPS = {
    'fx': ('ECB EUR/USD reference rate', lambda cfg: ecb.fetch_eurusd(cfg['start'])),
    'fred': ('FRED Brent + EU gas', lambda cfg: fred.pull(cfg)),
    'eua': ('EUA carbon (manual csv)', lambda cfg: eua.pull(cfg=cfg)),
    'ppi': ('Eurostat STS chemicals PPI', lambda cfg: eurostat_sts.pull(cfg)),
    'comext': ('Eurostat Comext unit values', lambda cfg: comext.pull(cfg)),
    'headlines': ('ECP settlement headlines (manual csv)', lambda cfg: headlines.load()),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', nargs='*', choices=sorted(STEPS), default=sorted(STEPS),
                    help='subset of steps to run (default: all)')
    ap.add_argument('--strict', action='store_true',
                    help='fail on the first error instead of continuing')
    ap.add_argument('--refresh', action='store_true', help='ignore cached Comext cells')
    args = ap.parse_args(argv)

    cfg = load_cfg('data')
    if args.refresh:
        os.environ['ECP_REFRESH'] = '1'
    failures = []
    for step in args.only:
        label, fn = STEPS[step]
        t0 = time.time()
        print(f'[data] {step:10s} {label} ...', flush=True)
        try:
            out = fn(cfg)
            n = len(out) if hasattr(out, '__len__') else '-'
            print(f'[data] {step:10s} ok ({n} rows, {time.time() - t0:.1f}s)', flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append((step, exc))
            print(f'[data] {step:10s} FAILED: {exc}', flush=True)
            if args.strict:
                raise

    if failures:
        warnings.warn(f'{len(failures)} step(s) failed: {[f[0] for f in failures]}', stacklevel=1)
        return 1
    print('[data] done. Now open notebooks/01_explore_data.ipynb before modelling.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

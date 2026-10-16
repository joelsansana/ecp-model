"""Plots and the monthly markdown summary.

Everything here reads from ``data/processed`` - no network, no refits - so the
reporting step is cheap and reproducible.
"""

from __future__ import annotations

import argparse
import datetime as dt

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecpmodel.io import INTERIM, PROCESSED, load_cfg, load_interim, load_processed  # noqa: E402

FIGS = PROCESSED / 'figures'
BRAND = ['lo80', 'hi80', 'lo90', 'hi90']


def _live(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the diffuse-start burn-in rows flagged by the model."""
    if 'burn_in' not in df.columns:
        return df
    return df[~df['burn_in'].fillna(False).astype(bool)]


def _finish(fig, path, title=''):
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'[viz] wrote {path}')


def _headline_points(heads) -> pd.Series | None:
    """Accept either the long headline frame or an already-plotted Series."""
    if heads is None:
        return None
    if isinstance(heads, pd.Series):
        return heads
    return pd.Series(heads['y'].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(heads['month']))


def plot_latent(df: pd.DataFrame, heads=None,
                path=FIGS / 'ecp_latent.png', title='Latent ECP (nominal, EUR/t)'):
    d = _live(df)
    heads = _headline_points(heads)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if all(c in d.columns for c in BRAND):
        ax.fill_between(d.index, d['lo90'], d['hi90'], color='#c9d6e3', alpha=.6,
                        label='90% band')
        ax.fill_between(d.index, d['lo80'], d['hi80'], color='#7f9fbf', alpha=.6,
                        label='80% band')
    ax.plot(d.index, d['ecp_mean'], color='#12395b', lw=1.8, label='latent ECP')
    if 'cost_anchor' in d:
        ax.plot(d.index, d['cost_anchor'], color='#c1666b', lw=1.0, ls='--',
                label='cost anchor (beta\'x)')
    if heads is not None and len(heads):
        ax.scatter(heads.index, heads.to_numpy(dtype=float), s=26, color='k', zorder=5,
                   label='headlines')
    ax.set_title(title)
    ax.set_ylabel('EUR / t')
    ax.legend(loc='upper left', fontsize=8, ncol=4, frameon=False)
    ax.grid(alpha=.25)
    _finish(fig, path, title)


def plot_margin(df: pd.DataFrame, path=FIGS / 'ecp_margin.png'):
    d = _live(df)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.fill_between(d.index, 0, d['margin'], where=d['margin'] >= 0, color='#3f7d4e',
                    alpha=.5, label='margin over cost anchor')
    ax.fill_between(d.index, 0, d['margin'], where=d['margin'] < 0, color='#a33b3b',
                    alpha=.5)
    ax.axhline(0, color='k', lw=.8)
    ax.set_title('Implied contract margin (latent ECP - cost anchor), EUR/t')
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.grid(alpha=.25)
    _finish(fig, path)


def plot_real(latent: pd.DataFrame, real: pd.DataFrame | None = None,
              ppi_name: str | None = None, base_month: str | None = None,
              path=FIGS / 'ecp_real.png'):
    from ecpmodel.models.real import deflator

    ppi = load_interim(ppi_name) if ppi_name else _first_ppi()
    if ppi is None or not len(ppi):
        print('[viz] no PPI series - skipping the real-price chart')
        return
    base_month = base_month or load_cfg('model')['real_price_base']
    base_month = pd.Timestamp(base_month)
    if base_month not in ppi.index:
        base_month = ppi.index[0]

    d = _live(latent)
    defl = pd.Series(deflator(ppi, d.index, base_month), index=d.index)
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.plot(d.index, d['ecp_mean'] / defl, color='#12395b', lw=1.8,
            label='real ECP (PPI-deflated)')
    if real is not None and len(real):
        ax.plot(real.index, real.get('p50'), color='#7f9fbf', lw=1.0, alpha=.8)
    ax.set_title(f'Real ECP, chemicals PPI rebased to {base_month:%Y-%m}')
    ax.set_ylabel('EUR / t (base-period purchasing power)')
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.grid(alpha=.25)
    _finish(fig, path)


def _first_ppi() -> pd.Series | None:
    hits = sorted(INTERIM.glob('ppi_*.parquet'))
    return pd.read_parquet(hits[0]).iloc[:, 0] if hits else None


def plot_backtest(rep: pd.DataFrame, path=FIGS / 'backtest.png'):
    if not len(rep):
        return
    if 'slice' not in rep.columns:
        rep = rep.reset_index().rename(columns={'index': 'slice'})
    rep = rep[rep['slice'].astype(str).str.startswith(('h=', 'all', 'stress', 'calm'))]
    rep = rep.set_index('slice')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    cols = [c for c in ('rmse_ssm', 'rmse_naive', 'rmse_ols', 'rmse_ssm_core') if c in rep]
    rep[cols].plot.bar(ax=axes[0], rot=45, width=.7)
    axes[0].set_title('RMSE by benchmark (EUR/t)')
    axes[0].set_ylabel('EUR/t')
    axes[0].grid(alpha=.25, axis='y')
    rep[['n', 'cov80']].plot.bar(ax=axes[1], rot=45, width=.7)
    axes[1].axhline(0.8, color='k', ls='--', lw=.8, label='nominal 80%')
    axes[1].set_title('80% coverage (bars) and n (labels in the table)')
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=.25, axis='y')
    _finish(fig, path)


def plot_diagnostics(res, path=FIGS / 'diagnostics.png'):
    """Standardised residuals for each observation equation."""
    names = list(res.model.obs_names)
    err = np.asarray(res.standardized_forecasts_error).reshape(len(names), -1)
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.2 * len(names)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, i, name in zip(axes, range(len(names)), names, strict=True):
        std = err[i]
        ax.plot(res.model.data.row_labels, std, lw=.8, color='#12395b')
        ax.axhline(0, color='k', lw=.6)
        ax.axhline(2, color='#a33b3b', lw=.6, ls=':')
        ax.axhline(-2, color='#a33b3b', lw=.6, ls=':')
        ax.set_title(f'{name}: standardised residuals', fontsize=9)
        ax.grid(alpha=.25)
    _finish(fig, path)


def monthly_summary(latent: pd.DataFrame, path=PROCESSED / 'monthly_summary.md') -> str:
    d = _live(latent)
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else last
    now = d.index[-1]
    q = (d.index[-6:] if len(d) > 6 else d.index)

    lines = [
        f'# ECP nowcast - {now:%Y-%m}',
        f'generated {dt.date.today().isoformat()} by `make model`',
        '',
        f'- **Latent ECP (nominal)**: EUR {last["ecp_mean"]:,.0f}/t '
        f'(80% band {last["lo80"]:,.0f} - {last["hi80"]:,.0f})',
        f'- **90% band**: {last["lo90"]:,.0f} - {last["hi90"]:,.0f}',
        f'- **Change vs previous month**: {last["ecp_mean"] - prev["ecp_mean"]:+,.0f} EUR/t',
    ]
    if 'cost_anchor' in d:
        lines.append(f'- **Cost anchor**: EUR {last["cost_anchor"]:,.0f}/t, '
                     f'implied margin EUR {last["margin"]:,.0f}/t')
    lines += ['', '## Recent path', '',
              '| month | ECP | lo80 | hi80 | anchor | margin |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for m in q:
        r = d.loc[m]
        lines.append(f'| {m:%Y-%m} | {r["ecp_mean"]:,.0f} | {r["lo80"]:,.0f} | '
                     f'{r["hi80"]:,.0f} | {r.get("cost_anchor", float("nan")):,.0f} | '
                     f'{r.get("margin", float("nan")):,.0f} |')
    lines += ['',
              '_Bands are conditional on the fitted parameters. Use '
              '`ecpmodel.models.draws` for parameter-uncertainty bands._', '']
    md = '\n'.join(lines)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md)
        print(f'[viz] wrote {path}')
    return md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Plots and the monthly summary.')
    ap.add_argument('--backtest', default='backtest_raw',
                    help='base name written by `make backtest`')
    args = ap.parse_args(argv)

    latent = load_processed('ecp_latent')
    heads = None
    try:
        from ecpmodel.ingest.headlines import load as load_headlines

        h = load_headlines()
        heads = pd.Series(h['y'].to_numpy(), index=h['month'], name='y')
    except Exception as exc:  # noqa: BLE001
        print(f'[viz] no headlines: {exc}')

    plot_latent(latent, heads)
    plot_margin(latent)
    plot_real(latent)
    try:
        plot_backtest(load_processed(f'{args.backtest}_report'))
    except FileNotFoundError:
        print('[viz] no backtest report yet - run `make backtest`')
    monthly_summary(latent)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Generate the two analysis notebooks.  Run from the repo root:

    python tools/make_notebooks.py

They are checked in; this script just keeps them readable in review.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / 'notebooks'


def md(src: str) -> dict:
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src.strip().splitlines(keepends=True)}


def code(src: str) -> dict:
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': src.strip().splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                           'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


EXPLORE = [
    md("""
# 01 - Explore the public data

Before any modelling: **do not skip this notebook.**  If the ethylene unit
value does not track the headline settlement series after a one-to-two month
lag, you have a unit error or the wrong product code, and every downstream
number will be wrong in a way that looks plausible.

Run `make data` first.
"""),
    code("""
import sys
from pathlib import Path

# work whether the kernel starts in the repo root or in notebooks/
root = next((p.parent for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'src' / 'ecpmodel').is_dir()), Path.cwd())
sys.path.insert(0, str(root / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecpmodel.features.build import MMBTU_TO_MWH, build_panel
from ecpmodel.io import load_cfg, load_interim

pd.options.display.width = 120
plt.style.use('seaborn-v0_8-whitegrid')
"""),
    md("""
## 1. What is in the cache

Ingestion never applies publication lags - that happens once, in
`features/build.py`.  Everything here is therefore "the latest vintage" and is
exactly what a naive (wrong) backtest would use.
"""),
    code("""
SERIES = ['eurusd', 'brent_usd_bbl', 'gas_usd_mmbtu', 'eua_eur_t',
          'uv_ethylene', 'uvmed_ethylene', 'uv_propylene', 'uv_butadiene', 'uv_naphtha']

rows = []
for name in SERIES:
    s = load_interim(name)
    rows.append({'series': name, 'n': len(s),
                 'first': s.index.min() if len(s) else None,
                 'last': s.index.max() if len(s) else None,
                 'last_value': round(float(s.iloc[-1]), 3) if len(s) else None})

extra = sorted(p.stem for p in Path('data/interim').glob('ppi_*.parquet'))
for name in extra:
    s = load_interim(name)
    rows.append({'series': name, 'n': len(s), 'first': s.index.min(), 'last': s.index.max(),
                 'last_value': round(float(s.iloc[-1]), 3)})

pd.DataFrame(rows)
"""),
    md("""
## 2. Brent in EUR/bbl

The ECB quotes **USD per EUR**, so divide the USD price by it.  Getting this
backwards is the single most common data bug in this kind of model.
"""),
    code("""
brent = load_interim('brent_usd_bbl')
fx = load_interim('eurusd')
gas = load_interim('gas_usd_mmbtu')

brent_eur = brent / fx.reindex(brent.index).ffill()
gas_eur_mwh = gas / fx.reindex(gas.index).ffill() / MMBTU_TO_MWH

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(brent.index, brent, label='Brent USD/bbl', lw=1)
axes[0].plot(brent_eur.index, brent_eur, label='Brent EUR/bbl', lw=1.4)
axes[0].set_ylabel('USD or EUR / bbl')
axes[0].legend(frameon=False, fontsize=8)

axes[1].plot(gas_eur_mwh.index, gas_eur_mwh, color='#b5651d', lw=1.4)
axes[1].set_ylabel('gas EUR / MWh')
fig.suptitle('Cost drivers in EUR', y=1.0)
"""),
    md("""
## 3. Ethylene unit value vs the headline settlements

This is the sanity check.  Expect:

* the unit value within a factor of roughly 1.5 of the contract range
  (customs value, mix and delivery basis put them apart);
* the same direction of travel with a one-to-two month lag;
* a level that is **not** 3x off - that would be a units error (100 kg vs t is
  the usual culprit) or the wrong HS code.
"""),
    code("""
uv = load_interim('uv_ethylene')
tonnes = load_interim('tonnes_ethylene')

try:
    from ecpmodel.ingest.headlines import load as load_headlines
    h = load_headlines()
    heads = pd.Series(h['y'].to_numpy(), index=pd.DatetimeIndex(h['month']))
except Exception as exc:                                    # no headline file yet
    heads = pd.Series(dtype=float)
    print('no headlines:', exc)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(uv.index, uv, lw=1.4, label='Comext unit value (EUR/t)')
if len(heads):
    axes[0].scatter(heads.index, heads.to_numpy(), s=28, color='k', zorder=5,
                    label='ECP headline settlements')
axes[0].set_ylabel('EUR / t')
axes[0].legend(frameon=False, fontsize=8)

axes[1].semilogy(tonnes.index, tonnes, lw=1, color='#666')
axes[1].set_ylabel('trade volume, tonnes')
axes[1].set_title('Reported ethylene volume behind the unit value')
"""),
    code("""
# correlation of the unit value with the settlements at several lags
if len(heads) > 5:
    z = pd.concat([uv.rename('uv'), heads.rename('head')], axis=1).ffill()
    rows = []
    for k in range(4):
        rows.append({'comext_lag_months': k,
                     'corr_uv_vs_headline': z['uv'].corr(z['head'].shift(-k))})
    display(pd.DataFrame(rows).round(3))
"""),
    md("""
## 4. Chemicals PPI

Weak evidence about ethylene specifically: it is a whole-economy index for
NACE C20 and is partly endogenous to ethylene itself.  It is also heavily
smoothed, which makes it look informative when it is not.
"""),
    code("""
ppi_files = sorted(Path('data/interim').glob('ppi_*.parquet'))
ppi = pd.read_parquet(ppi_files[0]).iloc[:, 0] if ppi_files else pd.Series(dtype=float)

fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(ppi.index, ppi, color='#444', lw=1.4, label='chemicals PPI (NACE C20)')
if len(uv):
    ax.plot(uv.index, uv / uv.max() * ppi.max(), color='#12395b', lw=1,
            label='ethylene unit value, rescaled to the PPI axis')
ax.legend(frameon=False, fontsize=8)
ax.set_title('Chemicals PPI vs ethylene')
"""),
    md("""
## 5. The as-of panel

This is the object the model consumes.  Availability lags from
`configs/data.yaml` are applied here - and nowhere else.
"""),
    code("""
cfg = load_cfg('data')
panel = build_panel()

X = panel['X']
print('panel months :', len(X), X.index[0].date(), '->', X.index[-1].date())
print('regressors   :', list(X.columns))
print()
panel['endog'].notna().sum().rename('n_observed').to_frame().T
"""),
    code("""
fig, ax = plt.subplots(figsize=(11, 3.4))
for c in ('brent_eur', 'gas_eur_mwh', 'eua'):
    if c in panel['cost']:
        ax.plot(panel['cost'].index, panel['cost'][c], lw=1.2, label=c)
ax.set_title('Cost block used by the model (EUR units, published data only)')
ax.legend(frameon=False, fontsize=8)
"""),
    md("""
## 6. Where the data stops

Each series has a different publication lag, so they end on different months.
This chart is the reason the backtest must be as-of rather than on the latest
vintage.
"""),
    code("""
lagd = cfg['availability_lag_days']
ends = []
for name, key in [('brent_usd_bbl', 'brent'), ('gas_usd_mmbtu', 'eu_gas'),
                  ('eua_eur_t', 'eua'), ('uv_ethylene', 'comext'),
                  (ppi_files[0].stem if ppi_files else 'ppi', 'ppi')]:
    s = load_interim(name)
    if len(s):
        ends.append({'series': name, 'lag_days': lagd.get(key), 'last_month': s.index.max()})
ends.append({'series': 'ECP headline', 'lag_days': lagd.get('headline'),
             'last_month': heads.index.max() if len(heads) else None})
display(pd.DataFrame(ends))
"""),
    md("""
## 7. What to check before you model

1. Brent in EUR/bbl moves with EUR gas - if not, your FX sign is wrong.
2. The unit value tracks the settlements at lag 1-2. If the best correlation is
   at lag 0 or 3, check the product code before touching the model.
3. The unit value is within ~1.5x of the settlement range, not 3x off.
4. The PPI is smooth and slow. That is expected; do not read it as a signal.
5. The panel has at least 25-30 headline observations. Without them the model
   is a cost anchor plus smoothed trade data, not an estimate of the contract.
"""),
]

DIAGNOSTICS = [
    md("""
# 02 - Fit, diagnose, and choose between variants

Run `make data` and `make model` first.  This notebook is where you decide what
the model is allowed to keep.
"""),
    code("""
import sys
from pathlib import Path

# work whether the kernel starts in the repo root or in notebooks/
root = next((p.parent for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'src' / 'ecpmodel').is_dir()), Path.cwd())
sys.path.insert(0, str(root / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecpmodel.features.build import REL_MULT, build_panel
from ecpmodel.io import load_cfg, load_interim
from ecpmodel.models.draws import param_draws, state_draws
from ecpmodel.models.ecm_ssm import fit, summarize
from ecpmodel.models.real import deflator, real_draws

pd.options.display.width = 120
plt.style.use('seaborn-v0_8-whitegrid')
"""),
    code("""
panel = build_panel(horizon=3)
mod, res = fit(panel, obs=['headline', 'comext'], quiet=False)
print(res.summary())
"""),
    md("""
## 1. Parameter table

Read the signs before anything else.  Every coefficient on the cost block should
be positive (higher feedstock cost, higher contract price); `gamma` must be
negative (the price closes the gap) and `theta` in [0, 1] (partial same-month
pass-through).  `gamma` is weakly identified in short samples - report its
standard error and do not interpret a half-life without one.
"""),
    code("""
params = pd.Series(np.asarray(res.params), index=mod.param_names, name='coef')
se = pd.Series(np.sqrt(np.diag(np.asarray(res.cov_params()))), index=mod.param_names,
               name='std_err')
table = pd.concat([params, se], axis=1)
table['t'] = table['coef'] / table['std_err'].replace(0, np.nan)
table
"""),
    md("""
## 2. Standardised residuals

You want roughly unit variance, no obvious autocorrelation and no trending
bias.  Run the Ljung-Box test per observation equation.
"""),
    code("""
fig, axes = plt.subplots(len(mod.obs_names), 1, figsize=(11, 2.3 * len(mod.obs_names)),
                         sharex=True)
axes = np.atleast_1d(axes)
idx = mod.data.row_labels
err = np.asarray(res.standardized_forecasts_error).reshape(len(mod.obs_names), -1)
for ax, i, name in zip(axes, range(len(mod.obs_names)), mod.obs_names):
    e = err[i]
    ax.plot(idx, e, lw=.8, color='#12395b')
    ax.axhline(0, color='k', lw=.6)
    ax.axhline(2, color='#a33b3b', lw=.6, ls=':')
    ax.axhline(-2, color='#a33b3b', lw=.6, ls=':')
    ax.set_title(f'{name}: standardised residuals', fontsize=9)
"""),
    code("""
from statsmodels.stats.diagnostic import acorr_ljungbox

# statsmodels' built-in test is univariate-only, so run it per observation
err = np.asarray(res.standardized_forecasts_error).reshape(len(mod.obs_names), -1)
for i, name in enumerate(mod.obs_names):
    e = err[i][~np.isnan(err[i])]
    if len(e) < 30:
        print(f'{name:10s} too few residuals ({len(e)}) to test')
        continue
    q = acorr_ljungbox(e, lags=[6, 12], return_df=True)
    print(f'{name:10s} Ljung-Box  h=6: Q={q.loc[6, "lb_stat"]:8.2f} p={q.loc[6, "lb_pvalue"]:.4f}'
          f'   h=12: Q={q.loc[12, "lb_stat"]:8.2f} p={q.loc[12, "lb_pvalue"]:.4f}')
"""),
    md("""
## 3. Smoothed ECP versus the cost anchor

The gap is the implied cracker margin.  It should look economically sensible -
compressed in downturns, wide in tight markets.  If it sits permanently far
positive or negative, the intercept or `a_c` is soaking up a basis difference
rather than a real margin.

The first `k_states - 1` months are diffuse-start burn-in; hide them.
"""),
    code("""
out = summarize(res, panel)
live = out[~out['burn_in']]

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.fill_between(live.index, live['lo80'], live['hi80'], color='#7f9fbf', alpha=.5,
                label='80% band')
ax.plot(live.index, live['ecp_mean'], color='#12395b', lw=1.8, label='latent ECP')
ax.plot(live.index, live['cost_anchor'], color='#c1666b', lw=1.1, ls='--',
        label="cost anchor (beta'x)")
ax.set_ylabel('EUR / t')
ax.legend(frameon=False, fontsize=8)
ax.set_title('Latent ECP and the estimated cost anchor')
"""),
    code("""
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.fill_between(live.index, 0, live['margin'],
                where=live['margin'] >= 0, color='#3f7d4e', alpha=.5)
ax.fill_between(live.index, 0, live['margin'],
                where=live['margin'] < 0, color='#a33b3b', alpha=.5)
ax.axhline(0, color='k', lw=.8)
ax.set_title('Implied margin (latent ECP - cost anchor), EUR/t')
"""),
    md("""
## 4. Variant comparison - in-sample log-likelihood

Likelihood comparison only *proposes*; out-of-sample CRPS decides.  The
question each row answers:

* **lag weights**: pick among 3-5 shapes, never a continuous grid.
* **comext lag**: 1 vs 2 (a lag-2 Comext needs a deeper state).
* **PPI on/off**: the PPI is a weak instrument - if turning it on makes
  `gamma` collapse to zero and `b_p` explode, it has taken over the price.
* **regime on/off**: is `kappa` significantly above zero, and does it help
  out of sample?
"""),
    code("""
rows = []
for w in load_cfg('model')['lag_weights']:
    panel_w = build_panel(horizon=0, lag_w=w)
    try:
        m, r = fit(panel_w, n_starts=2, obs=['headline', 'comext'])
        rows.append({'variant': f'lag_w={tuple(w)}', 'llf': r.llf,
                     'gamma': r.params[m.layout.idx['gamma']],
                     'n_headlines': int(panel_w['endog']['headline'].notna().sum())})
    except Exception as exc:
        rows.append({'variant': f'lag_w={tuple(w)}', 'llf': np.nan,
                     'gamma': np.nan, 'n_headlines': 0, 'error': str(exc)})
pd.DataFrame(rows).round(3)
"""),
    code("""
rows = []
for label, obs in [('headline only', ['headline']),
                   ('headline+comext', ['headline', 'comext']),
                   ('headline+comext+ppi', ['headline', 'comext', 'ppi'])]:
    try:
        m, r = fit(panel, n_starts=3, obs=obs)
        rows.append({'variant': label, 'llf': r.llf,
                     'gamma': r.params[m.layout.idx['gamma']],
                     'theta': r.params[m.layout.idx['theta']],
                     'b_p': r.params[m.layout.idx['b_p']] if 'b_p' in m.layout.idx else np.nan,
                     's_p': r.params[m.layout.idx['s_p']] if 's_p' in m.layout.idx else np.nan})
    except Exception as exc:
        rows.append({'variant': label, 'llf': np.nan, 'error': str(exc)})
pd.DataFrame(rows).round(4)
"""),
    md("""
**Reading the PPI row.**  If `b_p` is large (several k EUR/t per unit of the
standardised index) and `s_p` is tiny, the index has taken the place of the
price: the filter is no longer estimating a contract price, it is reporting a
chemicals PPI.  That is the failure mode the tutorial warns about, and it is
why the shipped default leaves PPI switched off.
"""),
    md("""
## 5. Posterior draws: bands that include parameter uncertainty

The smoother bands are conditional on the fitted parameters.  Draw parameters
from their asymptotic normal and run the simulation smoother for each draw.
"""),
    code("""
draws = param_draws(res, n=200, seed=0)
paths = state_draws(mod, draws, seed=0, results=res)          # (n_draws, nobs), k EUR/t

q = np.quantile(paths * 1000.0, [0.05, 0.5, 0.95], axis=0)
draws_df = pd.DataFrame({'p05': q[0], 'p50': q[1], 'p95': q[2]}, index=idx)
draws_df.tail()
"""),
    code("""
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.fill_between(live.index, live['lo90'], live['hi90'], color='#d9d9d9', alpha=.8,
                label='90%, smoother only')
ax.fill_between(live.index, draws_df['p05'].reindex(live.index),
                draws_df['p95'].reindex(live.index), color='#7f9fbf', alpha=.6,
                label='90% incl. parameter uncertainty')
ax.plot(live.index, live['ecp_mean'], color='#12395b', lw=1.8, label='smoother mean')
ax.plot(draws_df.index, draws_df['p50'], color='#c1666b', lw=1.0, ls='--',
        label='draws median')
ax.legend(frameon=False, fontsize=8)
ax.set_ylabel('EUR / t')
ax.set_title('Smoother bands vs simulation-smoother draws')
"""),
    md("""
## 6. Real price

Two definitions, different questions: the PPI deflator is a purchasing-power
view, the Brent ratio is a margin view.  For forecast months the deflator is
held at its last value - document that, because the bands then ignore
inflation risk.
"""),
    code("""
ppi_files = sorted(Path('data/interim').glob('ppi_*.parquet'))
ppi = pd.read_parquet(ppi_files[0]).iloc[:, 0] if ppi_files else None

if ppi is not None and len(ppi):
    base = ppi.index[0]
    d = deflator(ppi, idx, base)
    real_draw = real_draws(paths, idx, ppi, base)
    rq = np.quantile(real_draw, [0.05, 0.5, 0.95], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx, rq[1], color='#12395b', lw=1.6, label='real ECP, median')
    ax.fill_between(idx, rq[0], rq[2], color='#7f9fbf', alpha=.4, label='90% band')
    ax.set_ylabel(f'EUR / t, deflated by NACE C20 PPI (base {base:%Y-%m})')
    ax.legend(frameon=False, fontsize=8)
else:
    print('no PPI series - run `make data` first')
"""),
    md("""
## 7. Before you publish

* `make backtest` and report `n` next to every number.
* Compare against the naive and cost-plus-OLS benchmarks at h=0 and h=1.
* Check 80% coverage by horizon, and separately over the 2021-22 window.
* Snapshot the raw pulls every month (`make data` writes dated files) - Comext
  revisions are the biggest silent source of overstatement.
* State plainly what this is: a *reference* price for central Europe inferred
  from public proxies. It is not any single buyer's contract; collars, rebates,
  indexation formulas and delivery basis are invisible in public data.
"""),
]


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in (('01_explore_data.ipynb', EXPLORE),
                        ('02_diagnostics.ipynb', DIAGNOSTICS)):
        path = NB_DIR / name
        path.write_text(json.dumps(notebook(cells), indent=1) + '\n')
        print('wrote', path)


if __name__ == '__main__':
    main()
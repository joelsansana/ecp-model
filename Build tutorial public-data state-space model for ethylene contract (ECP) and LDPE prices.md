# Build tutorial: public-data state-space model for the real ethylene contract price (ECP)

A step-by-step repo build: data-pull plan, state-space equations, estimation, backtesting. Part 1 covers setup and data; Part 2 covers the model, evaluation and operations.

## 0. What you are building

The European ethylene contract price (ECP) is negotiated bilaterally and published only by paid services. With public data only, you cannot observe it. You can, however, observe several noisy signals that are driven by it:

- Cost drivers (Brent, gas, carbon, FX) that anchor it.
- Sparse free headlines of monthly/quarterly settlements.
- Eurostat Comext trade unit values for ethylene (HS 290121), lagged.
- Producer price indices for basic chemicals, heavily smoothed.

The repo estimates a **latent ECP** with a Kalman filter/smoother (error-correction state-space model) and returns a posterior band rather than a point, then converts it to a **real (deflated) price**. Everything is backtested with vintage discipline (only data that was public at the forecast date).

**Deliverables at the end:**

1. `make data` pulls and caches every public series with a recorded availability date.
2. `make model` fits the ethylene state-space model and writes a latent-ECP series with 80/90% bands; the real-price transforms (Section 11) deflate it.
3. `make backtest` runs rolling-origin evaluation against benchmarks.
4. `make test` runs unit tests, including a simulate-and-recover test of the state-space model.

## 1. Repo layout

```
ecp-model/
  pyproject.toml
  Makefile
  README.md
  .env.example
  configs/
    data.yaml            # series, countries, HS codes, lags
    model.yaml           # priors/starts, lag weights, regimes
  data/
    raw/                 # untouched pulls (parquet), dated
    interim/             # cleaned monthly series
    processed/           # model-ready panel + model outputs
    manual/              # hand-curated: headlines, EUA, regimes
  src/ecpmodel/
    __init__.py
    io.py
    ingest/
      ecb.py fred.py comext.py eurostat_sts.py headlines.py eua.py
    features/
      build.py           # unit conversions, lags, as-of panel
    models/
      ecm_ssm.py         # ethylene state-space model
      draws.py           # parameter + state posterior draws
      real.py            # real-price transforms
    eval/
      backtest.py metrics.py
    viz/plots.py
  tests/
    test_units.py test_ssm_recovery.py test_asof.py
  notebooks/
    01_explore_data.ipynb 02_diagnostics.ipynb
  .github/workflows/ci.yml
```

Create it:

```bash
mkdir ecp-model && cd ecp-model
git init
mkdir -p configs data/{raw,interim,processed,manual} notebooks tests .github/workflows
mkdir -p src/ecpmodel/{ingest,features,models,eval,viz}
touch src/ecpmodel/__init__.py
printf 'data/raw/\ndata/interim/\ndata/processed/\n.env\n__pycache__/\n.venv/\n' > .gitignore
```

Keep `data/manual/` in git (small, hand-curated, and it is your only ground truth). Keep raw/interim/processed out of git, or track them with DVC if you want reproducibility of data snapshots.

## 2. Environment

Python 3.11+. Using `uv` (·conda works equally well):

```bash
uv venv && source .venv/bin/activate
```

`pyproject.toml`:

```toml
[project]
name = 'ecpmodel'
version = '0.1.0'
requires-python = '>=3.11'
dependencies = [
  'pandas>=2.1', 'numpy', 'scipy', 'statsmodels>=0.14', 'pyarrow',
  'requests', 'pyyaml', 'eurostat', 'pymc>=5', 'arviz', 'matplotlib',
  'pandera', 'typer', 'tqdm',
]

[project.optional-dependencies]
dev = ['pytest', 'ruff', 'jupyter']

[build-system]
requires = ['setuptools>=68']
build-backend = 'setuptools.build_meta'

[tool.setuptools.packages.find]
where = ['src']
```

```bash
uv pip install -e '.[dev]'
```

`Makefile`:

```make
.PHONY: data model backtest test
data:
	python -m ecpmodel.ingest.run_all
model:
	python -m ecpmodel.models.ecm_ssm
backtest:
	python -m ecpmodel.eval.backtest
test:
	pytest -q
```

## 3. Data inventory and pull plan

| Block | Series | Source | Freq | Publication lag | Role |
| --- | --- | --- | --- | --- | --- |
| Costs | Brent spot (USD/bbl) | FRED `DCOILBRENTEU` (EIA-derived) | daily | \~1 week | naphtha anchor |
| Costs | EU natural gas (USD/MMBtu) | FRED `PNGASEU` (IMF) | monthly | \~1 month | energy cost; free TTF proxy |
| Costs | EUA carbon (EUR/t) | manual CSV (Ember tracker, EEX auction results, or exchange delayed data) | monthly | days | ETS cost |
| Costs | EUR/USD | ECB SDW `EXR` | monthly avg | days | FX conversion |
| Label | ECP settlement headlines | free trade press, producer results/calls | monthly/quarterly | days | noisy ground truth |
| Obs | Ethylene unit values | Comext, HS 290121, DE/NL/BE/AT/PL/CZ/SK/HU reporters | monthly | \~2 months | lagged transacted price |
| Obs | Chemicals PPI | Eurostat `sts_inpp_m`, NACE C20 (try C201) | monthly | \~1 month | smoothed price level; deflator for the real price |
| Co-products | Propylene (290122), butadiene (290124) unit values | Comext | monthly | \~2 months | credit proxy (optional) |
| Naphtha check | CN 271012 unit values | Comext | monthly | \~2 months | sanity check only; heterogeneous |

Notes:

- There is no good free naphtha series. Do not hunt for one: use Brent in EUR/bbl and let the model estimate the effective pass-through coefficient (it absorbs naphtha/crude ratio, yield and co-product credits).
- `PNGASEU` is a monthly IMF series, not TTF itself. It is fine for monthly modelling. If you have a delayed TTF feed, swap it in at `configs/data.yaml`.
- Comext access is the most fragile part (API dimension order and dataset IDs change). Section 4.5 builds it defensively and tells you where to verify.
- Respect licences. For paywalled trade press, record only the data point you are legally allowed to quote, the source name and date, never article text.

**Phased pull plan**

1. *Phase A (day 1):* ECB FX, FRED Brent and gas, manual EUA CSV. You can already run a pure cost-plus baseline.
2. *Phase B (day 2):* Eurostat STS (PPI).
3. *Phase C (days 3-4):* Comext. Start with one reporter (DE) and one product (290121), confirm the unit values look sane against the headline range, then widen.
4. *Phase D (ongoing):* headlines CSV. This is manual and the most valuable hour you spend each month. Backfill at least 3 years.
5. *Phase E:* schedule with cron/GitHub Actions monthly after Comext releases.

`configs/data.yaml`:

```yaml
start: '2015-01'
comext:
  reporters: [DE, NL, BE, AT, PL, CZ, SK, HU]
  partners: [WORLD]          # or explicit partner list
  flows: {import: 1, export: 2}
  products:
    ethylene: '290121'
    propylene: '290122'
    butadiene: '290124'
  min_tonnes: 200            # drop thin cells
fred:
  brent: DCOILBRENTEU
  eu_gas: PNGASEU
availability_lag_days:       # used for vintage-aware backtests
  fx: 5
  brent: 7
  eu_gas: 35
  eua: 5
  headline: 7
  comext: 65
  ppi: 35
```

## 4. Ingestion code

### 4.1 Shared I/O

`src/ecpmodel/io.py`:

```python
from pathlib import Path
import datetime as dt
import pandas as pd
import yaml

RAW = Path('data/raw')
INTERIM = Path('data/interim')
PROCESSED = Path('data/processed')


def load_cfg(name='data'):
    return yaml.safe_load(open(f'configs/{name}.yaml'))


def save_raw(df: pd.DataFrame, name: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    df.to_parquet(RAW / f'{name}__{stamp}.parquet')
    df.to_parquet(RAW / f'{name}__latest.parquet')


def save_interim(s: pd.Series, name: str) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    s.rename(name).to_frame().to_parquet(INTERIM / f'{name}.parquet')
```

Convention: every cleaned series is a monthly `pd.Series` indexed by month-start timestamps (`freq='MS'`). Availability is applied later in `features/build.py`, never in ingestion.

### 4.2 ECB FX

`src/ecpmodel/ingest/ecb.py`:

```python
import io
import pandas as pd
import requests
from ecpmodel.io import save_raw, save_interim

URL = 'https://data-api.ecb.europa.eu/service/data/EXR/M.USD.EUR.SP00.A'


def fetch_eurusd(start='2012-01'):
    r = requests.get(URL, params={'format': 'csvdata', 'startPeriod': start}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    save_raw(df, 'ecb_eurusd')
    s = df.set_index(pd.to_datetime(df['TIME_PERIOD']))['OBS_VALUE'].astype(float)
    s.index = s.index.to_period('M').to_timestamp()
    save_interim(s, 'eurusd')   # USD per 1 EUR
    return s
```

The ECB quotes USD per EUR. Keep that convention everywhere (divide USD prices by it to get EUR).

### 4.3 FRED (Brent, EU gas)

`src/ecpmodel/ingest/fred.py`:

```python
import pandas as pd
from ecpmodel.io import save_raw, save_interim


def fetch_fred(series_id: str) -> pd.Series:
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    df = pd.read_csv(url)
    df.columns = ['date', 'value']
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    save_raw(df, f'fred_{series_id}')
    return df.dropna().set_index('date')['value']


def monthly(s: pd.Series) -> pd.Series:
    return s.resample('MS').mean()


def pull(cfg):
    brent = monthly(fetch_fred(cfg['fred']['brent']))      # USD/bbl
    gas = monthly(fetch_fred(cfg['fred']['eu_gas']))       # USD/MMBtu
    save_interim(brent, 'brent_usd_bbl')
    save_interim(gas, 'gas_usd_mmbtu')
    return brent, gas
```

FRED endpoints occasionally rename the date column (`DATE` vs `observation_date`); assigning columns positionally as above is robust to that.

### 4.4 EUA carbon (manual)

There is no stable free API for EUA prices. Create `data/manual/eua_monthly.csv`:

```csv
date,eua_eur_t,source
2024-01-01,NN.NN,<where you got it>
```

Sources to use: Ember's carbon price tracker download, EEX primary auction results (published free), or delayed exchange data. Use monthly average of daily settles. Loader in `ingest/eua.py`:

```python
import pandas as pd
from ecpmodel.io import save_interim


def pull(path='data/manual/eua_monthly.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    s = df.set_index('date')['eua_eur_t'].astype(float).resample('MS').mean()
    save_interim(s, 'eua_eur_t')
    return s
```

### 4.5 Eurostat Comext (ethylene, propylene, butadiene)

Dataset: Comext 'easy' monthly detailed data, `ds-045409`, via the Eurostat Comext SDMX 2.1 API. **Verify before trusting:** (a) the dimension order in the key (I use frequency.reporter.partner.product.flow.indicator), and (b) the indicator codes. Fetch the dataset's structure definition from the Comext API documentation page on the Eurostat site, or print `df.columns` and `df.head()` from the first call and compare.

Codes used: product `290121` (ethylene), flow `1` (import) and `2` (export), indicators `VALUE_IN_EUROS` and `QUANTITY_IN_100KG`.

`src/ecpmodel/ingest/comext.py`:

```python
import io
import time
import numpy as np
import pandas as pd
import requests
from ecpmodel.io import save_raw, save_interim

BASE = 'https://ec.europa.eu/eurostat/api/comext/dissemination/sdmx/2.1/data/ds-045409'


def fetch_cell(reporter, partner, product, flow, indicator, start):
    key = f'M.{reporter}.{partner}.{product}.{flow}.{indicator}'
    r = requests.get(f'{BASE}/{key}',
                     params={'format': 'SDMX-CSV', 'startPeriod': start},
                     timeout=180)
    if r.status_code == 404:
        return pd.DataFrame()      # no data for this cell
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df['indicator'] = indicator
    return df


def pull_product(cfg, name):
    c = cfg['comext']
    code = c['products'][name]
    frames = []
    for rep in c['reporters']:
        for part in c['partners']:
            for flow_name, flow in c['flows'].items():
                for ind in ['VALUE_IN_EUROS', 'QUANTITY_IN_100KG']:
                    d = fetch_cell(rep, part, code, flow, ind, cfg['start'])
                    if len(d):
                        d['reporter'], d['partner'], d['flow'] = rep, part, flow_name
                        frames.append(d)
                    time.sleep(0.3)            # be polite
    raw = pd.concat(frames, ignore_index=True)
    save_raw(raw, f'comext_{name}')
    return raw


def unit_values(raw: pd.DataFrame, min_tonnes=200) -> pd.DataFrame:
    df = raw.copy()
    df['month'] = pd.to_datetime(df['TIME_PERIOD']).dt.to_period('M').dt.to_timestamp()
    wide = (df.pivot_table(index=['month', 'reporter', 'partner', 'flow'],
                           columns='indicator', values='OBS_VALUE', aggfunc='sum')
              .reset_index())
    wide['tonnes'] = wide['QUANTITY_IN_100KG'] / 10.0       # 100 kg = 0.1 t
    wide = wide[wide['tonnes'] >= min_tonnes].copy()
    wide['uv'] = wide['VALUE_IN_EUROS'] / wide['tonnes']     # EUR/t
    return wide


def aggregate(wide: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        return pd.Series({
            'uv_w': np.average(g['uv'], weights=g['tonnes']),
            'uv_med': g['uv'].median(),
            'tonnes': g['tonnes'].sum(),
            'n_cells': len(g),
        })
    return wide.groupby('month').apply(agg)


def pull(cfg):
    out = {}
    for name in cfg['comext']['products']:
        raw = pull_product(cfg, name)
        agg = aggregate(unit_values(raw, cfg['comext']['min_tonnes']))
        save_interim(agg['uv_w'], f'uv_{name}')
        save_interim(agg['tonnes'], f'tonnes_{name}')
        out[name] = agg
    return out
```

Design choices worth understanding:

- **Thin-cell filter.** Ethylene is mostly pipeline-traded, so many reporter-partner cells are tiny and dominated by mix and rounding. The 200 t floor removes them. Tune it by plotting `uv` against `tonnes`.
- **Volume-weighted unit value** is the model's observation; keep the median as a robustness column.
- **Fallback if the API fights you:** Eurostat also publishes monthly bulk Comext files (7z archives) on its bulk download facility. Add `py7zr` to dependencies, download one month at a time, filter on the product code, and feed the result into the same `unit_values` function. Keep the column names (`reporter`, `partner`, `flow`, `indicator`, `OBS_VALUE`, `TIME_PERIOD`) identical so downstream code is unchanged.
- **Sanity check:** the series should be within a factor of about 1.5 of your headline range and move in the same direction, lagged 1-2 months. If it is 3x off, you have a unit error (100 kg vs t is the usual culprit) or you are pulling the wrong product code.

### 4.6 Eurostat STS (PPI)

Use the `eurostat` package with named filters, which is robust to dimension order.

`src/ecpmodel/ingest/eurostat_sts.py`:

```python
import eurostat
import pandas as pd
from ecpmodel.io import save_raw, save_interim


def _tidy(df: pd.DataFrame) -> pd.Series:
    time_cols = [c for c in df.columns if c[:2].isdigit()]
    s = df[time_cols].iloc[0]
    s.index = pd.to_datetime(s.index, format='%Y-%m')
    return s.astype(float).dropna()


def ppi(geo='DE', nace='C20'):
    df = eurostat.get_data_df('sts_inpp_m', filter_pars={
        'geo': [geo], 'nace_r2': [nace], 'unit': ['I21'],
        's_adj': ['NSA'], 'indic_bt': ['PRC_PRR']})
    save_raw(df, f'sts_ppi_{geo}_{nace}')
    s = _tidy(df)
    save_interim(s, f'ppi_{nace}_{geo}')
    return s
```

Check the filter values against the dataset's own code lists (`eurostat.get_par_values('sts_inpp_m', 'nace_r2')`). Some country/NACE combinations are not published; fall back to C20 or the EU aggregate (`geo='EU27_2020'`).

### 4.7 Headlines (the ground-truth file)

This is the file that determines whether the model is any good. Create `data/manual/ecp_headlines.csv`:

```csv
month,settlement_eur_t,low,high,type,published_on,source,reliability,notes
2024-01-01,NNNN,,,monthly,2024-01-31,<trade press name>,2,<free text>
2024-04-01,NNNN,NNNN,NNNN,quarterly,2024-03-28,<producer call>,1,Q2 contract
```

Field rules:

- `month`: the month the price applies to (the first month for a quarterly contract).
- `published_on`: the date you could first have seen it. Used for vintage-aware backtests. Never backfill this with the month itself.
- `type`: `monthly` or `quarterly`. A quarterly contract becomes three monthly observations in the model, each with an inflated variance (Section 6.2).
- `reliability`: 1 (producer statement), 2 (reputable trade press), 3 (hearsay/secondhand). Maps to an observation variance multiplier.
- `low`/`high`: if only a range is published, put the midpoint in `settlement_eur_t` and leave the range for diagnostics.

Loader in `ingest/headlines.py`:

```python
import pandas as pd
import pandera as pa
from ecpmodel.io import save_raw

schema = pa.DataFrameSchema({
    'month': pa.Column('datetime64[ns]'),
    'settlement_eur_t': pa.Column(float, pa.Check.in_range(200, 4000)),
    'type': pa.Column(str, pa.Check.isin(['monthly', 'quarterly'])),
    'published_on': pa.Column('datetime64[ns]'),
    'reliability': pa.Column(int, pa.Check.isin([1, 2, 3])),
}, coerce=True)


def load(path='data/manual/ecp_headlines.csv') -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['month', 'published_on'])
    df = schema.validate(df)
    save_raw(df, 'headlines_ecp')
    rows = []
    for _, r in df.iterrows():
        n = 3 if r['type'] == 'quarterly' else 1
        for k in range(n):
            rows.append({
                'month': r['month'] + pd.DateOffset(months=k),
                'y': r['settlement_eur_t'],
                'available_on': r['published_on'],
                'rel': r['reliability'],
                'repl': n,
            })
    return pd.DataFrame(rows).sort_values('month')
```

### 4.8 Orchestrator

`src/ecpmodel/ingest/run_all.py`:

```python
from ecpmodel.io import load_cfg
from ecpmodel.ingest import ecb, fred, comext, eurostat_sts, eua


if __name__ == '__main__':
    cfg = load_cfg('data')
    ecb.fetch_eurusd(cfg['start'])
    fred.pull(cfg)
    eua.pull()
    eurostat_sts.ppi('DE', 'C20')
    comext.pull(cfg)
```

Run `make data` and open `notebooks/01_explore_data.ipynb`. Before moving on, plot: (1) Brent in EUR/bbl, (2) ethylene unit value against the headline series, (3) PPI. If the unit value does not track the headlines after a 1-2 month lag, fix the data before modelling.

## 5. Vintage discipline (do this now, not later)

Every series has a publication lag, and Comext is also revised for months after first release. If you backtest on today's fully-revised data you will overstate accuracy.

`src/ecpmodel/features/build.py` (as-of core):

```python
import pandas as pd


def as_of(series: pd.Series, lag_days: int, t0: pd.Timestamp) -> pd.Series:
    # Observation for month m is assumed visible at month_end(m) + lag_days.
    month_end = series.index + pd.offsets.MonthEnd(0)
    visible = month_end + pd.Timedelta(days=lag_days) <= t0
    return series[visible]


def headlines_as_of(h: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    return h[h['available_on'] <= t0]
```

The limitation to be honest about: this models publication lag, not revision. You only have final Comext values unless you snapshot monthly. Start snapshotting now (`save_raw` already writes dated files) so that in a year you have a real vintage archive.

# Part 2: Model, real price, backtest, operations

## 6. The state-space model

### 6.1 Notation

- `t` = month. All prices in k EUR/t inside the model (divide EUR/t by 1000) for numerical conditioning. Convert back for reporting.
- `p_t` = latent ECP (the thing we want).
- `x_t` = regressor vector `[1, brent_eur/100, gas_eur_mwh/100, eua/100]`, each cost series passed through a distributed lag (Section 7).
- `c_t = beta' x_t` = cost anchor (the level the contract price is pulled towards; `beta` is estimated, and it absorbs naphtha/crude ratio, cracker yields, and co-product credits).
- `r_t` = regime indicator (0/1) for known disruption months.

### 6.2 Equations

**State equation (error-correction form):**

```
p_t = p_{t-1} + gamma * (p_{t-1} - c_{t-1}) + theta * (c_t - c_{t-1}) + eta_t
eta_t ~ N(0, sigma_eta^2 * (1 + kappa * r_t))
-1 < gamma < 0,   0 <= theta <= 1,   kappa >= 0
```

- `gamma` is the speed at which the price closes the gap to the cost anchor (monthly). Typical values are small and negative; `gamma` near 0 means margins wander for a long time.
- `theta` is the immediate pass-through of a cost change in the same month.
- `kappa` inflates state noise in disruption months (cracker outages, force majeure, energy crisis), so the filter lets the price move away from the anchor there instead of treating it as noise.

Rewritten as a linear model with time-varying intercept:

```
p_t = (1 + gamma) * p_{t-1} + d_t + eta_t
d_t = -gamma * c_{t-1} + theta * (c_t - c_{t-1})        (known given beta, gamma, theta)
```

**State vector** stacks two lags so lagged observations are linear in the state:

```
s_t = (p_t, p_{t-1}, p_{t-2})'

s_t = T s_{t-1} + [d_t, 0, 0]' + R eta_t

T = | 1+gamma  0  0 |      R = | 1 |
    | 1        0  0 |          | 0 |
    | 0        1  0 |          | 0 |
```

**Observation equations** (each series may be missing in any month; the Kalman filter skips missing entries):

```
(1) headline:  y1_t = p_t                       + e1_t,  Var(e1_t) = sigma_h^2 * w(rel_t) * repl_t
(2) Comext UV: y2_t = a_c + p_{t-1}             + e2_t,  Var(e2_t) = sigma_c^2
(3) PPI (std): y3_t = a_p + b_p*(p_t+p_{t-1}+p_{t-2})/3 + e3_t,  Var(e3_t) = sigma_p^2
```

- `w(rel)` maps reliability to a variance multiplier (1, 2, 4 for reliability 1, 2, 3).
- `repl_t` is 3 for quarterly settlements copied across three months (so a quarterly number carries as much information as one monthly number, not three).
- `a_c` absorbs the systematic gap between Comext unit values (customs value, mix, delivery basis) and contract price.
- Comext lag is 1 month by default (the unit value reflects deliveries priced off the previous month's settlement). If your data says 2 months, extend the state to 4 lags and shift the design row; pick the lag by log-likelihood.
- Observation (1) fixes the scale and level of `p_t`. Without enough headlines the model collapses to 'cost anchor plus smoothed trade data', which is useful but is not an estimate of the contract price. Aim for at least 25-30 headline observations in the estimation window.

All equations are linear Gaussian given the parameter vector `psi = (sigma_eta, sigma_h, sigma_c, sigma_p, gamma, theta, kappa, a_c, a_p, b_p, beta)`, so the exact likelihood comes from the Kalman filter and `psi` is fitted by maximum likelihood.

### 6.3 What to expect and watch

- `a_c` and `beta_0` trade off against each other when headlines are scarce. If the optimiser wanders, fix `a_c = 0` and let `sigma_c` absorb it, or put a tight prior on it in the Bayesian version.
- `gamma` is weakly identified in short samples. Report its standard error and do not interpret half-lives without it.
- Fatter tails than Gaussian are plausible in the 2021-22 window. Check standardised residuals before reaching for Student-t errors.
- PPI is weak evidence about ethylene specifically. Run a model with `b_p` fixed at 0 (effectively dropping it) and compare out-of-sample before keeping it.

## 7. Feature builder

Extend `src/ecpmodel/features/build.py`:

```python
import numpy as np
import pandas as pd
from ecpmodel.io import load_cfg, INTERIM

MMBTU_TO_MWH = 0.29307
REL_MULT = {1: 1.0, 2: 2.0, 3: 4.0}


def _ld(name):
    return pd.read_parquet(INTERIM / f'{name}.parquet').iloc[:, 0]


def dist_lag(s: pd.Series, w) -> pd.Series:
    out = sum(wk * s.shift(k) for k, wk in enumerate(w))
    return out


def build_panel(t0=None, horizon=0, lag_w=(0.3, 0.5, 0.2), headlines=None):
    cfg = load_cfg('data')
    lagd = cfg['availability_lag_days']
    raw = {
        'fx': _ld('eurusd'), 'brent': _ld('brent_usd_bbl'),
        'gas': _ld('gas_usd_mmbtu'), 'eua': _ld('eua_eur_t'),
        'uv': _ld('uv_ethylene'), 'ppi': _ld('ppi_C20_DE'),
    }
    key = {'fx': 'fx', 'brent': 'brent', 'gas': 'eu_gas', 'eua': 'eua',
           'uv': 'comext', 'ppi': 'ppi'}
    if t0 is not None:
        for k, s in raw.items():
            me = s.index + pd.offsets.MonthEnd(0)
            raw[k] = s[me + pd.Timedelta(days=lagd[key[k]]) <= t0]

    last_obs = max(s.index.max() for s in raw.values())
    end = (pd.Timestamp(t0).to_period('M').to_timestamp() if t0 is not None
           else last_obs) + pd.DateOffset(months=horizon)
    idx = pd.date_range(cfg['start'], end, freq='MS')
    df = pd.DataFrame(raw).reindex(idx)

    # unit conversions; costs forward-filled over the not-yet-published tail
    cost = pd.DataFrame({
        'brent_eur': df['brent'] / df['fx'],
        'gas_eur_mwh': df['gas'] / df['fx'] / MMBTU_TO_MWH,
        'eua': df['eua'],
    }).ffill()
    cost_l = cost.apply(lambda c: dist_lag(c, lag_w))
    X = pd.concat([pd.Series(1.0, index=idx, name='const'), cost_l / 100.0], axis=1)

    if headlines is None:
        from ecpmodel.ingest.headlines import load
        headlines = load()
    h = headlines if t0 is None else headlines[headlines['available_on'] <= t0]
    h = h.assign(mult=h['rel'].map(REL_MULT) * h['repl'])
    hm = h.groupby('month').agg(y=('y', 'mean'), mult=('mult', 'min'))

    endog = pd.DataFrame(index=idx, columns=['headline', 'comext', 'ppi'], dtype=float)
    endog['headline'] = (hm['y'] / 1000.0).reindex(idx)
    endog['comext'] = df['uv'] / 1000.0
    ppi = df['ppi']
    endog['ppi'] = (ppi - ppi.mean()) / ppi.std()

    regime = load_regimes(idx)
    hvar = hm['mult'].reindex(idx).fillna(1.0)

    keep = X.dropna().index                     # drops lag warm-up rows
    return {'endog': endog.loc[keep], 'X': X.loc[keep], 'regime': regime.loc[keep],
            'hvar': hvar.loc[keep], 'cost': cost.loc[keep], 'ppi_raw': ppi.loc[keep]}


def load_regimes(idx, path='data/manual/regimes.csv'):
    r = pd.Series(0.0, index=idx)
    try:
        d = pd.read_csv(path, parse_dates=['start', 'end'])
    except FileNotFoundError:
        return r
    for _, row in d.iterrows():
        r[(r.index >= row['start']) & (r.index <= row['end'])] = 1.0
    return r
```

`data/manual/regimes.csv`:

```csv
start,end,label
YYYY-MM-01,YYYY-MM-01,<e.g. cracker outage / energy crisis / Rhine low water>
```

The regime windows are your judgement. Fill them from public news (plant outage announcements, force majeure notices) and keep each one documented in `label`. Resist the urge to tune them to make the fit look good; define them before looking at residuals, then test whether `kappa` is significantly above zero.

Two caveats on this builder: the cost tail is forward-filled when the latest months are not yet published (document it, and prefer a delayed daily feed if you have one); and lag weights `(0.3, 0.5, 0.2)` are a starting point, to be chosen by grid search over a few shapes using out-of-sample likelihood, not in-sample fit.

## 8. The model class

`src/ecpmodel/models/ecm_ssm.py`:

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel

PARAMS = ['s_eta', 's_h', 's_c', 's_p', 'gamma', 'theta', 'kappa', 'a_c', 'a_p', 'b_p']
LOG_IDX = [0, 1, 2, 3, 6]          # positive parameters, optimised on log scale


class ECPModel(MLEModel):
    def __init__(self, endog, X, regime, hvar):
        self.X = np.asarray(X, float)
        self.regime = np.asarray(regime, float)
        self.hvar = np.asarray(hvar, float)
        self.k_beta = self.X.shape[1]
        nobs = len(endog)
        super().__init__(endog, k_states=3, k_posdef=1,
                         initialization='approximate_diffuse')
        self['selection'] = np.array([[1.0], [0.0], [0.0]])
        self['state_intercept'] = np.zeros((3, nobs))
        self['state_cov'] = np.zeros((1, 1, nobs))
        self['obs_cov'] = np.zeros((3, 3, nobs))

    @property
    def param_names(self):
        return PARAMS + [f'beta{i}' for i in range(self.k_beta)]

    def transform_params(self, u):
        p = np.array(u, float)
        p[LOG_IDX] = np.exp(p[LOG_IDX])
        p[4] = -1.0 / (1.0 + np.exp(-p[4]))          # gamma in (-1, 0)
        p[5] = 1.0 / (1.0 + np.exp(-p[5]))           # theta in (0, 1)
        return p

    def untransform_params(self, p):
        u = np.array(p, float)
        u[LOG_IDX] = np.log(u[LOG_IDX])
        u[4] = np.log(-u[4] / (1.0 + u[4]))
        u[5] = np.log(u[5] / (1.0 - u[5]))
        return u

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        s_eta, s_h, s_c, s_p, gamma, theta, kappa, a_c, a_p, b_p = params[:10]
        beta = params[10:]

        c = self.X @ beta
        c_lag = np.r_[c[0], c[:-1]]

        self['design'] = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [b_p / 3.0, b_p / 3.0, b_p / 3.0]])
        self['obs_intercept'] = np.array([[0.0], [a_c], [a_p]])
        self['transition'] = np.array([[1.0 + gamma, 0.0, 0.0],
                                       [1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0]])
        self['state_intercept', 0, :] = -gamma * c_lag + theta * (c - c_lag)
        self['state_cov', 0, 0, :] = s_eta ** 2 * (1.0 + kappa * self.regime)
        self['obs_cov', 0, 0, :] = s_h ** 2 * self.hvar
        self['obs_cov', 1, 1, :] = s_c ** 2
        self['obs_cov', 2, 2, :] = s_p ** 2
```

Points to understand in this class:

- `endog` has NaNs wherever a series is unobserved. statsmodels' Kalman filter handles that natively, which is why sparse headlines are no problem.
- `state_intercept`, `state_cov` and `obs_cov` are time-varying (last axis = time), which is how regime-dependent variance and per-observation headline variance enter.
- Optimisation runs in unconstrained space (`transform_params`), which keeps `gamma`, `theta` and all variances valid.
- Forecasting is done by extending the panel with future months (Section 7's `horizon` argument): endog is NaN there, so the filter predicts, and the smoothed variance in those rows is the forecast variance. Future `X` is the last known cost level unless you supply scenarios.

## 9. Fitting, outputs, diagnostics

Append to `ecm_ssm.py`:

```python
from scipy.stats import norm
from ecpmodel.features.build import build_panel
from ecpmodel.io import PROCESSED


def make_model(panel):
    return ECPModel(panel['endog'], panel['X'], panel['regime'], panel['hvar'])


def start_vector(panel):
    h = panel['endog']['headline'].dropna()
    X = panel['X'].loc[h.index]
    beta0 = np.linalg.lstsq(X.values, h.values, rcond=None)[0]
    base = np.array([0.05, 0.05, 0.05, 0.30, -0.30, 0.30, 1.0, 0.0, 0.0, 0.30])
    return np.r_[base, beta0]


def fit(panel, n_starts=5, seed=0):
    mod = make_model(panel)
    rng = np.random.default_rng(seed)
    s0 = start_vector(panel)
    best = None
    for i in range(n_starts):
        s = s0.copy()
        if i > 0:                                    # jitter the scale and dynamics
            s[[0, 1, 2, 3, 6]] *= np.exp(rng.normal(0, 0.5, 5))
            s[4] = np.clip(s[4] + rng.normal(0, 0.1), -0.9, -0.02)
        try:
            res = mod.fit(start_params=s, method='lbfgs', maxiter=3000, disp=False)
        except Exception:
            continue
        if best is None or res.llf > best.llf:
            best = res
    return mod, best


def summarize(res, panel, probs=(0.10, 0.05)):
    idx = panel['endog'].index
    mu = res.smoothed_state[0] * 1000.0
    sd = np.sqrt(res.smoothed_state_cov[0, 0]) * 1000.0
    out = pd.DataFrame({'ecp_mean': mu, 'ecp_sd': sd}, index=idx)
    for a in probs:
        z = norm.ppf(1 - a)
        out[f'lo{int(100 - 2 * a * 100)}'] = mu - z * sd
        out[f'hi{int(100 - 2 * a * 100)}'] = mu + z * sd
    out['cost_anchor'] = (panel['X'].values @ res.params[10:]) * 1000.0
    return out


if __name__ == '__main__':
    panel = build_panel()
    mod, res = fit(panel)
    print(res.summary())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    summarize(res, panel).to_parquet(PROCESSED / 'ecp_latent.parquet')
```

Columns come out as `lo80/hi80` and `lo90/hi90`. Note that the first 3 smoothed months are burn-in (diffuse start); hide them in plots.

**Diagnostics to run in `notebooks/02_diagnostics.ipynb`:**

1. `res.plot_diagnostics(variable=0)` for the headline residuals, and the same for `variable=1` (Comext). You want roughly unit-variance standardised residuals, no obvious autocorrelation, no trending bias.
2. `res.test_serial_correlation('ljungbox')` per series.
3. Smoothed ECP against the cost anchor: the gap is your implied margin. It should look economically sensible (compressed in downturns, wide in tight markets). If it sits permanently positive or negative by a large amount, the intercept or `a_c` is soaking up a basis difference.
4. Parameter table: `gamma`, `theta`, `kappa`, and `beta` with standard errors. Sanity-check the sign of every coefficient on the cost block before trusting anything else.
5. Likelihood comparison across variants: lag weights, Comext lag 1 vs 2, PPI on/off, regime on/off.

## 10. Posterior draws (parameter and state uncertainty)

Bands from the smoother alone ignore parameter uncertainty. A cheap fix, before moving to full MCMC: draw parameters from the asymptotic normal and run the simulation smoother for each draw.

`src/ecpmodel/models/draws.py`:

```python
import numpy as np


def param_draws(res, n=300, seed=0):
    rng = np.random.default_rng(seed)
    cov = res.cov_params()
    out = []
    while len(out) < n:
        d = rng.multivariate_normal(res.params, cov)
        ok = (d[[0, 1, 2, 3, 6]] > 0).all() and (-0.99 < d[4] < -0.005) and (0 < d[5] < 1)
        if ok:
            out.append(d)
    return np.array(out)


def state_draws(mod, params, seed=0):
    # returns array (n_draws, nobs): sampled paths of p_t in k EUR/t
    paths = []
    for i, par in enumerate(params):
        r = mod.smooth(par)
        sim = r.simulation_smoother(random_state=seed + i)
        sim.simulate()
        paths.append(sim.simulated_state[0].copy())
    return np.array(paths)
```

If `cov_params()` is not positive definite (it happens when `gamma` or `kappa` sit at a boundary), use `res.cov_params(method='robust')`, or draw in unconstrained space via `mod.untransform_params`. Save the draws; they give you the real-price bands in Section 11.

**Full Bayesian option (when you are ready):** wrap the exact Kalman log-likelihood as a black-box op and sample with a gradient-free sampler. This is a skeleton, not tested end-to-end:

```python
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op


def build_pymc(mod, beta_start):
    @as_op(itypes=[pt.dvector], otypes=[pt.dscalar])
    def ssm_ll(theta):
        try:
            return np.array(mod.loglike(theta, transformed=True))
        except Exception:
            return np.array(-1e10)

    with pm.Model() as m:
        sd = pm.HalfNormal('sd', 0.3, shape=4)
        gamma = pm.Uniform('gamma', -0.99, -0.005)
        theta = pm.Beta('theta', 2, 2)
        kappa = pm.HalfNormal('kappa', 2.0)
        a_c = pm.Normal('a_c', 0, 0.1)
        a_p = pm.Normal('a_p', 0, 1)
        b_p = pm.Normal('b_p', 0, 1)
        beta = pm.Normal('beta', beta_start, 1.0, shape=len(beta_start))
        theta_vec = pt.concatenate([sd, pt.stack([gamma, theta, kappa, a_c, a_p, b_p]), beta])
        # NOTE: order must match PARAMS: s_eta,s_h,s_c,s_p,gamma,theta,kappa,a_c,a_p,b_p,beta
        pm.Potential('ll', ssm_ll(theta_vec))
    return m
```

Sample with `pm.DEMetropolisZ` (no gradients available through the black box). A fully differentiable version needs the Kalman recursion written in pytensor `scan`, or the state-space tooling in `pymc-extras`. Check what your installed version offers before investing time. For monthly data with about 10 parameters, the asymptotic-draw approach above is usually good enough.

## 11. Real price

Two definitions; pick based on the question:

```python
def real_price(ecp_eur_t, ppi_c20, base_month):
    # purchasing power: deflate by chemicals PPI rebased to 100 at base_month
    base = ppi_c20.loc[base_month]
    return ecp_eur_t / (ppi_c20 / base)


def naphtha_equiv_ratio(ecp_eur_t, brent_eur_bbl, bbl_per_t=7.3):
    # margin-compression view: ECP relative to crude cost per tonne
    return ecp_eur_t / (brent_eur_bbl * bbl_per_t)
```

The PPI deflator is a general chemicals index and partly endogenous to ethylene itself; for a cleaner deflator use the euro-area manufacturing PPI. The Brent-based ratio is a rough proxy (naphtha trades at a variable premium or discount to crude), so treat it as an indicator, not an economic measure.

**Applying it to the posterior.** The model targets the nominal ECP; the real price is a transform of each state draw, so the real bands carry parameter and state uncertainty. For forecast months the deflator is unknown: hold it at its last value (the default here) or pass a scenario. The bands then ignore deflator uncertainty, which is small next to ethylene's own volatility but not zero.

```python
def real_draws(p_paths, idx, ppi_c20, base_month):
    # p_paths: (n_draws, len(idx)) in k EUR/t; returns real EUR/t draws
    defl = (ppi_c20 / ppi_c20.loc[base_month]).reindex(idx).ffill()
    return p_paths * 1000.0 / defl.values
```

Report real-price quantiles with `np.quantile(real_paths, [0.1, 0.5, 0.9], axis=0)`, and score the backtest in nominal terms (the real price is the nominal one divided by a known deflator, so errors carry over).

## 12. Backtest

### 12.1 Design

- **Rolling origin:** for each month-end `t0` from (start + 36 months) to the latest, build the panel as of `t0` (publication lags enforced), fit, and produce nowcast `h=0` and forecasts `h=1..3`.
- **Truth:** headlines for the evaluation months, but only those with `published_on > t0` (not in the information set).
- **Predictive distribution for scoring headlines:** the headline is `p_t + noise`, so the predictive variance is `Var(p_t) + sigma_h^2 * w(rel)`. Forgetting this makes your intervals look overconfident.
- **Benchmarks:** (a) naive: last visible headline; (b) cost-plus OLS: regress headline on `X` over the visible window, predict with the latest costs; (c) your model without Comext/PPI (headline + costs only). The state-space model only earns its complexity if it beats (b) at h=0 and h=1 and its intervals are calibrated.
- **Stress windows:** report metrics separately for 2021-22 and for the rest.

### 12.2 Metrics

`src/ecpmodel/eval/metrics.py`:

```python
import numpy as np
from scipy.stats import norm


def rmse(y, mu):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(mu)) ** 2)))


def crps_normal(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def coverage(y, mu, sd, level=0.80):
    z = norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(np.asarray(y) - mu) <= z * sd))
```

### 12.3 Loop

`src/ecpmodel/eval/backtest.py`:

```python
import numpy as np
import pandas as pd
from ecpmodel.features.build import build_panel
from ecpmodel.models.ecm_ssm import fit
from ecpmodel.ingest.headlines import load as load_headlines
from ecpmodel.eval import metrics as M


def run(origins, horizon=3):
    heads = load_headlines()
    rows = []
    for t0 in origins:
        t0 = pd.Timestamp(t0)
        panel = build_panel(t0=t0, horizon=horizon, headlines=heads)
        mod, res = fit(panel, n_starts=3)
        p = res.params
        mu = pd.Series(res.smoothed_state[0] * 1000.0, index=panel['endog'].index)
        var_p = pd.Series(res.smoothed_state_cov[0, 0] * 1e6, index=panel['endog'].index)
        s_h = p[1] * 1000.0

        truth = heads[(heads['available_on'] > t0) &
                      (heads['month'] >= t0.to_period('M').to_timestamp()) &
                      (heads['month'] <= t0 + pd.DateOffset(months=horizon))]
        vis = heads[heads['available_on'] <= t0].sort_values('month')
        naive = vis['y'].iloc[-1]

        X = panel['X']
        vh = panel['endog']['headline'].dropna() * 1000.0
        beta = np.linalg.lstsq(X.loc[vh.index].values, vh.values, rcond=None)[0]
        ols_pred = pd.Series(X.values @ beta, index=X.index)

        for _, r in truth.iterrows():
            m = r['month']
            if m not in mu.index:
                continue
            h = (m.year - t0.year) * 12 + (m.month - t0.month)
            sd = np.sqrt(var_p[m] + (s_h ** 2) * REL.get(r['rel'], 2.0) * r['repl'])
            rows.append({'t0': t0, 'month': m, 'h': h, 'y': r['y'],
                         'ssm_mu': mu[m], 'ssm_sd': sd,
                         'naive': naive, 'ols': ols_pred[m]})
    return pd.DataFrame(rows)


REL = {1: 1.0, 2: 2.0, 3: 4.0}


def report(df):
    g = df.groupby('h')
    def f(x):
        return pd.Series({
            'n': len(x),
            'rmse_ssm': M.rmse(x['y'], x['ssm_mu']),
            'rmse_ols': M.rmse(x['y'], x['ols']),
            'rmse_naive': M.rmse(x['y'], x['naive']),
            'crps_ssm': float(np.mean(M.crps_normal(x['y'], x['ssm_mu'], x['ssm_sd']))),
            'cov80': M.coverage(x['y'], x['ssm_mu'], x['ssm_sd'], 0.80),
        })
    return g.apply(f)


if __name__ == '__main__':
    origins = pd.date_range('2019-01-31', '2025-12-31', freq='ME')
    df = run(origins)
    df.to_parquet('data/processed/backtest_raw.parquet')
    print(report(df))
```

The loop refits at every origin, which is slow at 3-5 random starts each. For development, evaluate every third month, and run the full grid overnight. Note the sample size will be small (headlines are sparse), so report `n` next to every number and be sceptical of differences smaller than the headline noise.

**Honest limitation:** you are scoring against noisy headlines, so the RMSE has a floor near `sigma_h`. Better scoring comes from a second independent source: fit using headlines from source A and score against source B (or against producer-reported realised prices in quarterly results, if available).

## 13. Tests

The most valuable test is simulate-and-recover: if the model cannot recover its own parameters from data it generated, no real-data result means anything.

`tests/test_ssm_recovery.py`:

```python
import numpy as np
import pandas as pd
from ecpmodel.models.ecm_ssm import ECPModel, PARAMS


def make_inputs(nobs=240, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2000-01-01', periods=nobs, freq='MS')
    cost = np.cumsum(rng.normal(0, 0.03, (nobs, 3)), axis=0) + 1.0
    X = np.column_stack([np.ones(nobs), cost])
    return idx, X, np.zeros(nobs), np.ones(nobs)


def test_recover_gamma_and_noise():
    idx, X, regime, hvar = make_inputs()
    true = np.r_[0.05, 0.04, 0.05, 0.30, -0.25, 0.40, 1.0, 0.02, 0.0, 0.50,
                 0.10, 0.30, 0.20, 0.10]
    dummy = pd.DataFrame(np.nan, index=idx, columns=['headline', 'comext', 'ppi'])
    sim_model = ECPModel(dummy, X, regime, hvar)
    sim_model.update(true)
    y = sim_model.simulate(true, len(idx), initial_state=np.full(3, 1.2))
    y = pd.DataFrame(np.asarray(y), index=idx, columns=dummy.columns)
    # make headlines sparse (~40% observed) and Comext lagged-complete
    rng = np.random.default_rng(2)
    y.loc[rng.random(len(idx)) > 0.40, 'headline'] = np.nan
    est = ECPModel(y, X, regime, hvar)
    res = est.fit(start_params=true * 0.8 + 0.01, method='lbfgs', maxiter=2000, disp=False)
    p = dict(zip(est.param_names, res.params))
    assert -0.6 < p['gamma'] < -0.02
    assert 0.5 * 0.05 < p['s_eta'] < 2.0 * 0.05


def test_missing_data_does_not_break_filter():
    idx, X, regime, hvar = make_inputs(60)
    y = pd.DataFrame(np.nan, index=idx, columns=['headline', 'comext', 'ppi'])
    y.iloc[::4, 0] = 1.2
    m = ECPModel(y, X, regime, hvar)
    m.update(np.r_[0.05, 0.04, 0.05, 0.30, -0.25, 0.40, 1.0, 0.0, 0.0, 0.5,
                   0.1, 0.3, 0.2, 0.1])
    assert np.isfinite(m.loglike(m.params if hasattr(m, 'params') else np.r_[0.05, 0.04, 0.05, 0.30, -0.25, 0.40, 1.0, 0.0, 0.0, 0.5, 0.1, 0.3, 0.2, 0.1]))
```

If the simulate call rejects `initial_state` in your statsmodels version, simulate the state recursion by hand in NumPy (five lines) and generate observations from the same matrices. Treat tolerances in the first test as loose on purpose: a single simulated sample will not nail `gamma`, but it should not land near zero or at the boundary.

Add `tests/test_units.py` (a 100 kg to tonne conversion check on a hand-computed example; USD/MMBtu to EUR/MWh on a worked number) and `tests/test_asof.py` (a series with a known lag is invisible at `t0` before its publication date and visible after). These two catch the bugs that silently corrupt the whole model.

## 14. Continuous integration and scheduling

`.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e '.[dev]'
      - run: ruff check src tests
      - run: pytest -q
```

The network-dependent ingestion should not run in CI by default. Run it from a separate scheduled workflow (monthly, a few days after the Comext release) that commits nothing but uploads the dated raw snapshots as artifacts, which gradually builds your vintage archive.

## 15. Order of work, and failure modes

**Suggested sequence**

1. Repo skeleton, environment, CI with a trivial test (half a day).
2. Phase A data plus the pure cost-plus OLS baseline. Plot it against headlines (1 day).
3. Headline file backfill, at least 3 years, with `published_on` (1-2 days; the highest-value work).
4. Comext ethylene for one reporter, then all (1-2 days). Check unit values against headlines at lags 0-3.
5. State-space model with headlines plus costs only. Pass the recovery test first.
6. Add Comext, then PPI, one at a time, keeping each only if it helps out-of-sample.
7. Backtest harness, then the stress window analysis.
8. Posterior-draw plumbing (parameter and state draws).
9. Real-price and reporting layer (plots, a monthly markdown summary).

**Failure modes to expect**

- *Headlines too sparse or inconsistent in basis (FD vs delivered, Q vs M).* The model will fit but mean little. Fix the data first.
- *Comext unit values dominated by thin cells.* Raise `min_tonnes`; consider one big reporter pair only.
- *Optimiser stalls or gives boundary `gamma`.* Use multi-start, fix `a_c`, or simplify the observation set.
- *Lag-weight search overfits.* Choose among 3-5 shapes, not a continuous grid, and pick on out-of-sample CRPS.
- *2021-22 break.* The cost anchor decoupled from prices for reasons that were partly supply-driven. Expect `kappa` to be large and compare the rolling coverage; if the 80% band covers well under 80% there, widen via regime variance or a Student-t observation model.
- *Revisions.* Backtests on final data overstate accuracy. Snapshot monthly from day one.
- *Scope.* The output is a reference price for central Europe, not any single buyer's contract. Collars, rebates, indexation formulas and delivery basis are invisible in public data.

**Extensions once the baseline works**

- Time-varying `beta` as a random walk (add it to the state; observation design becomes time-varying).
- Separate Comext observations per reporter country with their own bias terms (a panel of unit values).
- Student-t or mixture observation errors for headline outliers.
- Deflator as an exogenous series with its own forecast uncertainty, so real-price bands also reflect inflation risk.
- Scenario forecasts: pass alternative future paths for Brent, gas and EUA in the `horizon` rows of `X`.

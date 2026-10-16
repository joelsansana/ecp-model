# Data

Every input, every conversion, every file the pipeline writes. If a number looks
wrong, this is the first place to look.

---

## 1. Inputs

`configs/data.yaml` is the single source of truth for which series are pulled and
how long they take to become public.

| block | series | source | cadence | first release | role |
| --- | --- | --- | --- | --- | --- |
| `fred` | `DCOILBRENTEU` Brent, USD/bbl | FRED (EIA) | daily | ~7 days | naphtha anchor |
| `fred` | `PNGASEUUSDM` EU gas, USD/MMBtu | FRED (IMF) | monthly | ~35 days | energy cost |
| `eurostat` | `sts_inpp_m` NACE C20 PPI, I21, PRC_PRR, NSA | Eurostat STS | monthly | ~35 days | smoothed level; real-price deflator |
| `comext` | HS `290121` ethylene unit value | Comext `ds-045409` | monthly | ~65 days | lagged transacted price |
| `comext` | HS `290122` propylene, `290124` butadiene | Comext | monthly | ~65 days | co-product credit proxy (optional) |
| `comext` | CN `271012` naphtha | Comext | monthly | ~65 days | sanity check only; heterogeneous basket |
| `ecb` | `EXR/M.USD.EUR.SP00.A` | ECB SDW | monthly | ~5 days | FX conversion |
| manual | `eua_eur_t` | see `data/manual/README.md` | monthly | ~5 days | ETS cost |
| manual | ECP settlements | trade press / producer statements | monthly or quarterly | ~7 days | **the ground truth** |

### Notes on the awkward ones

* **There is no good free naphtha series.** Do not go looking for one. Brent in
  EUR/bbl is the anchor and the model estimates the effective pass-through
  coefficient, which absorbs the naphtha/crude ratio, cracker yields and
  co-product credits.
* **`PNGASEU` is retired on FRED and returns 404.** The live IMF Europe gas
  series is `PNGASEUUSDM`. If you swap in a delayed TTF feed, change
  `configs/data.yaml` and nothing else.
* **Comext is the fragile endpoint.** The dimension order in the data key
  (`frequency.reporter.partner.product.flow.indicator`) and the indicator codes
  have both changed without notice. `ingest/comext.py` prints nothing about this
  — if a pull starts returning empty frames, print `df.columns` from the first
  call and compare against the dataset structure definition in the Eurostat
  Comext documentation, rather than trying to read the failure out of a warning.
* **FRED's `fredgraph.csv` endpoint needs no API key.** The date column has been
  called several things over the years, so columns are assigned positionally.
* **Ingestion never applies publication lags.** That happens once, in
  `features/build.py`, so the fit and the backtest cannot drift apart.

### Comext specifics

Indicators are `VALUE_IN_EUROS` and `QUANTITY_IN_100KG`. The quantity is in
**100 kg units**, so `tonnes = quantity / 10`. Getting this wrong puts the unit
value at 10x and it looks plausible in a log chart.

The observation is a **volume-weighted unit value** across reporters, with a
median kept alongside for robustness. A thin-cell filter drops cells below
`min_tonnes` (200 by default): ethylene is mostly pipeline-traded, so many
reporter-partner cells are too small to mean anything and are dominated by mix
and rounding.

Responses are cached per cell under `data/raw/_cells/`, so re-running `make data`
does not re-fetch 128 requests. `make data-refresh` forces a refetch.

---

## 2. Unit conversions

Convention throughout: **USD per 1 EUR** is the ECB's. Divide USD prices by it.

```
brent_eur     = brent_usd_bbl   / eurusd
gas_eur_mwh   = gas_usd_mmbtu   / eurusd / 0.29307        # 1 MMBtu = 0.29307 MWh
eua           = eua_eur_t                                   # already EUR
regressors    = cost / 100                                  # scaling only
observations  = headline / 1000,  comext / 1000            # k EUR/t
output        = * 1000                                      # back to EUR/t
```

`tests/test_units.py` pins every one of these against a hand-computed example,
including the 100 kg to tonne conversion and the MMBtu to MWh constant. These
are the two that silently corrupt the whole model.

If the EUR/USD series is missing the code warns and leaves costs in USD, which
means the anchor and `p_t` are not on the same scale — treat the output as
meaningless until you fix it.

---

## 3. Vintage discipline

`features/build.py::as_of` is the whole policy:

> the observation for month `m` is visible at `month_end(m) + lag_days`

Month-*end*, not month-start: a monthly average is a complete month of data.
`tests/test_asof.py` pins the behaviour at the boundary dates.

Two rules that are easy to break and are pinned by tests:

* **Standardise on the visible sample.** The PPI z-score is computed from the
  rows visible at `t0`, not from the full series. The full-sample version is a
  silent look-ahead and passes every eyeball check.
* **Never drop a series by faking it.** A missing EUA file drops the carbon term
  from the anchor; a missing PPI drops the observation equation. Neither becomes
  a column of zeros.

The cost tail beyond the last published month is forward-filled. This is
documented rather than hidden, and is the reason to prefer a delayed daily feed
if you have one.

### What this does not model

**Revisions.** Comext values you pull today are final. This models publication
lag only. `io.save_raw` writes a dated snapshot on every run and
`.github/workflows/monthly-data.yml` archives them as artifacts; start collecting
on day one and in a year you will have a real vintage archive instead of an
approximation.

---

## 4. The as-of panel

`features/build.py::build_panel(t0, horizon, lag_w, headlines)` returns:

| key | what it is |
| --- | --- |
| `endog` | DataFrame with `headline`, `comext`, `ppi`, all in k EUR/t (PPI z-scored); NaN where unobserved |
| `X` | DataFrame `const` + one column per surviving cost series, `/100` |
| `regime` | 0/1 disruption indicator from `data/manual/regimes.csv` |
| `hvar` | headline variance multiplier `w(rel) * repl`, 1.0 where there is no headline |
| `cost` | the cost block in EUR, forward-filled |
| `ppi_raw` | the PPI in index levels |
| `cost_cols` | the regressor columns that survived |
| `horizon` | forecast months appended at the end (NaN observations → the filter forecasts) |

The first rows are dropped while the distributed lag warms up. Pass `t0` and only
visible data is loaded; pass `horizon` and the panel extends past the origin so
the filter produces genuine multi-step forecasts.

Cost regressors that are entirely absent are dropped from `X` rather than
zero-filled, so `beta` re-indexes itself to whatever survives. With no EUA file,
`X` is `['const', 'brent_eur', 'gas_eur_mwh']`.

---

## 5. What gets written

`data/raw/` — gitignored, the vintage archive.

| file | contents |
| --- | --- |
| `<name>__<YYYY-MM-DD>.parquet` | dated snapshot of an untouched pull |
| `<name>__latest.parquet` | the same, for downstream use |
| `_cells/<key>.parquet` | Comext per-cell response cache |

`data/interim/` — gitignored, cleaned monthly series, one parquet per series,
indexed by month start (`freq='MS'`):

| series | unit |
| --- | --- |
| `eurusd` | USD per EUR |
| `brent_usd_bbl` | USD/bbl |
| `gas_usd_mmbtu` | USD/MMBtu |
| `eua_eur_t` | EUR/t |
| `uv_<product>` | EUR/t, volume-weighted unit value |
| `uvmed_<product>` | EUR/t, median across cells (robustness column) |
| `tonnes_<product>` | tonnes behind the unit value |
| `ppi_<NACE>_<GEO>` | index, e.g. `ppi_C20_DE` |

`data/processed/` — gitignored.

### `ecp_latent.parquet` (from `make model`)

| column | meaning |
| --- | --- |
| `ecp_mean` | smoothed latent ECP, EUR/t |
| `ecp_sd` | posterior standard deviation, EUR/t |
| `lo80` / `hi80` | 80% band, EUR/t |
| `lo90` / `hi90` | 90% band, EUR/t |
| `burn_in` | **true** for the first `k_states - 1` months; those are diffuse-start rows and should be hidden in plots and excluded from scores |
| `cost_anchor` | `beta' x_t` in EUR/t |
| `margin` | `ecp_mean - cost_anchor`, EUR/t — the implied cracker margin |
| `ppi` | the PPI index level, for reference |

Bands here are **conditional on the fitted parameters**. Use
`models/draws.py` for bands that include parameter uncertainty.

### `ecp_params.parquet`

One row, columns = parameter names, so a run's estimates can be diffed against
the next run without refitting.

### `backtest_raw.parquet` (from `make backtest`)

| column | meaning |
| --- | --- |
| `t0` | forecast origin (month end) |
| `month` | the target month |
| `h` | horizon in months, 0 = nowcast |
| `y` | the headline settlement, EUR/t |
| `rel`, `repl` | the target's reliability and replication |
| `ssm_mu` | model mean for that month, EUR/t |
| `ssm_sd` | full predictive sd: `sqrt(Var(p_t) + s_h^2 * w(rel) * repl)` |
| `naive` | last visible headline |
| `ols` | cost-plus OLS prediction |
| `ssm_core_mu`, `core_sd` | same model with Comext and PPI switched off |

### `backtest_raw_report.parquet`

Indexed by `slice` (`h=0`, `h=1`, ..., `all`, `stress` = the 2021–22 window,
`calm` = the rest), with columns `n`, `rmse_ssm`, `crps_ssm`, `cov80`,
`bias_ssm`, `rmse_naive`, `rmse_ols`, `rmse_ssm_core`, `crps_core`.

**Always read `n` next to the numbers.** Headlines are sparse, so the sample is
small and differences smaller than the headline noise mean nothing.

### `monthly_summary.md` and `figures/` (from `make report`)

A short markdown nowcast and four charts: latent ECP with bands against the
headlines and the cost anchor, the implied margin, the real price, and the
backtest by benchmark.

---

## 6. Environment variables

All optional; see `.env.example`.

| variable | effect |
| --- | --- |
| `ECP_HEADLINES`, `ECP_EUA`, `ECP_REGIMES` | relocate the three hand-curated files |
| `ECP_OFFLINE=1` | reuse cached raw pulls only, never touch the network; fails loudly if a series has no snapshot |
| `ECP_REFRESH=1` | ignore the Comext per-cell cache and refetch (also `make data-refresh`) |

There are no API keys anywhere in this repo.
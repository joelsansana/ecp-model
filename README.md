# ecp-model — a public-data state-space model for the European ethylene contract price

The European ethylene contract price (ECP) is negotiated bilaterally and
published only by paid services. With public data you cannot observe it. You can,
however, observe several noisy signals driven by it:

| signal | why it helps |
| --- | --- |
| Brent, EU gas, EUA carbon, EUR/USD | the cost anchor |
| sparse free headlines of monthly/quarterly settlements | the only direct evidence of the level |
| Eurostat Comext unit values, HS 290121 | lagged transacted price |
| producer price indices for basic chemicals | heavily smoothed, and weaker evidence than it looks |

This repo estimates a **latent ECP** with a Kalman filter/smoother
(error-correction state-space model), returns a posterior band rather than a
point, converts it to a **real (deflated) price**, and backtests all of it with
vintage discipline — only data that was actually public at the forecast date.

> **Read this before you use any number out of here.** This is a *reference*
> price for central Europe inferred from public proxies. It is not any single
> buyer's contract. Collars, rebates, indexation formulas and delivery basis are
> invisible in public data, and the headline series you feed it is sparse and
> noisy. Report `n` next to every metric.

**Documentation**

| document | what's in it |
| --- | --- |
| this file | what it is, how to run it, and what to be suspicious of |
| [`docs/method.md`](docs/method.md) | the full specification: equations, estimation, identification, diagnostics, how to read the output |
| [`docs/data.md`](docs/data.md) | every series, unit conversion, publication lag, and the schema of every file written |
| [`data/manual/README.md`](data/manual/README.md) | how to curate the three hand-collected files |

---

## Quickstart

```bash
uv venv && source .venv/bin/activate          # or: python -m venv .venv
uv pip install -e '.[dev]'                    # or: pip install -e '.[dev]'

make data        # pull every public series into data/raw + data/interim
make model       # fit the model -> data/processed/ecp_latent.parquet
make report      # plots + data/processed/monthly_summary.md
make backtest    # rolling-origin evaluation vs naive / cost-plus-OLS (slow)
make test        # unit tests, including simulate-and-recover
```

`make` prefers `.venv/bin/python` automatically; override with `make PY=python3`.

**Before `make model` means anything**, replace the placeholder headline file:

```bash
cp data/manual/ecp_headlines.example.csv data/manual/ecp_headlines.csv
# fill in real settlements you are legally allowed to quote, >= 3 years
```

The loader falls back to the `.example.csv` placeholder and warns loudly on every
run. The scheduled GitHub workflow refuses to publish at all until the real file
exists. Everything else in the pipeline runs without it — that is the point of
`docs/data.md`, which tells you exactly what each input contributes.

Then open `notebooks/01_explore_data.ipynb` and check that the Comext unit value
tracks the headline settlements at a 1–2 month lag *before* you trust anything
downstream. A 3x level error means a units mistake (100 kg vs tonnes is the
usual one) or the wrong HS code, and it looks plausible in a chart.

---

## Status

Everything in this repo runs today against live public endpoints: `make data`,
`make model`, `make report`, `make backtest`, `make test`, and every cell of both
notebooks executes without error. Comext, ECB SDW, FRED and Eurostat STS are all
open endpoints; there is nothing to authenticate.

What is *not* done, deliberately: the headline file still holds placeholders, so
the shipped numbers are illustrative. Nothing in `data/` is committed except
`data/manual/`, which is the only ground truth.

---

## Layout

```
configs/        data.yaml (series, availability lags) · model.yaml (obs set, starts, grids)
data/manual/    hand-curated: headlines, EUA, regimes   <-- only ground truth, kept in git
data/raw/       untouched pulls, dated (vintage archive) <-- gitignored
data/interim/   cleaned monthly series                   <-- gitignored
data/processed/ model-ready panel + outputs               <-- gitignored
src/ecpmodel/
  io.py                 paths, config, parquet helpers
  ingest/               ecb, fred, comext, eurostat_sts, headlines, eua, run_all
  features/build.py     units, distributed lags, as-of panel  <-- vintage discipline
  models/ecm_ssm.py     the state-space model, fitting, summary
  models/draws.py       parameter + state posterior draws
  models/real.py        real-price transforms
  eval/backtest.py      rolling origin · eval/metrics.py  CRPS, coverage, interval score
  viz/plots.py          plots and the monthly markdown summary
tests/          test_units, test_asof, test_ssm_recovery, test_pipeline
notebooks/      01_explore_data, 02_diagnostics
tools/          make_notebooks.py  (regenerates the notebooks from readable source)
```

---

## The model in one screen

All prices are in **k EUR/t** inside the model for numerical conditioning; the
code converts back to EUR/t for everything it writes.

```
state      p_t = p_{t-1} + gamma*(p_{t-1} - c_{t-1}) + theta*(c_t - c_{t-1}) + eta_t
           eta_t ~ N(0, sigma_eta^2 * (1 + kappa * r_t))
           -1 < gamma < 0,  0 <= theta <= 1,  kappa >= 0

anchor     c_t = beta' x_t,   x_t = [1, brent_eur/100, gas_eur_mwh/100, eua/100]
                                each cost series through a distributed lag

obs (1)    y_headline_t = p_t                            + e,  Var = s_h^2 * w(rel) * repl
obs (2)    y_comext_t   = a_c + p_{t-1}                  + e,  Var = s_c^2
obs (3)    y_ppi_t      = a_p + b_p * mean(p_t..p_{t-2}) + e,  Var = s_p^2   (off by default)
```

* `gamma` — speed at which the price closes the gap to the cost anchor. Negative
  by construction; weakly identified in short samples, so read its standard
  error before quoting a half-life.
* `theta` — same-month pass-through of a cost change.
* `kappa` — extra state noise in documented disruption months, so the filter lets
  the price move away from the anchor there instead of calling it noise.
* `beta` — absorbs the naphtha/crude ratio, cracker yields and co-product
  credits. It is **not** a physical pass-through coefficient, and there is no
  good free naphtha series to estimate one; the model estimates it.

Only observation (1) pins the level of `p_t`. Below roughly 25 headline
observations the model is a cost anchor plus smoothed trade data — useful, but
**not** an estimate of the contract price.

Everything is linear-Gaussian given the parameter vector, so the exact likelihood
comes from the Kalman filter and parameters are fitted by maximum likelihood over
an unconstrained parameterisation. `docs/method.md` has the derivations, the
identification story and the diagnostics.

### Vintage discipline

`features/build.py` is the only place that applies publication lags: the entry
for month `m` becomes visible at `month_end(m) + lag_days`. Every standardised
column (the PPI z-score) is standardised on the **visible** sample only, because
standardising on the full series is a silent look-ahead. `tests/test_asof.py`
fails if either ever regresses.

**The honest limitation:** this models publication lag, not *revision*. Comext
values you pull today are final. `io.save_raw` writes a dated snapshot on every
run and `.github/workflows/monthly-data.yml` archives them as artifacts — start
that on day one and in a year you will have a real vintage archive instead of an
approximation.

---

## Decisions worth knowing about

Full reasoning in `docs/method.md`; the short version:

* **PPI is off by default, and that is a finding, not a preference.** With `b_p`
  free the filter stops estimating a contract price and starts reporting a
  chemicals PPI: the standardised index ends up moving the state by ~7 k EUR/t
  per unit, `s_p` collapses to ~20 EUR/t, `gamma` is pushed to zero and the cost
  anchor stops mattering. Headline + Comext recovers `gamma = -0.17 (0.05)` and
  `theta = 0.50 (0.10)`; adding PPI destroys both. Re-enable with
  `--obs headline comext ppi` and keep it only if `make backtest` agrees.
* **Optional inputs degrade instead of faking.** No EUA file drops the carbon
  term from the anchor; no PPI drops the observation equation. Neither becomes a
  column of zeros, and `beta` re-indexes itself to whatever regressors survive.
* **No free naphtha series is hunted for.** Brent in EUR/bbl is the anchor and
  the model estimates the effective pass-through.

### Library traps hit while building this

Each of these produced a wrong number rather than an exception, and each is now
either fixed or covered by a test:

* `FRED PNGASEU` is retired and now 404s — the live IMF Europe gas series is
  `PNGASEUUSDM`.
* statsmodels' optimiser works in unconstrained space, but `MLEResults.cov_params()`
  is the inverse information matrix of the **model-space** log-likelihood, the
  same space as `res.params`. Sampling `res.params` with `cov_params()` is
  consistent — but only if `cov_params_model()` has first clipped a slightly
  indefinite numerical covariance back onto the cone.
* Complex-step differentiation passes *complex* parameters through `update()`.
  Casting them to float silently drops the imaginary part and reports standard
  errors that are simply wrong; the system matrices reallocate to a complex
  dtype instead.
* `res.standardized_forecasts_error` is `(k_endog, nobs)` once the single state
  shock is squeezed out — not `(k_endog, 1, nobs)`.
* `SimulationSmoothResults.simulate()` takes no `nsimulations`, and the correct
  entry point is `mod.simulation_smoother(simulator=res)`.
* `pd.DataFrame([named_series, ...])` puts the series *names* in the index, so
  the obvious transpose is backwards.
* `pandas` 3 no longer allows positional indexing on a `Series`, so
  `transform_params`/`untransform_params` coerce their argument to an array
  first.
* Comext per-cell responses are cached under `data/raw/_cells/`, so re-running
  `make data` does not re-fetch 128 requests. `make data --refresh` forces it.

---

## Failure modes to expect

* **Headlines too sparse or inconsistent in basis** (free-on-board vs delivered,
  quarterly vs monthly). The model will fit and mean nothing. Fix the data first.
* **Comext unit values dominated by thin cells.** Raise `min_tonnes` in
  `configs/data.yaml`, or consider a single large reporter.
* **The optimiser parks `gamma` at zero.** That is a *symptom*: the anchor is
  unidentified, not the optimiser broken. Multi-start, drop `a_c`, or remove an
  observation equation — `python -m ecpmodel.models.ecm_ssm --obs headline` is
  a useful diagnostic.
* **Lag-weight search overfits.** Choose among 3–5 shapes, never a continuous
  grid, and pick on out-of-sample CRPS.
* **The 2021–22 break.** The anchor decoupled from prices for reasons that were
  partly supply-driven. Expect a large `kappa`; if 80% coverage there falls well
  below 80%, widen through the regime variance rather than by hand.
* **Revisions.** Backtests on final data overstate accuracy. Snapshot monthly.
* **Scope.** Reference price for central Europe. Not a contract.

---

## Extensions, once the baseline works

* Time-varying `beta` as a random walk (add it to the state; the observation
  design becomes time-varying).
* Separate Comext observations per reporter country, each with its own basis.
* Student-t or mixture observation errors for headline outliers.
* The deflator as an exogenous series with its own forecast uncertainty, so the
  real-price bands also reflect inflation risk.
* Scenario forecasts: pass alternative Brent/gas/EUA paths into the `horizon`
  rows of `X`.

---

## Licence and data use

No licence is asserted over the market data this repo caches; check each source
before redistributing anything:

| source | terms |
| --- | --- |
| ECB SDW (FX reference rates) | free reuse with attribution |
| FRED (Brent, IMF gas) | free; EIA and IMF underlying terms apply |
| Eurostat Comext `ds-045409` | free reuse with attribution |
| Eurostat STS `sts_inpp_m` | free reuse with attribution |
| trade-press settlement headlines | **yours to license** |

For paywalled trade press, record only the data point you are legally allowed to
quote, plus the source name and date — never article text. That rule is why
`data/manual/` is hand-curated and hand-maintained rather than scraped.
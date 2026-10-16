# Method

The specification of what `src/ecpmodel/models/ecm_ssm.py` actually estimates,
why it is written this way, and how to decide what to keep.

---

## 1. Why a state-space model at all

The ECP is not observed. What exists are noisy proxies that respond to it:

* **headlines** — the published settlement itself, sparse, lagged by weeks, and
  quoted with varying reliability;
* **Comext unit values** — actual transactions, but customs value with a mix and
  delivery-basis basis, thin cells, and a publication lag near two months;
* **PPIs** — heavily smoothed, and mostly about the whole chemical complex
  rather than ethylene.

A regression of the headline on costs handles the first two badly: it throws away
the continuous information in the trade data and treats a noisy, sparse series as
if it were the regressor. A Kalman filter keeps all of it, weights each source by
how much it should be trusted *in that month*, and produces a distribution for
the unobserved price instead of a point.

The dynamics are an error-correction model because a contract price does not
wander freely: it is pulled towards a cost anchor, and the interesting question
is how fast and how completely.

---

## 2. Notation and units

`t` indexes months. Inside the model every price is in **k EUR/t** (i.e. EUR
1 250/t is 1.25), which keeps variances and Hessians in a range where the
optimiser and the complex-step differentiation behave. Everything written to
disk is converted back to EUR/t.

| symbol | meaning |
| --- | --- |
| `p_t` | the latent ECP |
| `x_t` | regressors: `[1, brent_eur/100, gas_eur_mwh/100, eua/100]`, each cost series through a distributed lag |
| `c_t` | cost anchor, `c_t = beta' x_t` |
| `r_t` | regime indicator, 1 in documented disruption months |
| `w(rel)` | headline variance multiplier: 1 / 2 / 4 for reliability 1 / 2 / 3 |
| `repl_t` | 3 for a quarterly settlement copied across three months, else 1 |

The `/100` on the cost regressors is a pure scaling convention so that `beta`
values are readable ("EUR 0.74 per 100 EUR/bbl of Brent"), not an economic
statement.

---

## 3. The state equation

```
p_t = p_{t-1} + gamma * (p_{t-1} - c_{t-1}) + theta * (c_t - c_{t-1}) + eta_t
eta_t ~ N(0, sigma_eta^2 * (1 + kappa * r_t))
-1 < gamma < 0,   0 <= theta <= 1,   kappa >= 0
```

Split it into a **level** effect and a **difference** effect:

* `gamma` is the speed of error correction. `gamma = -0.20` closes half the gap
  to the anchor in about three months. `gamma -> 0` means margins wander for a
  long time.
* `theta` is the same-month pass-through of a cost move: what fraction of a cost
  change lands in the price immediately, with the rest left to `gamma`.

`kappa` multiplies the state variance in disruption months. The point is *not*
to model those months better; it is to stop the filter from reading a crack-out
as an error-correction opportunity and dragging the whole path toward the
anchor. The windows come from `data/manual/regimes.csv` and must be defined
before looking at residuals — see §9.

Rewritten with a time-varying intercept, which is what the filter consumes:

```
p_t = (1 + gamma) * p_{t-1} + d_t + eta_t
d_t = -gamma * c_{t-1} + theta * (c_t - c_{t-1})
```

Stacking two lags makes the lagged observations linear in the state:

```
s_t = (p_t, p_{t-1}, p_{t-2})'

s_t = T s_{t-1} + [d_t, 0, 0]' + R eta_t

T = | 1+gamma  0  0 |      R = | 1 |
    | 1        0  0 |          | 0 |
    | 0        1  0 |          | 0 |
```

### Comext lag

`c_t` enters one month lagged because pipeline deliveries in month `t` were
priced off the previous month's settlement. If your own data says the correlation
peaks at lag 2, widen the state (the layout already does this from
`configs/model.yaml: comext_lag`) and shift the design row accordingly. Pick the
lag by likelihood, not by eye.

---

## 4. The observation equations

```
(1) headline:  y1_t = p_t                              + e1_t,  Var = s_h^2 * w(rel_t) * repl_t
(2) Comext UV: y2_t = a_c + p_{t-1}                    + e2_t,  Var = s_c^2
(3) PPI (std): y3_t = a_p + b_p * (p_t+p_{t-1}+p_{t-2})/3 + e3_t,  Var = s_p^2
```

Any series may be missing in any month; the Kalman filter skips missing entries
natively, so sparse headlines are not a special case. The three-month window in
(3) is because a PPI cannot move faster than the index it tracks.

`a_c` absorbs the systematic gap between a customs unit value and a contract
price: delivery basis, contract versus spot mix, freight. It is a basis, not a
forecast.

**`repl` is the subtlety worth internalising.** A quarterly settlement reported
once is copied across three months, and if those three months each carried the
full observation variance the model would believe it had three independent
readings. It has one. So `Var = s_h^2 * w(rel) * 3`.

---

## 5. Identification, and what to do when it fails

This is where most of the practical work is.

**`a_c` versus `beta_0`.** Both are intercepts, of the Comext and the anchor
respectively. With few headlines they trade off freely and the optimiser will
wander. Fix `a_c = 0` and let `s_c` absorb it, or put a tight prior on it in the
Bayesian version.

**`gamma`.** Weakly identified in short samples. It is only pinned down if the
anchor has genuine variation *and* `p_t` is actually pulled toward it. If `gamma`
sits at zero, the anchor is unidentified — that is a finding about your data, not
an optimiser failure. Diagnostics:

* run `python -m ecpmodel.models.ecm_ssm --obs headline` (costs only). If `gamma`
  recovers there, an observation equation is crowding it out;
* plot `s_h`, `s_c`, `s_p`. When one of them collapses toward zero, that series
  has taken over the state (this is exactly the PPI failure in §7);
* check the standard error. Do not quote a half-life without one.

**`theta` versus contemporaneous `beta`.** A same-month pass-through and a
same-month cost coefficient are close to collinear. `theta` absorbs what
`beta` cannot. When `theta` parks at 1 the model has chosen "instant pass-through
plus whatever the anchor level happens to be" — a legitimate but weakly
identified corner.

**Unidentified directions.** When the regime column is all zeros, `kappa` is
exactly unidentified and the information matrix is singular. The code detects
the non-positive-definite covariance, clips it, and warns.

---

## 6. Estimation

Linear-Gaussian given the parameter vector, so the exact log-likelihood is the
Kalman filter's and the fit is maximum likelihood. Optimisation runs in an
unconstrained parameterisation:

* `sigma_eta, s_h, s_c, s_p, kappa` on the log scale;
* `gamma = -sigmoid(u)` maps to (-1, 0);
* `theta = sigmoid(u)` maps to (0, 1);
* intercepts and `beta` unconstrained.

Multi-start (5 by default) with jittered scale and dynamics parameters, keeping
the best finite log-likelihood. Starts that fail are skipped, not fatal.

`start_params` handed to `mod.fit` is in **model** space — statsmodels
untransforms internally. Getting this wrong produces `NaN` parameters and a
`log(negative)` warning rather than an error message that says so.

### Standard errors

`res.cov_params()` is the inverse information matrix of the **model-space**
log-likelihood at the model-space MLE — the same space as `res.params`, even
though the optimiser worked in unconstrained space. It is computed by complex-step
differentiation, which means `update()` can receive complex parameters; the
system matrices must reallocate to a complex dtype when that happens or the
imaginary part is discarded and the standard errors are simply wrong.

---

## 7. Why PPI is off by default

Measured, not assumed. With the PPI switched on and `b_p` free, the filter
reaches a **higher** likelihood by letting the index do the work:

* the standardised index ends up moving the state by ~7 k EUR/t per unit;
* `s_p` collapses to ~20 EUR/t — the PPI is treated as a near-exact observation
  of the price every month;
* `gamma` is pushed to zero and the entire cost anchor stops mattering, because
  the price never has to revert to anything.

What the model is then reporting is a transformed chemicals PPI, not an ethylene
contract price. On this data, headline + Comext recovers `gamma = -0.17 (0.05)`
and `theta = 0.50 (0.10)`; adding PPI collapses both to their boundaries.

Re-enable with `--obs headline comext ppi`, and keep it only if `make backtest`
shows it helps out of sample. In `notebooks/02_diagnostics.ipynb` the variant
table makes this visible in three lines.

---

## 8. Posterior draws

The smoother bands are conditional on the fitted parameters. To include
parameter uncertainty without MCMC:

1. `draws.param_draws` draws from the asymptotic normal in model space, rejects
   draws outside the parameter constraints, and warns loudly if it has to fall
   back to repeating the MLE point.
2. `draws.state_draws` runs the **simulation smoother** for each draw, seeded
   from that draw's smoothed first state.

The result is a `(n_draws, nobs)` array of paths in k EUR/t carrying both state
and parameter uncertainty. For monthly data with about ten parameters this is
usually good enough; full MCMC is only worth it if you have a differentiable
Kalman recursion (pytensor `scan`, or `pymc-extras`), which is a project in
itself.

---

## 9. Real price

Two definitions answering different questions:

* **purchasing power** — ECP deflated by the NACE C20 PPI, rebased to the base
  month. The deflator is a general chemicals index and partly endogenous to
  ethylene itself; the euro-area manufacturing PPI is a cleaner alternative.
* **margin** — ECP divided by crude cost per tonne (`naphtha_equiv_ratio`, or
  `margin_ratio` for the difference). Naphtha trades at a variable premium or
  discount to crude, so this is an indicator, not a measurement.

For forecast months the deflator is unknown and held at its last value. The bands
then ignore deflator uncertainty — small next to ethylene's own volatility, but
not zero, and worth saying out loud.

`models/real.py::real_draws` applies the transform to *draws*, not to a point
estimate, so the real bands carry parameter and state uncertainty. Report them
with `real_bands`, and score the backtest in nominal terms: the real price is the
nominal one divided by a known deflator, so errors simply carry over.

---

## 10. Backtest design

For each month-end `t0` from `start + 36 months`:

1. rebuild the panel **as of** `t0`, with publication lags enforced and every
   standardised column fitted on the visible sample only;
2. refit;
3. score the nowcast (`h=0`) and forecasts `h=1..H` against headlines with
   `published_on > t0` — that is, against targets that were **not** in the
   information set.

The predictive variance for a headline target is
`Var(p_t) + s_h^2 * w(rel) * repl`. Forgetting the second term is the most common
way to produce intervals that look overconfident.

**Benchmarks.** (a) naive: last visible headline; (b) cost-plus OLS on visible
headlines, carried forward along the cost path; (c) this model with Comext and
PPI switched off. The state-space model only earns its complexity if it beats
(b) at `h=0` and `h=1` *and* its intervals are calibrated.

**Metrics.** RMSE has a floor near `s_h`, because you are scoring against noisy
headlines. CRPS and coverage are what tell you whether the distribution is
honest; `eval/metrics.py` also has the Winkler interval score and pinball loss,
which is what a hedger actually cares about.

**The honest limitation.** You are scoring against noisy headlines. Better
scoring comes from a second independent source: fit using headlines from source
A and score against source B, or against producer-reported realised prices in
quarterly results if you can get them legally.

---

## 11. Diagnostics to run

In order, in `notebooks/02_diagnostics.ipynb`:

1. **Parameter table.** Read signs before magnitudes. Every cost coefficient
   should be positive; `gamma` negative; `theta` in [0, 1].
2. **Standardised residuals**, per observation equation: roughly unit variance,
   no autocorrelation, no trending bias. Ljung-Box per series — statsmodels'
   built-in `test_serial_correlation` is univariate-only, so use
   `statsmodels.stats.diagnostic.acorr_ljungbox`.
3. **Smoothed ECP against the cost anchor.** The gap is the implied margin. It
   should compress in downturns and widen in tight markets. If it sits
   permanently far from zero, an intercept is soaking up a basis difference.
4. **Variant comparison:** lag weights, Comext lag, PPI on/off, regime on/off.
   Likelihood only *proposes*; out-of-sample CRPS decides.
5. **Coverage by horizon**, and separately over 2021–22.

On regime windows: define them from public news — plant outages, force majeure,
the energy crisis — **before** looking at residuals, and document each in the
`label` column. Then test whether `kappa` is significantly above zero. Resisting
the urge to tune them to make the fit look good is the entire point.

---

## 12. Limitations

* You are estimating a **reference** price for central Europe from proxies. It is
  not any single buyer's contract. Collars, rebates, indexation formulas and
  delivery basis are invisible in public data.
* The headline series is sparse and its reliability is a judgement call, encoded
  as a variance multiplier.
* Comext unit values are transacted prices for a *mixture* of contract terms,
  not contract prices.
* The as-of filter models publication lag, not revision. Comext values pulled
  today are final. Until you have a vintage archive, the backtest still flatters
  the model slightly.
* The anchor is identified only when costs actually vary. A flat-cost decade
  gives you `gamma = 0` and a level that is pure diffusion.
* Monthly frequency. Nothing here supports anything intra-month.
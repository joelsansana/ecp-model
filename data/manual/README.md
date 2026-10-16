# data/manual

This directory is **kept in git on purpose**. It is the only ground truth in the
repo: everything else can be re-pulled from a public endpoint, but the headline
settlement file cannot.

## `ecp_headlines.csv`

The file that decides whether this model means anything.

| column | meaning |
| --- | --- |
| `month` | the month the price applies to (first month for a quarterly deal) |
| `settlement_eur_t` | EUR per tonne |
| `low` / `high` | optional published range, kept for interval diagnostics |
| `type` | `monthly` or `quarterly` |
| `published_on` | the date you could **first** have seen it |
| `source` | who said it |
| `reliability` | 1 producer statement, 2 reputable trade press, 3 hearsay |
| `notes` | anything a later reader will need |

Rules:

* `published_on` is what makes the backtest honest. Never backfill it with the
  month itself. A Q2 price announced in late March has `month = 2024-04-01` and
  `published_on = 2024-03-28`.
* A quarterly contract becomes three monthly observations, each carrying the
  **full** information (`repl = 3`, so the observation variance is multiplied by
  3). It must not count as three independent data points.
* Record only the data point you are legally allowed to quote, plus the source
  name and date. Never paste article text.
* Aim for **at least 25–30 observations** in the estimation window. Below that
  the model degenerates into "cost anchor plus smoothed trade data", which is
  not an estimate of the contract price.
* Keep the basis consistent: free-on-board vs delivered, quarterly vs monthly.
  Mixing them produces a fitted model that means nothing.

`ecp_headlines.example.csv` is an illustrative placeholder so the repo runs out
of the box. The loader warns loudly when it falls back to it, and the scheduled
GitHub workflow refuses to publish until the real file exists.

## `eua_monthly.csv`

Monthly average EUA settle in EUR/t, from a source you may redistribute (Ember's
carbon price tracker download, EEX primary auction results, or a delayed exchange
feed). If it is absent the carbon term is dropped from the cost anchor and the
model says so.

## `regimes.csv`

Known disruption windows: cracker outages, force majeure, energy crises, Rhine
low water. **Define these from public news before looking at residuals**, and
document each one in `label`. Then test whether `kappa` is significantly above
zero. Do not tune the windows to make the fit look good.
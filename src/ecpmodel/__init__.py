"""Public-data state-space model of the European ethylene contract price (ECP).

The contract price is bilateral and published only by paid services. This package
estimates it as a latent state from noisy public proxies - cost drivers, sparse
settlement headlines, Comext unit values and (optionally) a chemicals PPI - and
backtests the result with vintage discipline.

Start points:

* :mod:`ecpmodel.features.build`  - the as-of panel; the only place publication
  lags are applied.
* :mod:`ecpmodel.models.ecm_ssm`  - the Kalman filter model, fitting and summary.
* :mod:`ecpmodel.eval.backtest`   - rolling-origin evaluation.

Command line: ``make data`` / ``make model`` / ``make backtest`` / ``make report``.
"""

__version__ = '0.1.0'

__all__ = ['__version__']

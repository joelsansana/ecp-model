"""Ingestion: public endpoints in, dated raw snapshots and clean monthly series out.

One module per source (``ecb``, ``fred``, ``comext``, ``eurostat_sts``) plus two
for the hand-curated files (``headlines``, ``eua``) and ``run_all`` to orchestrate
them. None of these apply publication lags - that happens once, in
:mod:`ecpmodel.features.build`, so the fit and the backtest cannot disagree about
what was public.
"""

"""End-to-end plumbing on a small synthetic panel.

No network, no real data: the point is that the pieces fit together - panel ->
model -> summary -> report -> real price - and that the output columns mean what
the documentation says they mean.
"""

import numpy as np
import pandas as pd
import pytest

from ecpmodel.eval import metrics as M
from ecpmodel.eval.backtest import _h_month, _ols_predict, report
from ecpmodel.features.build import REL_MULT
from ecpmodel.models.ecm_ssm import fit, headline_observation, make_model, summarize


def make_panel(nobs=200, seed=3, headline_share=0.7):
    """A panel shaped exactly like the real thing, driven by known parameters."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2018-01-01', periods=nobs, freq='MS')

    brent = 60 + 15 * np.sin(np.arange(nobs) / 11) + rng.normal(0, 2, nobs)
    gas = 30 + 6 * np.cos(np.arange(nobs) / 17) + rng.normal(0, 1, nobs)
    X = pd.DataFrame({'const': 1.0, 'brent_eur': brent / 100, 'gas_eur_mwh': gas / 100},
                     index=idx)

    beta = np.array([0.35, 0.80, 0.25])
    c = X.to_numpy() @ beta
    gamma, theta, s_eta = -0.20, 0.40, 0.03

    p = np.empty(nobs)
    p[0] = c[0] + 0.05
    for t in range(1, nobs):
        p[t] = p[t - 1] + gamma * (p[t - 1] - c[t - 1]) + theta * (c[t] - c[t - 1]) \
            + rng.normal(0, s_eta)

    endog = pd.DataFrame(index=idx, columns=['headline', 'comext', 'ppi'], dtype=float)
    endog['headline'] = p + rng.normal(0, 0.03, nobs)
    mask = rng.random(nobs) > headline_share
    endog.loc[mask, 'headline'] = np.nan
    endog['comext'] = np.r_[np.nan, p[:-1]] + 0.06 + rng.normal(0, 0.04, nobs)
    ppi = pd.Series(rng.normal(0, 1, nobs) + 2 * p, index=idx)      # standardised
    endog['ppi'] = (ppi - ppi.mean()) / ppi.std(ddof=0)

    regime = pd.Series(0.0, index=idx)
    hvar = pd.Series(1.0, index=idx)
    return {'endog': endog, 'X': X, 'regime': regime, 'hvar': hvar,
            'cost': X * 100.0, 'ppi_raw': p, 'cost_cols': ['brent_eur', 'gas_eur_mwh'],
            'horizon': 0}, pd.Series(p, index=idx)


def test_fit_summarize_round_trip():
    panel, truth = make_panel()
    mod, res = fit(panel, n_starts=3, obs=['headline', 'comext'])
    out = summarize(res, panel)

    idx = panel['endog'].index
    assert list(out.index) == list(idx)
    for col in ('ecp_mean', 'ecp_sd', 'lo80', 'hi80', 'lo90', 'hi90', 'cost_anchor',
                'margin', 'burn_in'):
        assert col in out.columns

    # bands are ordered and nested: lo90 <= lo80 <= mean <= hi80 <= hi90
    assert (out['lo90'] <= out['lo80']).all()
    assert (out['lo80'] < out['ecp_mean']).all()
    assert (out['ecp_mean'] < out['hi80']).all()
    assert (out['hi80'] <= out['hi90']).all()

    live = out[~out['burn_in']]
    corr = np.corrcoef(live['ecp_mean'], truth.reindex(live.index) * 1000.0)[0, 1]
    assert corr > 0.9, corr

    # the anchor is exactly X @ beta, in EUR/t
    L = res.model.layout
    beta = np.asarray(res.params)[[L.idx[f'beta{i}'] for i in range(3)]]
    np.testing.assert_allclose(out['cost_anchor'],
                               panel['X'].to_numpy() @ beta * 1000.0)
    np.testing.assert_allclose(out['margin'], out['ecp_mean'] - out['cost_anchor'])

    # burn-in rows are flagged, not silently shown
    assert out['burn_in'].iloc[:res.model.layout.k_states - 1].all()
    assert not out['burn_in'].iloc[res.model.layout.k_states - 1:].any()


def test_summary_bands_cover_the_truth_at_roughly_the_nominal_rate():
    panel, truth = make_panel()
    _, res = fit(panel, n_starts=3, obs=['headline', 'comext'])
    out = summarize(res, panel)
    live = out[~out['burn_in']]
    y = truth.reindex(live.index) * 1000.0        # truth is in k EUR/t

    # smoother bands are conditional on the parameters and are a little
    # optimistic; anything below ~65% at the nominal 80% is a red flag
    assert M.coverage(y, live['ecp_mean'], live['ecp_sd'], 0.80) > 0.65
    assert M.coverage(y, live['ecp_mean'], live['ecp_sd'], 0.90) > 0.75


def test_headline_observation_sd_scales_with_reliability_and_replication():
    panel, _ = make_panel()
    _, res = fit(panel, n_starts=2, obs=['headline'])
    s = headline_observation(res, rel=2, repl=3)
    assert s == pytest.approx(headline_observation(res, rel=1, repl=1)
                              * np.sqrt(REL_MULT[2] * 3))
    # a monthly producer statement is the most informative shape available
    assert headline_observation(res, rel=1, repl=1) < headline_observation(res, rel=3, repl=1)


def test_ols_benchmark_predicts_along_the_cost_path():
    panel, _ = make_panel()
    pred = _ols_predict(panel)
    assert len(pred) == len(panel['X'])
    assert pred.notna().all()
    # the benchmark is in EUR/t, so it regresses the *de-kiloed* headlines
    vh = panel['endog']['headline'].dropna() * 1000.0
    Xh = panel['X'].loc[vh.index]
    beta = np.linalg.lstsq(Xh.to_numpy(float), vh.to_numpy(float), rcond=None)[0]
    np.testing.assert_allclose(pred.loc[vh.index], Xh.to_numpy(float) @ beta)


def test_report_layout_and_benchmark_columns():
    df = pd.DataFrame({
        't0': pd.to_datetime(['2024-01-31'] * 6),
        'month': pd.to_datetime(['2024-02-01', '2024-02-01', '2024-03-01',
                                 '2024-03-01', '2021-05-01', '2021-06-01']),
        'h': [0, 1, 0, 1, 0, 1],
        'y': [1000.0, 1010.0, 1020.0, 1030.0, 900.0, 910.0],
        'ssm_mu': [995.0, 1005.0, 1015.0, 1025.0, 905.0, 915.0],
        'ssm_sd': [20.0] * 6,
        'naive': [990.0] * 6,
        'ols': [985.0] * 6,
        'ssm_core_mu': [994.0] * 6,
        'core_sd': [21.0] * 6,
    })
    rep = report(df, stress_window=['2021-01-01', '2021-12-31'])

    assert rep.index.name == 'slice'
    assert set(rep.index) == {'h=0', 'h=1', 'all', 'stress', 'calm'}
    for col in ('n', 'rmse_ssm', 'crps_ssm', 'cov80', 'bias_ssm', 'rmse_naive',
                'rmse_ols', 'rmse_ssm_core', 'crps_core'):
        assert col in rep.columns
    assert rep.loc['stress', 'n'] == 2
    assert rep.loc['calm', 'n'] == 4
    # errors of 5 EUR/t
    assert rep.loc['all', 'rmse_ssm'] == pytest.approx(5.0)


def test_metrics_match_closed_forms():
    y = np.array([1000.0, 1000.0, 1000.0])
    mu = np.array([1000.0, 1010.0, 990.0])
    sd = np.array([10.0, 10.0, 10.0])
    assert M.rmse(y, mu) == pytest.approx(np.sqrt(200 / 3))
    assert M.mae(y, mu) == pytest.approx(20 / 3)
    assert M.bias(y, mu) == pytest.approx(0.0)

    # CRPS(N(mu, s), mu) = s * (2*phi(0) - 1/sqrt(pi))
    from scipy.stats import norm as _norm
    perfect = 10 * (2 * _norm.pdf(0) - 1 / np.sqrt(np.pi))
    assert M.crps_normal([1000.0], [1000.0], [10.0])[0] == pytest.approx(perfect, rel=1e-9)
    assert M.coverage(y, mu, sd, 0.80) == pytest.approx(1.0)      # all within +-12.8
    assert M.coverage([1000.0, 1300.0], [1000.0, 1000.0], [10.0, 10.0], 0.80) == \
        pytest.approx(0.5)
    assert M.coverage([1000.0], [1000.0], [1e9], 0.80) == pytest.approx(1.0)
    # the Winkler score penalises width
    assert M.interval_score([1000.0], [990.0], [1010.0])[0] == pytest.approx(20.0)
    # 10 EUR/t below the lower bound costs 2/width * (lo - y) = 1.0
    assert M.interval_score([980.0], [990.0], [1010.0])[0] == pytest.approx(21.0)
    assert M.interval_score([1020.0], [990.0], [1010.0])[0] == pytest.approx(21.0)
    # pinball loss at the median is half the absolute error
    assert M.pinball([1000.0], [990.0], 0.5)[0] == pytest.approx(5.0)


def test_metrics_ignore_nan_pairs():
    y = [1.0, np.nan, 3.0]
    mu = [1.0, 2.0, 3.0]
    assert M.rmse(y, mu) == pytest.approx(0.0)
    assert M.coverage([1.0, np.nan], [1.0, 5.0], [1.0, 1.0], 0.8) == pytest.approx(1.0)
    # a non-positive predictive sd is dropped rather than crashing or dividing
    assert np.isnan(M.coverage([1.0], [1.0], [0.0], 0.8))
    assert np.isnan(M.rmse([], []))


def test_h_month_counts_calendar_months():
    assert _h_month(pd.Timestamp('2024-01-31'), pd.Timestamp('2024-02-01')) == 1
    assert _h_month(pd.Timestamp('2024-12-31'), pd.Timestamp('2025-02-01')) == 2
    assert _h_month(pd.Timestamp('2024-03-31'), pd.Timestamp('2024-03-01')) == 0


def test_model_layout_shrinks_with_the_observation_set():
    panel, _ = make_panel()
    full = make_model(panel, obs=['headline', 'comext', 'ppi'])
    core = make_model(panel, obs=['headline'])
    assert len(full.param_names) == len(core.param_names) + 5
    assert core.k_states == 1
    assert full.k_states == 3

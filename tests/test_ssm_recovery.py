"""Simulate-and-recover.

If the model cannot recover its own parameters from data it generated, no
real-data result means anything.  Tolerances here are loose on purpose - a
single simulated sample will not nail ``gamma``, but it must not land near zero
or at a boundary.
"""

import numpy as np
import pandas as pd

from ecpmodel.models.draws import param_draws
from ecpmodel.models.ecm_ssm import ECPModel, make_layout

NAMES = ['headline', 'comext', 'ppi']


def make_inputs(nobs=300, seed=1, k_beta=4, rho=0.9, sd=0.12):
    """Cost regressors as mean-reverting AR(1) around 1.

    Near-unit-root regressors would be collinear with the level of ``p_t`` and
    the individual ``beta`` coefficients would not be identified - the model
    would fit the *path* of the anchor perfectly and split it arbitrarily
    across the cost series.  ``sd`` is the innovation standard deviation; keep
    it large enough that the cost block moves the price by several sigma of the
    state noise, otherwise the recovery test is testing nothing.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2000-01-01', periods=nobs, freq='MS')
    cost = np.ones((nobs, k_beta - 1))
    for j in range(k_beta - 1):
        for t in range(1, nobs):
            cost[t, j] = 1.0 + rho * (cost[t - 1, j] - 1.0) + rng.normal(0, sd)
    X = np.column_stack([np.ones(nobs), cost])
    regime = np.zeros(nobs)
    hvar = np.ones(nobs)
    return idx, X, regime, hvar


def true_params(k_beta=4, sigma_eta=0.05, gamma=-0.25, theta=0.40, kappa=1.0,
                s_h=0.04, s_c=0.05, s_p=0.30, a_c=0.02, a_p=0.0, b_p=0.50):
    beta = np.array([0.10, 0.30, 0.20, 0.10])
    beta = np.resize(beta, k_beta) if k_beta <= 4 else np.r_[beta, 0.1 * (k_beta - 4)]
    return np.array([sigma_eta, s_h, s_c, s_p, gamma, theta, kappa, a_c, a_p, b_p, *beta])


def _simulate(true, idx, X, regime, hvar, seed=1, headline_p=0.6):
    dummy = pd.DataFrame(np.nan, index=idx, columns=NAMES)
    sim_model = ECPModel(dummy, X, regime, hvar)
    sim_model.update(true)
    y = np.asarray(sim_model.simulate(true, len(idx),
                                      initial_state=np.full(sim_model.k_states, 1.2),
                                      anchor='start', rng=np.random.default_rng(seed)))
    y = pd.DataFrame(y, index=idx, columns=NAMES)

    rng = np.random.default_rng(seed + 1)
    mask = rng.random(len(idx)) > headline_p          # make headlines sparse
    y.loc[mask, 'headline'] = np.nan
    y.loc[rng.random(len(idx)) > 0.9, 'ppi'] = np.nan  # and PPI sparse too
    return y


def test_simulate_and_recover():
    idx, X, regime, hvar = make_inputs()
    true = true_params()
    y = _simulate(true, idx, X, regime, hvar)

    est = ECPModel(y, X, regime, hvar)
    res = est.fit(start_params=true * 0.8 + 0.01, method='lbfgs', maxiter=2000, disp=False)
    p = dict(zip(est.param_names, np.asarray(res.params), strict=True))

    assert np.isfinite(res.llf)
    # loose on purpose: one sample will not nail gamma, but it must not sit at
    # zero (no error correction) nor at the boundary (instant adjustment)
    assert -0.6 < p['gamma'] < -0.05, p['gamma']
    assert 0.5 * 0.05 < p['s_eta'] < 2.0 * 0.05, p['s_eta']
    assert 0.5 * 0.04 < p['s_h'] < 2.0 * 0.04, p['s_h']
    assert 0.05 < p['theta'] < 0.95, p['theta']
    assert all(p[f'beta{i}'] > 0 for i in range(1, X.shape[1])), p

    # standard errors must be computable: exercises the complex-step path
    cov = np.asarray(res.cov_params())
    assert np.isfinite(cov).all()
    assert np.isfinite(np.sqrt(np.clip(np.diag(cov), 0, None))).all()


def test_system_matrices_match_the_documented_equations():
    """design/transition/intercept are the ECM written out by hand."""
    idx, X, regime, hvar = make_inputs(nobs=30, k_beta=3)
    y = pd.DataFrame(np.zeros((len(idx), 3)), index=idx, columns=NAMES)
    mod = ECPModel(y, X, regime, hvar)
    p = np.array([0.05, 0.04, 0.05, 0.30, -0.25, 0.40, 1.0, 0.02, 0.0, 0.50, 1.0, 2.0, 3.0])
    mod.update(p)

    beta = p[10:]
    c = X @ beta
    c_lag = np.r_[c[0], c[:-1]]

    assert mod.layout.k_states == 3
    np.testing.assert_allclose(mod['transition'],
                               np.array([[0.75, 0, 0], [1, 0, 0], [0, 1, 0]]))
    # d_t = -gamma*c_{t-1} + theta*(c_t - c_{t-1})
    np.testing.assert_allclose(mod['state_intercept'][0],
                               -p[4] * c_lag + p[5] * (c - c_lag))
    # headline -> p_t ; comext -> p_{t-1} ; ppi -> mean(p_t, p_{t-1}, p_{t-2})
    np.testing.assert_allclose(mod['design'],
                               np.array([[1, 0, 0], [0, 1, 0], [0.5 / 3, 0.5 / 3, 0.5 / 3]]))
    np.testing.assert_allclose(mod['obs_intercept'].ravel(), [0.0, 0.02, 0.0])
    # regime-dependent state variance
    np.testing.assert_allclose(mod['state_cov'][0, 0], 0.05 ** 2)
    # headline variance scales with the reliability/replication multiplier
    y2 = y.copy()
    hvar2 = np.full(len(idx), 6.0)          # reliability 2 x quarterly replication 3
    mod2 = ECPModel(y2, X, regime, hvar2)
    mod2.update(p)
    np.testing.assert_allclose(mod2['obs_cov'][0, 0], 0.04 ** 2 * 6.0)


def test_missing_data_does_not_break_filter():
    idx, X, regime, hvar = make_inputs(nobs=60)
    y = pd.DataFrame(np.nan, index=idx, columns=NAMES)
    y.iloc[::4, 0] = 1.2                    # a headline every 4 months
    m = ECPModel(y, X, regime, hvar)
    p = true_params()
    llf = m.loglike(p)
    assert np.isfinite(llf)
    res = m.smooth(p)
    assert np.isfinite(np.asarray(res.smoothed_state)).all()


def test_transform_roundtrip_and_bounds():
    idx, X, regime, hvar = make_inputs(nobs=20, k_beta=3)
    y = pd.DataFrame(np.zeros((len(idx), 3)), index=idx, columns=NAMES)
    mod = ECPModel(y, X, regime, hvar)

    raw = np.array([0.05, 0.04, 0.05, 0.30, -0.25, 0.40, 1.0, 0.02, 0.0, 0.50, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(mod.transform_params(mod.untransform_params(raw)), raw)

    for u in np.linspace(-4, 4, 41):
        q = mod.transform_params(np.full(len(mod.param_names), u))
        assert -1.0 < q[4] < 0.0
        assert 0.0 < q[5] < 1.0
        assert (q[[0, 1, 2, 3, 6]] > 0).all()


def test_optional_observations_change_the_layout():
    lay = make_layout(('headline',), k_beta=4)
    assert 's_c' not in lay.idx and 'a_c' not in lay.idx and 's_p' not in lay.idx
    assert lay.k_states == 1
    assert len(lay) == 5 + 4          # s_eta, s_h, gamma, theta, kappa + 4 betas

    lay3 = make_layout(('headline', 'comext', 'ppi'), k_beta=4)
    assert lay3.k_states == 3
    assert len(lay3) == 10 + 4


def test_param_draws_respect_parameter_constraints():
    idx, X, regime, hvar = make_inputs(nobs=300)
    y = _simulate(true_params(), idx, X, regime, hvar)
    est = ECPModel(y, X, regime, hvar)
    res = est.fit(start_params=true_params() * 0.9 + 0.01, method='lbfgs', maxiter=1000,
                  disp=False)

    d = param_draws(res, n=60, seed=0)
    assert d.shape == (60, len(est.param_names))
    assert (d[:, est.layout.log_idx] > 0).all()
    assert ((d[:, est.layout.gamma_idx] > -1) & (d[:, est.layout.gamma_idx] < 0)).all()
    assert ((d[:, est.layout.theta_idx] > 0) & (d[:, est.layout.theta_idx] < 1)).all()


def test_state_draws_have_the_right_shape_and_track_the_data():
    from ecpmodel.models.draws import state_draws

    idx, X, regime, hvar = make_inputs(nobs=120)
    true = true_params()
    y = _simulate(true, idx, X, regime, hvar)
    est = ECPModel(y, X, regime, hvar)
    res = est.fit(start_params=true, method='lbfgs', maxiter=500, disp=False)

    params = np.tile(np.asarray(res.params), (15, 1))
    paths = state_draws(est, params, seed=0, results=res)
    assert paths.shape == (15, len(idx))
    smoothed = np.asarray(res.smoothed_state)[0]
    # every simulated path must sit in the same region as the smoother estimate
    assert np.all(paths.mean(axis=0) > smoothed - 3.0)
    assert np.all(paths.mean(axis=0) < smoothed + 3.0)

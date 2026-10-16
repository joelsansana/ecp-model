"""Vintage discipline: only data that was public at the forecast date.

These tests are what stop the backtest from quietly cheating.  If they fail,
the reported accuracy is fiction.
"""

import numpy as np
import pandas as pd
import pytest

from ecpmodel.features.build import as_of, build_panel, headlines_as_of


def _series():
    idx = pd.date_range('2024-01-01', periods=6, freq='MS')
    return pd.Series(np.arange(1.0, 7.0), index=idx)


def test_as_of_hides_unpublished_months():
    s = _series()
    # Comext-style lag of 65 days: the Jan-2024 observation (month end 31 Jan)
    # only becomes visible on 5 Apr 2024.
    assert list(as_of(s, 65, pd.Timestamp('2024-04-04')).index) == []
    assert list(as_of(s, 65, pd.Timestamp('2024-04-05')).index) == [pd.Timestamp('2024-01-01')]
    out = as_of(s, 65, pd.Timestamp('2024-06-30'))
    assert list(out.index) == list(pd.date_range('2024-01-01', periods=3, freq='MS'))


def test_as_of_uses_month_end_not_month_start():
    s = _series()
    # 5 days after January's month end is 5 Feb; 5 days after the *start* would
    # be 6 Jan.  With a 5-day lag the row is visible from 5 Feb onwards.
    assert list(as_of(s, 5, pd.Timestamp('2024-02-04')).index) == []
    assert list(as_of(s, 5, pd.Timestamp('2024-02-05')).index) == [pd.Timestamp('2024-01-01')]


def test_as_of_on_empty_series_is_a_noop():
    s = pd.Series(dtype=float)
    assert len(as_of(s, 65, pd.Timestamp('2024-02-15'))) == 0


def test_headlines_as_of_uses_publication_date_not_month():
    h = pd.DataFrame({
        'month': [pd.Timestamp('2024-04-01')],
        'y': [1000.0],
        'available_on': [pd.Timestamp('2024-03-28')],   # Q2 price, published in March
        'rel': [1],
        'repl': [3],
    })
    assert len(headlines_as_of(h, pd.Timestamp('2024-03-27'))) == 0
    assert len(headlines_as_of(h, pd.Timestamp('2024-03-28'))) == 1
    # it stays visible later
    assert len(headlines_as_of(h, pd.Timestamp('2024-09-30'))) == 1


def test_panel_has_no_lookahead_in_ppi_standardisation(monkeypatch, tmp_path):
    """The PPI z-score must use the visible sample, not the full series."""
    import ecpmodel.features.build as B

    idx = pd.date_range('2024-01-01', periods=12, freq='MS')
    ppi = pd.Series(np.linspace(100, 111, len(idx)), index=idx)
    raw = {
        'fx': pd.Series(1.1, index=idx),
        'brent': pd.Series(80.0, idx),
        'gas': pd.Series(10.0, idx),
        'uv': pd.Series(1000.0, idx),
        'ppi': ppi,
    }
    monkeypatch.setattr(B, 'SERIES_FILES', {k: k for k in raw})
    monkeypatch.setattr(B, '_ld', lambda name: raw[name])
    monkeypatch.setattr(B, '_discover_ppi', lambda: 'ppi')
    monkeypatch.setattr(B, 'load_cfg', lambda name='data': {
        'start': '2024-01',
        'availability_lag_days': {'fx': 5, 'brent': 7, 'gas': 35, 'eua': 5,
                                  'uv': 65, 'ppi': 35},
        'manual': {'regimes': str(tmp_path / 'none.csv')},
    })

    heads = pd.DataFrame({'month': [], 'y': [], 'available_on': [], 'rel': [], 'repl': []})
    panel = build_panel(t0=pd.Timestamp('2024-07-31'), headlines=heads)

    # PPI with a 35-day lag is visible for Jan..May at this origin
    visible = ppi.loc[:pd.Timestamp('2024-05-01')]
    z = panel['endog']['ppi'].dropna()
    assert len(z)
    expected = (ppi.loc[z.index] - visible.mean()) / visible.std(ddof=0)
    np.testing.assert_allclose(z.to_numpy(), expected.to_numpy())
    # and the standardisation is *not* the full-sample one, otherwise the test
    # would pass while the model peeked
    full = (ppi.loc[z.index] - ppi.mean()) / ppi.std(ddof=0)
    assert not np.allclose(z.to_numpy(), full.to_numpy())


def test_panel_end_dates_after_the_origin_by_the_horizon(monkeypatch):
    import ecpmodel.features.build as B

    idx = pd.date_range('2024-01-01', periods=12, freq='MS')
    raw = {k: pd.Series(1.0, index=idx) for k in ('fx', 'brent', 'gas', 'uv')}
    monkeypatch.setattr(B, 'SERIES_FILES', {k: k for k in raw})
    monkeypatch.setattr(B, '_ld', lambda name: raw[name])
    monkeypatch.setattr(B, '_discover_ppi', lambda: None)
    monkeypatch.setattr(B, 'load_cfg', lambda name='data': {
        'start': '2024-01',
        'availability_lag_days': {k: 1 for k in raw},
        'manual': {'regimes': 'none.csv'},
    })
    heads = pd.DataFrame({'month': [], 'y': [], 'available_on': [], 'rel': [], 'repl': []})

    panel = build_panel(t0=pd.Timestamp('2024-06-30'), horizon=3, headlines=heads)
    assert panel['endog'].index.max() == pd.Timestamp('2024-09-01')
    assert panel['X'].index.max() == pd.Timestamp('2024-09-01')
    # the forecast tail is unobserved: that is what makes the filter forecast
    assert panel['endog']['comext'].loc['2024-08-01':].isna().all()


def test_cost_tail_is_forward_filled(monkeypatch, tmp_path):
    """Unpublished cost months are carried forward into the forecast horizon."""
    import ecpmodel.features.build as B

    idx = pd.date_range('2024-01-01', periods=12, freq='MS')
    brent = pd.Series(np.arange(80.0, 92.0), index=idx)
    raw = {k: pd.Series(1.0, index=idx) for k in ('fx', 'gas', 'uv')}
    raw['brent'] = brent
    monkeypatch.setattr(B, 'SERIES_FILES', {k: k for k in raw})
    monkeypatch.setattr(B, '_ld', lambda name: raw[name])
    monkeypatch.setattr(B, '_discover_ppi', lambda: None)
    monkeypatch.setattr(B, 'load_cfg', lambda name='data': {
        'start': '2024-01',
        'availability_lag_days': {k: 40 for k in raw},   # visible through 2024-03
        'manual': {'regimes': str(tmp_path / 'none.csv')},
    })
    heads = pd.DataFrame({'month': [], 'y': [], 'available_on': [], 'rel': [], 'repl': []})
    panel = build_panel(t0=pd.Timestamp('2024-06-30'), horizon=2, headlines=heads)

    b = panel['cost']['brent_eur']
    last_visible = pd.Timestamp('2024-04-01')          # 40-day lag, origin 2024-06-30
    assert b.loc[last_visible] == pytest.approx(83.0)
    assert b.loc[last_visible:].nunique() == 1          # the tail repeats, not NaN
    assert b.loc[last_visible] == b.iloc[-1]
    assert b.loc[:'2024-04-01'].is_monotonic_increasing
    assert panel['X'].index.max() == pd.Timestamp('2024-08-01')


def test_missing_cost_series_is_dropped_from_the_regressors(monkeypatch, tmp_path):
    """No EUA file means no EUA column - not a column of zeros."""
    import ecpmodel.features.build as B

    idx = pd.date_range('2024-01-01', periods=12, freq='MS')
    raw = {k: pd.Series(1.0, index=idx) for k in ('fx', 'brent', 'gas', 'uv')}
    monkeypatch.setattr(B, 'SERIES_FILES', {k: k for k in raw})
    monkeypatch.setattr(B, '_ld', lambda name: raw[name])
    monkeypatch.setattr(B, '_discover_ppi', lambda: None)
    monkeypatch.setattr(B, 'load_cfg', lambda name='data': {
        'start': '2024-01',
        'availability_lag_days': {k: 1 for k in raw},
        'manual': {'regimes': str(tmp_path / 'none.csv')},
    })
    heads = pd.DataFrame({'month': [], 'y': [], 'available_on': [], 'rel': [], 'repl': []})
    panel = build_panel(headlines=heads)
    assert panel['cost_cols'] == ['brent_eur', 'gas_eur_mwh']
    assert list(panel['X'].columns) == ['const', 'brent_eur', 'gas_eur_mwh']

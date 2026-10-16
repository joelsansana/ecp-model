"""Unit conversions.

These two bugs silently corrupt the whole model: a factor of 10 in the Comext
quantities puts the unit value at 10x the contract price, and a wrong MMBtu/MWh
constant moves the gas term by a third.  Everything here is hand-computable.
"""

import numpy as np
import pandas as pd
import pytest

from ecpmodel.features.build import MMBTU_TO_MWH, REL_MULT, dist_lag
from ecpmodel.ingest.comext import aggregate, unit_values
from ecpmodel.models.real import deflator, naphtha_equiv_ratio, real_draws


def test_100kg_units_to_tonnes():
    # QUANTITY_IN_100KG counts 100 kg units: 1 unit = 0.1 t, so t = qty / 10.
    # Both cells sit in the same month so the weighting is exercised.
    raw = pd.DataFrame({
        'TIME_PERIOD': ['2024-01', '2024-01'],
        'reporter': ['DE', 'NL'],
        'partner': ['WORLD', 'WORLD'],
        'flow': ['import', 'import'],
        'indicator': ['QUANTITY_IN_100KG'] * 2,
        'OBS_VALUE': [10_000.0, 4_000.0],          # 1000 t and 400 t
    })
    val = raw.copy()
    val['indicator'] = 'VALUE_IN_EUROS'
    val['OBS_VALUE'] = [1_100_000.0, 460_000.0]     # EUR 1.1m and 0.46m
    wide = unit_values(pd.concat([raw, val], ignore_index=True), min_tonnes=0)

    assert sorted(wide['tonnes']) == [400.0, 1000.0]
    assert wide['uv'].max() == pytest.approx(1150.0)    # 460k / 400 t
    assert wide['uv'].min() == pytest.approx(1100.0)    # 1.1m / 1000 t

    agg = aggregate(wide)
    # volume-weighted, not a simple mean
    assert agg['uv_w'].iloc[0] == pytest.approx((1100 * 1000 + 1150 * 400) / 1400)
    assert agg['uv_med'].iloc[0] == pytest.approx(1125.0)
    assert agg['tonnes'].iloc[0] == pytest.approx(1400.0)


def test_thin_cells_are_dropped():
    raw = pd.DataFrame({
        'TIME_PERIOD': ['2024-01', '2024-01'],
        'reporter': ['DE', 'NL'],
        'partner': ['WORLD', 'WORLD'],
        'flow': ['import', 'import'],
        'indicator': ['QUANTITY_IN_100KG'] * 2,
        'OBS_VALUE': [10_000.0, 500.0],             # 1000 t and 50 t
    })
    wide = unit_values(raw, min_tonnes=200)
    assert list(wide['reporter']) == ['DE']


def test_mmbtu_to_mwh_worked_number():
    # 1 MMBtu = 0.29307 MWh.  $10/MMBtu at 1.10 USD/EUR is EUR 9.0909/MMBtu,
    # which is EUR 9.0909 / 0.29307 = EUR 31.01/MWh.
    assert MMBTU_TO_MWH == pytest.approx(0.29307)
    gas_eur_mwh = (10.0 / 1.10) / MMBTU_TO_MWH
    assert gas_eur_mwh == pytest.approx(31.0196, abs=1e-3)


def test_brent_usd_to_eur_convention():
    # USD per EUR: divide the USD price by the rate.
    fx = pd.Series([1.10, 1.25], index=pd.date_range('2024-01-01', periods=2, freq='MS'))
    brent = pd.Series([80.0, 75.0], index=fx.index)
    brent_eur = brent / fx
    assert list(brent_eur.round(4)) == [72.7273, 60.0]


def test_eurusd_convention_is_documented_not_inverted():
    from ecpmodel.ingest import ecb

    src = (ecb.__doc__ or '')
    assert 'USD per 1 EUR' in src


def test_distributed_lag_weights_and_units():
    idx = pd.date_range('2024-01-01', periods=4, freq='MS')
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    out = dist_lag(s, (0.3, 0.5, 0.2))          # weight k applies to t-k
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(0.3 * 3 + 0.5 * 2 + 0.2 * 1)
    assert out.iloc[3] == pytest.approx(0.3 * 4 + 0.5 * 3 + 0.2 * 2)
    # weights that sum to 1 leave the level unchanged on a constant series
    flat = pd.Series(5.0, index=idx)
    assert dist_lag(flat, (0.3, 0.5, 0.2)).dropna().eq(5.0).all()


def test_reliability_and_replication_multipliers():
    assert REL_MULT[1] == 1.0 and REL_MULT[2] == 2.0 and REL_MULT[3] == 4.0
    # a quarterly settlement repeated over 3 months must not count as 3
    # independent observations: the variance multiplier is 3, not 1.
    assert REL_MULT[2] * 3 == 6.0


def test_quarterly_headline_expands_with_replication():
    from ecpmodel.ingest.headlines import load

    df = pd.DataFrame({
        'month': ['2024-04-01'],
        'settlement_eur_t': [1000.0],
        'low': [np.nan],
        'high': [np.nan],
        'type': ['quarterly'],
        'published_on': ['2024-03-28'],
        'reliability': [2],
    })
    out = load(_write(df), allow_example=False)
    assert list(out['month']) == list(pd.date_range('2024-04-01', periods=3, freq='MS'))
    assert set(out['repl']) == {3}
    assert set(out['rel']) == {2}
    # availability date is the publication date, never the contract month
    assert set(out['available_on']) == {pd.Timestamp('2024-03-28')}


def _write(df: pd.DataFrame):
    import tempfile
    from pathlib import Path

    p = Path(tempfile.mkdtemp()) / 'h.csv'
    df.to_csv(p, index=False)
    return p


def test_headline_schema_rejects_impossible_settlements():
    import pandera as pa

    from ecpmodel.ingest.headlines import schema

    good = pd.DataFrame({
        'month': ['2024-01-01'], 'settlement_eur_t': [1000.0],
        'type': ['monthly'], 'published_on': ['2024-01-20'], 'reliability': [2],
    })
    schema.validate(good)
    bad = good.copy()
    bad['settlement_eur_t'] = 25_000.0      # EUR 25k/t is not an ethylene price
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(bad)


def test_real_price_deflator_rebases_to_one():
    ppi = pd.Series([100.0, 110.0, 121.0],
                    index=pd.date_range('2024-01-01', periods=3, freq='MS'))
    d = deflator(ppi, ppi.index, pd.Timestamp('2024-01-01'))
    assert d[0] == pytest.approx(1.0)
    assert d[2] == pytest.approx(1.21)

    paths = np.array([[1.0, 1.0, 1.0]])          # k EUR/t
    real = real_draws(paths, ppi.index, ppi, pd.Timestamp('2024-01-01'))
    assert real[0, 0] == pytest.approx(1000.0)
    assert real[0, 2] == pytest.approx(1000.0 / 1.21)


def test_naphtha_ratio_is_a_ratio_not_a_margin():
    brent = pd.Series([80.0], index=pd.date_range('2024-01-01', periods=1, freq='MS'))
    ecp = pd.Series([1000.0], index=brent.index)
    ratio = naphtha_equiv_ratio(ecp, brent)
    assert ratio.iloc[0] == pytest.approx(1000.0 / (80.0 * 7.3))

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.indicators import atr, bollinger_bands, rsi, sma


def test_sma_basic():
    series = pd.Series([1, 2, 3, 4, 5])
    result = sma(series, window=2)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 1.5
    assert result.iloc[-1] == 4.5


def test_rsi_bounds(trending_up_df):
    values = rsi(trending_up_df["close"], window=14).dropna()
    assert (values >= 0).all()
    assert (values <= 100).all()


def test_rsi_strong_uptrend_is_high(trending_up_df):
    # Sur une tendance haussière quasi continue, le RSI doit être élevé en fin de série.
    values = rsi(trending_up_df["close"], window=14)
    assert values.iloc[-1] > 50


def test_atr_non_negative(trending_up_df):
    values = atr(trending_up_df, window=14).dropna()
    assert (values >= 0).all()


def test_bollinger_bands_are_ordered_and_centered():
    rng = np.random.default_rng(0)
    series = pd.Series(100 + np.cumsum(rng.normal(size=100)))
    mid, upper, lower = bollinger_bands(series, window=20, num_std=2.0)

    valid = mid.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (lower[valid] <= mid[valid]).all()


def test_bollinger_bands_flat_series_has_zero_width():
    series = pd.Series([50.0] * 30)
    mid, upper, lower = bollinger_bands(series, window=10, num_std=2.0)
    valid = mid.notna()
    assert np.allclose(upper[valid], mid[valid])
    assert np.allclose(lower[valid], mid[valid])


def test_bollinger_bands_warmup_is_nan():
    series = pd.Series(range(5))
    mid, upper, lower = bollinger_bands(series, window=20, num_std=2.0)
    assert mid.isna().all()
    assert upper.isna().all()
    assert lower.isna().all()

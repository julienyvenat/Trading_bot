from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.indicators import atr, rsi, sma


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

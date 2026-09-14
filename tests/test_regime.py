from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.portfolio.regime import latest_regime_scale, regime_scale_series


def _make_close(prices) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=index)


def test_regime_scale_is_bullish_above_sma():
    close = _make_close(100 + np.arange(300) * 0.5)  # tendance haussière franche
    scale = regime_scale_series(close, sma_window=200, bearish_scale=0.3)
    assert scale.iloc[-1] == 1.0


def test_regime_scale_is_bearish_below_sma():
    up = 100 + np.arange(250) * 1.0  # établit une SMA haute
    down = up[-1] - np.arange(1, 60) * 2.0  # puis chute nettement en dessous
    close = _make_close(np.concatenate([up, down]))
    scale = regime_scale_series(close, sma_window=200, bearish_scale=0.3)
    assert scale.iloc[-1] == pytest.approx(0.3)


def test_regime_scale_defaults_to_bullish_during_warmup():
    close = _make_close(100 + np.arange(50) * 1.0)  # moins de points que sma_window
    scale = regime_scale_series(close, sma_window=200, bearish_scale=0.3)
    assert (scale == 1.0).all()


def test_latest_regime_scale_matches_series_last_value():
    close = _make_close(100 + np.arange(300) * 0.5)
    expected = regime_scale_series(close, 200, 0.3).iloc[-1]
    assert latest_regime_scale(close, sma_window=200, bearish_scale=0.3) == expected

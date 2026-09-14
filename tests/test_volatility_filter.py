from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.portfolio.volatility_filter import latest_volatility_scale, volatility_scale_series


def _make_close(prices) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=index)


def test_volatility_scale_is_normal_when_close_to_sma():
    close = _make_close(np.full(60, 20.0))  # plat : toujours égal à sa propre SMA
    scale = volatility_scale_series(close, sma_window=20, spike_threshold_pct=0.15, spike_scale=0.5)
    assert scale.iloc[-1] == 1.0


def test_volatility_scale_drops_on_spike_above_sma():
    base = np.full(40, 20.0)
    spike = np.full(5, 30.0)  # +50% d'un coup, bien au-dessus du seuil de 15%
    close = _make_close(np.concatenate([base, spike]))
    scale = volatility_scale_series(close, sma_window=20, spike_threshold_pct=0.15, spike_scale=0.5)
    assert scale.iloc[-1] == pytest.approx(0.5)


def test_volatility_scale_defaults_to_normal_during_warmup():
    close = _make_close(np.full(10, 20.0))  # moins de points que sma_window
    scale = volatility_scale_series(close, sma_window=20, spike_threshold_pct=0.15, spike_scale=0.5)
    assert (scale == 1.0).all()


def test_latest_volatility_scale_matches_series_last_value():
    base = np.full(40, 20.0)
    spike = np.full(5, 30.0)
    close = _make_close(np.concatenate([base, spike]))
    expected = volatility_scale_series(close, 20, 0.15, 0.5).iloc[-1]
    assert latest_volatility_scale(close, 20, 0.15, 0.5) == expected


def test_latest_volatility_scale_empty_series_is_neutral():
    assert latest_volatility_scale(pd.Series(dtype=float), 20, 0.15, 0.5) == 1.0

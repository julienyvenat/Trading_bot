"""Tests des fonctions de correspondance `universe.timeframe` <-> granularité
attendue par yfinance (backtest) et Alpaca (live). Ne nécessitent aucun accès
réseau : ce sont de pures fonctions de parsing/mapping."""

from __future__ import annotations

import pytest

from trading_bot.data.historical import yfinance_interval_for_timeframe
from trading_bot.data.market_data import _parse_timeframe


@pytest.mark.parametrize(
    "timeframe,expected",
    [
        ("1Day", "1d"),
        ("1Hour", "1h"),
        ("30Min", "30m"),
        ("15Min", "15m"),
        ("5Min", "5m"),
        ("1Min", "1m"),
    ],
)
def test_yfinance_interval_for_timeframe_known_values(timeframe, expected):
    assert yfinance_interval_for_timeframe(timeframe) == expected


def test_yfinance_interval_for_timeframe_unknown_raises():
    with pytest.raises(ValueError):
        yfinance_interval_for_timeframe("2Hour")


def test_parse_timeframe_builds_expected_alpaca_timeframe():
    from alpaca.data.timeframe import TimeFrameUnit

    tf = _parse_timeframe("5Min")
    assert tf.amount == 5
    assert tf.unit == TimeFrameUnit.Minute

    tf_day = _parse_timeframe("1Day")
    assert tf_day.amount == 1
    assert tf_day.unit == TimeFrameUnit.Day


def test_parse_timeframe_unknown_format_raises():
    with pytest.raises(ValueError):
        _parse_timeframe("banane")

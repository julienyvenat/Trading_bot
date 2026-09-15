from __future__ import annotations

import pytest

from trading_bot import cli
from trading_bot.config import AppConfig, BacktestConfig, LiveConfig, MarketConfig, RiskConfig, StrategyConfig

_RISK = RiskConfig(
    allow_short=False,
    max_gross_exposure_pct=0.9,
    max_position_weight_pct=0.25,
    risk_per_trade_pct=0.01,
    atr_stop_multiple=2.5,
    atr_window=14,
    max_open_positions=5,
    max_daily_loss_pct=0.03,
    max_drawdown_pct=0.20,
)


def _make_config(data_source: str, timeframe: str = "5Min") -> AppConfig:
    return AppConfig(
        symbols=["TSLA"],
        timeframe=timeframe,
        strategies=[StrategyConfig(name="bollinger_scalping", enabled=True, weight=1.0, params={})],
        risk=_RISK,
        market=MarketConfig(calendar="NYSE", close_buffer_minutes=15),
        backtest=BacktestConfig(
            start_date="2024-01-01", end_date=None, initial_cash=100_000, commission_pct=0.0005, data_source=data_source
        ),
        live=LiveConfig(loop_interval_seconds=300, trade_only_when_market_open=True),
    )


def test_fetch_data_dispatches_to_yfinance_by_default(monkeypatch):
    called = {}

    def _fake_fetch(symbols, start_date, end_date, interval):
        called.update(symbols=symbols, start_date=start_date, end_date=end_date, interval=interval)
        return {"TSLA": "yfinance-data"}

    monkeypatch.setattr("trading_bot.data.historical.fetch_historical_data", _fake_fetch)

    result = cli._fetch_data(_make_config("yfinance"), ["TSLA"])

    assert result == {"TSLA": "yfinance-data"}
    assert called["interval"] == "5m"


def test_fetch_data_dispatches_to_alpaca(monkeypatch):
    called = {}

    def _fake_fetch(symbols, start_date, end_date, timeframe, credentials):
        called.update(symbols=symbols, timeframe=timeframe)
        return {"TSLA": "alpaca-data"}

    monkeypatch.setattr("trading_bot.data.market_data.fetch_historical_bars", _fake_fetch)
    monkeypatch.setattr("trading_bot.config.load_alpaca_credentials", lambda: object())

    result = cli._fetch_data(_make_config("alpaca"), ["TSLA"])

    assert result == {"TSLA": "alpaca-data"}
    assert called["timeframe"] == "5Min"


def test_fetch_data_rejects_unknown_data_source():
    with pytest.raises(ValueError):
        cli._fetch_data(_make_config("not_a_real_source"), ["TSLA"])

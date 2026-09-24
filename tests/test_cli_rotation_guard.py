from __future__ import annotations

import argparse

import pytest

from trading_bot import cli
from trading_bot.config import (
    AppConfig,
    BacktestConfig,
    LiveConfig,
    MarketConfig,
    RiskConfig,
    StrategyConfig,
    UniverseRotationConfig,
)
from trading_bot.state import LiveState

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


def _make_config(universe_rotation: UniverseRotationConfig) -> AppConfig:
    return AppConfig(
        symbols=["SPY"],
        timeframe="5Min",
        strategies=[
            StrategyConfig(
                name="bollinger_scalping", enabled=True, weight=1.0, params={}, universe_rotation=universe_rotation
            )
        ],
        risk=_RISK,
        market=MarketConfig(calendar="NYSE", close_buffer_minutes=15),
        backtest=BacktestConfig(start_date="2024-01-01", end_date=None, initial_cash=100_000, commission_pct=0.0005),
        live=LiveConfig(loop_interval_seconds=300, trade_only_when_market_open=True),
    )


def test_paper_refuses_to_start_with_rotation_enabled(monkeypatch):
    """Le support de `universe_rotation` est backtest/optimize/walk-forward
    uniquement pour l'instant (voir `trading_bot.portfolio.allocator`) :
    `paper` doit refuser de démarrer plutôt que de trader silencieusement
    `universe.symbols` en ignorant la rotation configurée, ce qui romprait
    la parité backtest/live."""
    config = _make_config(UniverseRotationConfig(enabled=True, candidates=["TSLA"], min_confidence=0.6))
    monkeypatch.setattr(cli, "load_config", lambda path: config)

    with pytest.raises(SystemExit):
        cli.cmd_paper(argparse.Namespace(config=None, once=True, dry_run=True))


def test_paper_proceeds_past_guard_when_rotation_disabled(monkeypatch):
    """Sans rotation active, le garde-fou ne doit pas se déclencher (on
    vérifie juste qu'on dépasse le point du garde-fou, pas tout le cycle)."""
    config = _make_config(UniverseRotationConfig(enabled=False))
    monkeypatch.setattr(cli, "load_config", lambda path: config)

    sentinel = RuntimeError("reached past the rotation guard")

    def _fake_run_once(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr("trading_bot.live.engine.run_once", _fake_run_once)
    monkeypatch.setattr("trading_bot.config.load_alpaca_credentials", lambda: None)
    monkeypatch.setattr("trading_bot.execution.alpaca_broker.AlpacaBroker", lambda credentials: None)
    monkeypatch.setattr("trading_bot.state.load_state", lambda path: LiveState())
    monkeypatch.setattr("trading_bot.state.save_state", lambda path, state: None)

    with pytest.raises(RuntimeError) as excinfo:
        cli.cmd_paper(argparse.Namespace(config=None, once=True, dry_run=True))
    assert excinfo.value is sentinel

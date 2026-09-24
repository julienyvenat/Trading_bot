from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.engine import run_backtest
from trading_bot.config import (
    AppConfig,
    BacktestConfig,
    LiveConfig,
    MarketConfig,
    RiskConfig,
    StrategyConfig,
    load_config,
)

from conftest import make_ohlcv

FORTUNEO = {
    "tiers": [
        {"up_to": 500, "pct": 0.005, "min": 1.95},
        {"up_to": 2000, "fixed": 1.95},
        {"up_to": None, "pct": 0.002},
    ]
}


def make_buyhold_config(**risk_overrides) -> AppConfig:
    risk = dict(
        allow_short=False,
        max_gross_exposure_pct=1.0,
        max_position_weight_pct=1.0,
        risk_per_trade_pct=1.0,
        atr_stop_multiple=6.0,
        atr_window=14,
        max_open_positions=1,
        max_daily_loss_pct=1.0,
        max_drawdown_pct=1.0,
        stop_mode="none",
    )
    risk.update(risk_overrides)
    return AppConfig(
        symbols=["ETF"],
        timeframe="1Day",
        strategies=[StrategyConfig(name="buy_and_hold", enabled=True, weight=1.0)],
        risk=RiskConfig(**risk),
        market=MarketConfig(calendar="NYSE", close_buffer_minutes=15),
        backtest=BacktestConfig(
            start_date="2020-01-01",
            end_date=None,
            initial_cash=2306.59,
            commission_pct=0.005,
            commission_schedule=FORTUNEO,
            whole_shares=True,
            min_order_value=150,
            max_fee_pct=0.02,
            rebalance_tolerance_pct=0.10,
        ),
        live=LiveConfig(loop_interval_seconds=3600, trade_only_when_market_open=True),
    )


def test_buy_and_hold_whole_shares_single_order_and_real_fees():
    df = make_ohlcv(np.linspace(59.0, 80.0, 300))
    config = make_buyhold_config()
    result = run_backtest(config, {"ETF": df})

    qty = result.final_positions["ETF"]
    assert qty == int(qty) and qty > 0
    assert result.num_orders == 1  # une seule entrée, aucun micro-rééquilibrage
    # Signal sur la 1re séance simulée -> exécution à l'ouverture de la suivante.
    fill_price = df["open"].loc[result.equity_curve.index[1]]
    fee = qty * fill_price * 0.002  # palier > 2 000 € : 0,20 %
    assert qty * fill_price + fee <= 2306.59
    assert (qty + 1) * fill_price * 1.002 > 2306.59  # on a bien acheté le max d'actions entières
    assert result.total_fees == pytest.approx(fee)
    # Pas de marge : le cash ne passe jamais sous zéro.
    final_cash = result.equity_curve.iloc[-1] - qty * df["close"].iloc[-1]
    assert final_cash >= 0


def test_legacy_config_keeps_fractional_shares():
    df = make_ohlcv(np.linspace(59.0, 80.0, 300))
    config = make_buyhold_config()
    config.backtest = BacktestConfig(start_date="2020-01-01", end_date=None, initial_cash=2306.59, commission_pct=0.0)
    result = run_backtest(config, {"ETF": df})
    assert result.final_positions["ETF"] != int(result.final_positions["ETF"])


def _crash_then_recover() -> pd.DataFrame:
    up = np.linspace(100, 150, 250)
    crash = np.linspace(150, 90, 30)
    flat = np.full(60, 90.0)
    recover = np.linspace(90, 200, 200)
    return make_ohlcv(np.concatenate([up, crash, flat, recover]))


def test_trailing_pct_stop_exits_and_sma_reentry_waits():
    df = _crash_then_recover()
    config = make_buyhold_config(stop_mode="trailing_pct", trailing_stop_pct=0.15, stop_reentry="sma", stop_reentry_sma_window=50)
    result = run_backtest(config, {"ETF": df})

    assert result.num_stop_exits == 1
    stop_trade = next(t for t in result.trades if t.exit_reason == "stop")
    # Stop à -15 % du plus haut (les plus hauts de make_ohlcv sont à close x 1.005).
    assert stop_trade.exit_price == pytest.approx(150 * 1.005 * 0.85, rel=0.01)
    # Ré-entrée seulement une fois la clôture repassée au-dessus de la SMA50 :
    # on rejoue le backtest jusqu'à la veille de ce croisement (toujours flat)
    # puis quelques séances après (réinvesti).
    sma50 = df["close"].rolling(50).mean()
    after_stop = df.index[df.index > stop_trade.exit_date]
    cross = next(d for d in after_stop if df.loc[d, "close"] > sma50.loc[d])
    assert (cross - stop_trade.exit_date).days > 30  # blocage réel (plateau à 90 sous la SMA)

    before = replace(config, backtest=replace(config.backtest, end_date=str((cross - pd.Timedelta(days=1)).date())))
    assert run_backtest(before, {"ETF": df}).final_positions["ETF"] == 0
    after = replace(config, backtest=replace(config.backtest, end_date=str((cross + pd.Timedelta(days=5)).date())))
    assert run_backtest(after, {"ETF": df}).final_positions["ETF"] > 0


def test_trailing_pct_stop_immediate_reentry_rebuys_next_day():
    df = _crash_then_recover()
    config = make_buyhold_config(stop_mode="trailing_pct", trailing_stop_pct=0.15, stop_reentry="immediate")
    result = run_backtest(config, {"ETF": df})
    assert result.num_stop_exits >= 1
    # stop + rachat immédiat : au moins 3 ordres (entrée, stop, ré-entrée)
    assert result.num_orders >= 3


def test_stop_mode_none_never_exits():
    df = _crash_then_recover()
    result = run_backtest(make_buyhold_config(stop_mode="none"), {"ETF": df})
    assert result.num_stop_exits == 0 and result.num_orders == 1


def test_trailing_pct_from_atr_when_no_fixed_pct():
    df = _crash_then_recover()
    config = make_buyhold_config(stop_mode="trailing_pct", trailing_stop_pct=None, atr_stop_multiple=4.0, stop_reentry="sma")
    result = run_backtest(config, {"ETF": df})
    assert result.num_stop_exits >= 1


def test_invalid_stop_reentry_rejected():
    df = make_ohlcv(np.linspace(59.0, 80.0, 100))
    with pytest.raises(ValueError):
        run_backtest(make_buyhold_config(stop_reentry="foo"), {"ETF": df})


def test_example_pea_configs_load_and_run_on_synthetic_data():
    for name in ("config_pea_buyhold.example.yaml", "config_pea_etf_momentum.example.yaml"):
        config = load_config(f"config/{name}")
        assert config.backtest.whole_shares and config.backtest.commission_schedule
        assert config.live.broker == "manual" and config.live.daily_run_after == "18:30"
        config.market = replace(config.market, calendar="NYSE")
        config.backtest = replace(config.backtest, start_date="2021-01-01")
        rng = np.random.default_rng(0)
        data = {
            s: make_ohlcv(50 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 700))), start="2019-06-03")
            for s in config.symbols
        }
        result = run_backtest(config, data)
        assert all(q == int(q) for q in result.final_positions.values())
        assert result.num_orders >= 1

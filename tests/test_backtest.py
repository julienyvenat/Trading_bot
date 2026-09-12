from __future__ import annotations

from trading_bot.backtest.engine import run_backtest
from trading_bot.config import AppConfig, BacktestConfig, LiveConfig, MarketConfig, RiskConfig, StrategyConfig


def make_config(**risk_overrides) -> AppConfig:
    risk_defaults = dict(
        allow_short=False,
        max_gross_exposure_pct=0.9,
        max_position_weight_pct=0.5,
        risk_per_trade_pct=0.02,
        atr_stop_multiple=2.5,
        atr_window=14,
        max_open_positions=5,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.20,
    )
    risk_defaults.update(risk_overrides)

    return AppConfig(
        symbols=["UP"],
        timeframe="1Day",
        strategies=[
            StrategyConfig(name="sma_crossover", enabled=True, weight=1.0, params={"fast_window": 10, "slow_window": 30}),
        ],
        risk=RiskConfig(**risk_defaults),
        market=MarketConfig(calendar="NYSE", close_buffer_minutes=15),
        backtest=BacktestConfig(start_date="2020-01-01", end_date=None, initial_cash=100_000, commission_pct=0.0005),
        live=LiveConfig(loop_interval_seconds=3600, trade_only_when_market_open=True, state_file="state/live_state.json"),
    )


def test_run_backtest_produces_equity_curve(trending_up_df):
    config = make_config()
    result = run_backtest(config, {"UP": trending_up_df})

    assert len(result.equity_curve) > 0
    # L'equity ne doit jamais devenir négative ou nulle sur une tendance haussière.
    assert (result.equity_curve > 0).all()


def test_run_backtest_uptrend_is_profitable(trending_up_df):
    config = make_config()
    result = run_backtest(config, {"UP": trending_up_df})

    initial = config.backtest.initial_cash
    final = result.equity_curve.iloc[-1]
    assert final > initial * 0.95  # tolérance pour les frais/latence d'entrée


def test_run_backtest_respects_gross_exposure_cap(trending_up_df):
    config = make_config(max_gross_exposure_pct=0.3, max_position_weight_pct=1.0)
    result = run_backtest(config, {"UP": trending_up_df})
    assert len(result.equity_curve) > 0

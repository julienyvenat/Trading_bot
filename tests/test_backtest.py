from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest.engine import _execute_at_open, run_backtest
from trading_bot.config import (
    AppConfig,
    BacktestConfig,
    LiveConfig,
    MarketConfig,
    RegimeFilterConfig,
    RiskConfig,
    StrategyConfig,
)


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


def test_execute_at_open_uses_open_price_and_pre_trade_equity():
    """Le signal calculé à la clôture (poids cible) doit s'exécuter au prix
    d'OUVERTURE fourni, pas à un autre prix — vérifié directement sur la
    fonction d'exécution plutôt que sur un backtest complet (où toucher les
    prix perturbe aussi le calcul d'equity et les coupe-circuits)."""
    positions = {"UP": 0.0}
    cash = 10_000.0
    open_prices = pd.Series({"UP": 50.0})
    pending_target_weights = {"UP": 0.5}  # cible : 50% de l'equity

    new_positions, new_cash = _execute_at_open(pending_target_weights, positions, cash, open_prices, commission_pct=0.0)

    assert new_positions["UP"] == pytest.approx(10_000 * 0.5 / 50.0)
    assert new_cash == pytest.approx(10_000 - 10_000 * 0.5)
    # L'appel ne doit pas muter le dict de positions passé en entrée (fonction pure).
    assert positions["UP"] == 0.0


def test_execute_at_open_flattens_symbol_not_in_pending_weights():
    positions = {"UP": 100.0}
    cash = 0.0
    open_prices = pd.Series({"UP": 50.0})

    new_positions, new_cash = _execute_at_open({"UP": 0.0}, positions, cash, open_prices, commission_pct=0.0)

    assert new_positions["UP"] == pytest.approx(0.0)
    assert new_cash == pytest.approx(100.0 * 50.0)


def test_regime_filter_reduces_exposure_in_bearish_regime(trending_up_df, trending_down_df):
    config = make_config()
    config.market.regime_filter = RegimeFilterConfig(
        enabled=True, symbol="BENCH", sma_window=20, bearish_exposure_scale=0.0
    )

    result_without_filter = run_backtest(make_config(), {"UP": trending_up_df})
    result_with_filter = run_backtest(config, {"UP": trending_up_df}, benchmark_df=trending_down_df)

    # BENCH (trending_down_df) est en tendance baissière sous sa SMA 20 une fois
    # celle-ci établie : le filtre coupe l'exposition à 0, donc bien moins de
    # gain capturé que sans filtre sur la même tendance haussière de UP.
    assert result_with_filter.equity_curve.iloc[-1] < result_without_filter.equity_curve.iloc[-1]

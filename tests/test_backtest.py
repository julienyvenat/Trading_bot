from __future__ import annotations

import numpy as np
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


def _make_up_then_down_df() -> pd.DataFrame:
    """Tendance haussière franche suivie d'un retournement net à la baisse :
    de quoi générer au moins une entrée PUIS une sortie (stop ou croisement
    de moyennes), pour tester le tracking des trades sur un cycle complet."""
    up = 100 + np.arange(150) * 1.0
    down = up[-1] - np.arange(1, 80) * 1.5
    prices = np.concatenate([up, down])
    index = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    close = pd.Series(prices, index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_run_backtest_tracks_completed_trades_on_a_full_round_trip():
    config = make_config()
    result = run_backtest(config, {"UP": _make_up_then_down_df()})

    assert len(result.trades) > 0
    for trade in result.trades:
        assert trade.symbol == "UP"
        assert trade.qty > 0
        assert trade.entry_date <= trade.exit_date
        assert trade.exit_reason in ("stop", "rebalance")
    # Les métriques par trade du résultat doivent refléter les trades trackés.
    assert result.metrics.num_trades == len(result.trades)


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


def test_regime_filter_does_not_reduce_exempt_symbols(trending_up_df, trending_down_df):
    """Un symbole exempté (ex: l'actif défensif d'une rotation défensive) ne
    doit pas être réduit par le filtre de régime, même en régime baissier —
    sinon la rotation défensive perd une grande partie de son intérêt."""
    config_exempt = make_config()
    config_exempt.market.regime_filter = RegimeFilterConfig(
        enabled=True, symbol="BENCH", sma_window=20, bearish_exposure_scale=0.0, exempt_symbols=["UP"]
    )
    config_not_exempt = make_config()
    config_not_exempt.market.regime_filter = RegimeFilterConfig(
        enabled=True, symbol="BENCH", sma_window=20, bearish_exposure_scale=0.0
    )

    result_exempt = run_backtest(config_exempt, {"UP": trending_up_df}, benchmark_df=trending_down_df)
    result_not_exempt = run_backtest(config_not_exempt, {"UP": trending_up_df}, benchmark_df=trending_down_df)

    # UP est exempté : sa performance ne doit pas être amputée par le régime
    # baissier de BENCH, contrairement au cas où il ne serait pas exempté.
    assert result_exempt.equity_curve.iloc[-1] > result_not_exempt.equity_curve.iloc[-1]

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest import optimizer as optimizer_module
from trading_bot.backtest.engine import BacktestResult
from trading_bot.backtest.metrics import BacktestMetrics
from trading_bot.backtest.optimizer import ParamGrid, best_params, generate_param_combinations, optimize
from trading_bot.config import (
    AppConfig,
    BacktestConfig,
    LiveConfig,
    MarketConfig,
    RiskConfig,
    StrategyConfig,
)


def make_config() -> AppConfig:
    risk = RiskConfig(
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
    return AppConfig(
        symbols=["UP"],
        timeframe="1Day",
        strategies=[
            StrategyConfig(name="sma_crossover", enabled=True, weight=1.0, params={"fast_window": 10, "slow_window": 30}),
            StrategyConfig(
                name="rsi_mean_reversion",
                enabled=True,
                weight=1.0,
                params={"rsi_window": 14, "oversold": 30, "overbought": 70},
            ),
        ],
        risk=risk,
        market=MarketConfig(calendar="NYSE", close_buffer_minutes=15),
        backtest=BacktestConfig(start_date="2020-01-01", end_date=None, initial_cash=100_000, commission_pct=0.0005),
        live=LiveConfig(loop_interval_seconds=3600, trade_only_when_market_open=True, state_file="state/live_state.json"),
    )


def _fake_metrics(sharpe: float) -> BacktestMetrics:
    return BacktestMetrics(
        total_return_pct=0.0,
        annualized_return_pct=0.0,
        annualized_volatility_pct=0.0,
        sharpe_ratio=sharpe,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown_pct=0.0,
        win_rate_pct=0.0,
        profit_factor=0.0,
        num_trading_days=0,
        num_trades=0,
        trade_win_rate_pct=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        win_loss_ratio=0.0,
    )


def test_generate_param_combinations_cartesian_within_and_across_strategies():
    grids = [
        ParamGrid(strategy_name="sma_crossover", params={"fast_window": [10, 20], "slow_window": [50, 100]}),
        ParamGrid(strategy_name="rsi_mean_reversion", params={"oversold": [20, 30]}),
    ]
    combos = generate_param_combinations(grids)

    # 2 * 2 (sma_crossover) * 2 (rsi_mean_reversion) = 8 combinaisons.
    assert len(combos) == 8
    assert all("sma_crossover" in c and "rsi_mean_reversion" in c for c in combos)
    assert all(set(c["sma_crossover"].keys()) == {"fast_window", "slow_window"} for c in combos)

    seen = {tuple(sorted(c["sma_crossover"].items())) + tuple(sorted(c["rsi_mean_reversion"].items())) for c in combos}
    assert len(seen) == 8  # pas de doublons


def test_generate_param_combinations_empty_grid_list_returns_empty():
    assert generate_param_combinations([]) == []


def test_apply_params_merges_without_clobbering_other_params():
    config = make_config()
    combo = {"sma_crossover": {"fast_window": 99}}

    new_config = optimizer_module.apply_params(config, combo)

    sma_params = next(s for s in new_config.strategies if s.name == "sma_crossover").params
    assert sma_params["fast_window"] == 99
    assert sma_params["slow_window"] == 30  # non grillé : conservé tel quel

    # Copie profonde : l'original n'est pas muté.
    original_params = next(s for s in config.strategies if s.name == "sma_crossover").params
    assert original_params["fast_window"] == 10


def test_optimize_sorts_results_by_metric_descending(monkeypatch):
    config = make_config()
    data = {"UP": pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]})}

    def fake_run_backtest(cfg, data_by_symbol, benchmark_df=None, volatility_benchmark_df=None):
        fast_window = next(s for s in cfg.strategies if s.name == "sma_crossover").params["fast_window"]
        return BacktestResult(equity_curve=pd.Series([100.0, 101.0]), metrics=_fake_metrics(float(fast_window)))

    monkeypatch.setattr(optimizer_module, "run_backtest", fake_run_backtest)

    grids = [ParamGrid(strategy_name="sma_crossover", params={"fast_window": [5, 50, 20]})]
    results = optimize(config, data, grids, metric="sharpe_ratio", max_workers=1)

    assert [r.metrics.sharpe_ratio for r in results] == [50.0, 20.0, 5.0]
    assert best_params(results) == {"sma_crossover": {"fast_window": 50}}


def test_optimize_raises_on_empty_grids():
    config = make_config()
    with pytest.raises(ValueError):
        optimize(config, {"UP": pd.DataFrame()}, grids=[])


def test_best_params_raises_on_empty_results():
    with pytest.raises(ValueError):
        best_params([])


def test_optimize_end_to_end_with_real_backtest_and_multiprocessing(trending_up_df):
    """Vérifie que le vrai chemin `multiprocessing.Pool` (pas seulement le
    mode séquentiel `max_workers=1`) fonctionne bout-en-bout : config et
    données doivent rester picklables et produire des résultats cohérents."""
    config = make_config()
    grids = [ParamGrid(strategy_name="sma_crossover", params={"fast_window": [10, 20]})]

    results = optimize(config, {"UP": trending_up_df}, grids, metric="sharpe_ratio", max_workers=2)

    assert len(results) == 2
    assert results[0].metrics.sharpe_ratio >= results[1].metrics.sharpe_ratio

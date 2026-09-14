from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest.optimizer import ParamGrid
from trading_bot.backtest.walk_forward import _chain_equity_curves, make_folds, run_walk_forward
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
        ],
        risk=risk,
        market=MarketConfig(calendar="NYSE", close_buffer_minutes=15),
        backtest=BacktestConfig(start_date="2020-01-01", end_date=None, initial_cash=100_000, commission_pct=0.0005),
        live=LiveConfig(loop_interval_seconds=3600, trade_only_when_market_open=True, state_file="state/live_state.json"),
    )


def test_make_folds_produces_contiguous_non_overlapping_windows():
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-01-01") + pd.Timedelta(days=300)
    folds = make_folds(start, end, train_days=100, test_days=50, step_days=50)

    assert len(folds) >= 2
    for train_start, train_end, test_start, test_end in folds:
        assert train_end == test_start  # test démarre exactement où l'entraînement finit
        assert (test_end - test_start).days == 50
        assert (train_end - train_start).days == 100
    # Fenêtres de test contiguës (step == test_days) : chaque test suivant
    # démarre exactement où le précédent finit.
    for (_, _, _, test_end_a), (_, _, test_start_b, _) in zip(folds, folds[1:]):
        assert test_end_a == test_start_b


def test_make_folds_returns_empty_when_range_too_short():
    start = pd.Timestamp("2020-01-01")
    end = start + pd.Timedelta(days=10)
    folds = make_folds(start, end, train_days=100, test_days=50, step_days=50)
    assert folds == []


def test_chain_equity_curves_rescales_to_continue_from_previous_segment():
    idx_a = pd.date_range("2020-01-01", periods=3, freq="D")
    idx_b = pd.date_range("2020-01-10", periods=3, freq="D")
    segment_a = pd.Series([100_000.0, 105_000.0, 110_000.0], index=idx_a)
    # Deuxième segment reparti "à froid" de 100_000 (comme le ferait
    # `run_backtest` sur une nouvelle fenêtre) alors qu'il devrait continuer
    # depuis les 110_000 atteints par le premier segment.
    segment_b = pd.Series([100_000.0, 90_000.0, 95_000.0], index=idx_b)

    chained = _chain_equity_curves([segment_a, segment_b])

    assert chained.iloc[0] == pytest.approx(100_000.0)
    assert chained.iloc[2] == pytest.approx(110_000.0)  # fin du 1er segment inchangée
    # Le 2e segment est recalé pour repartir de 110_000 plutôt que 100_000 :
    # sa 1ère valeur doit donc valoir exactement 110_000 (facteur de repositionnement).
    assert chained.iloc[3] == pytest.approx(110_000.0)
    # Sa baisse de -10% (100k -> 90k) doit être appliquée sur la base recalée.
    assert chained.iloc[4] == pytest.approx(110_000.0 * 0.9)


def test_chain_equity_curves_empty_input_returns_empty_series():
    result = _chain_equity_curves([])
    assert len(result) == 0


def test_run_walk_forward_produces_folds_and_combined_metrics(trending_up_df):
    config = make_config()
    result = run_walk_forward(
        config,
        {"UP": trending_up_df},
        train_days=60,
        test_days=30,
        step_days=30,
    )

    assert len(result.folds) > 0
    for fold in result.folds:
        assert fold.test_start < fold.test_end
        assert fold.train_end == fold.test_start
    assert len(result.combined_out_of_sample_equity) > 0
    # Ne doit pas lever d'exception et produire un résumé textuel exploitable.
    assert "Cumulé sur toutes les fenêtres de TEST" in result.summary()


def test_run_walk_forward_raises_when_history_too_short(trending_up_df):
    config = make_config()
    with pytest.raises(ValueError):
        run_walk_forward(config, {"UP": trending_up_df}, train_days=10_000, test_days=1_000, step_days=1_000)


def test_run_walk_forward_with_param_grids_selects_and_applies_best_params_per_fold(trending_up_df):
    config = make_config()
    grids = [ParamGrid(strategy_name="sma_crossover", params={"fast_window": [5, 10]})]

    result = run_walk_forward(
        config,
        {"UP": trending_up_df},
        train_days=60,
        test_days=30,
        step_days=30,
        param_grids=grids,
        max_workers=1,  # déterministe et rapide pour le test
    )

    assert len(result.folds) > 0
    for fold in result.folds:
        assert fold.best_params is not None
        assert "sma_crossover" in fold.best_params
        assert fold.best_params["sma_crossover"]["fast_window"] in (5, 10)


def test_run_walk_forward_without_param_grids_leaves_best_params_none(trending_up_df):
    config = make_config()
    result = run_walk_forward(config, {"UP": trending_up_df}, train_days=60, test_days=30, step_days=30)
    assert all(fold.best_params is None for fold in result.folds)

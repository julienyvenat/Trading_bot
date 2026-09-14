from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest.metrics import compute_metrics
from trading_bot.backtest.trades import Trade

# --- Tests existants (non-régression : mêmes assertions qu'avant l'ajout
# des métriques Sortino/Calmar/profit factor/par trade) ---------------------


def test_compute_metrics_flat_curve_has_zero_return():
    curve = pd.Series([100.0] * 30)
    metrics = compute_metrics(curve)
    assert metrics.total_return_pct == 0.0
    assert metrics.max_drawdown_pct == 0.0


def test_compute_metrics_growth_curve_positive_return():
    curve = pd.Series([100.0 * (1.001**i) for i in range(300)])
    metrics = compute_metrics(curve)
    assert metrics.total_return_pct > 0
    assert metrics.annualized_return_pct > 0
    assert metrics.sharpe_ratio > 0


def test_compute_metrics_drawdown_detected():
    curve = pd.Series([100, 110, 90, 95, 105])
    metrics = compute_metrics(curve)
    # Pic à 110, creux à 90 -> drawdown d'environ -18.2%
    assert metrics.max_drawdown_pct < -15


# --- Nouvelles métriques ----------------------------------------------------


def _make_trade(pnl: float, entry_value: float = 1000.0, direction: int = 1) -> Trade:
    return Trade(
        symbol="UP",
        direction=direction,
        entry_date=pd.Timestamp("2020-01-01"),
        exit_date=pd.Timestamp("2020-01-10"),
        entry_price=10.0,
        exit_price=10.0 + pnl / (entry_value / 10.0),
        qty=entry_value / 10.0,
        entry_value=entry_value,
        pnl=pnl,
        exit_reason="rebalance",
    )


def test_sortino_ratio_ignores_upside_volatility():
    # Rendements alternant fortement à la hausse (aucune baisse) : la
    # volatilité "downside" est nulle, donc le Sortino doit rester à 0 (pas
    # de division par une volatilité négative artificiellement gonflée par
    # des hausses), contrairement au Sharpe qui, lui, la pénaliserait.
    curve = pd.Series([100.0 * (1 + 0.05 * (i % 2)) for i in range(60)])
    metrics = compute_metrics(curve)
    assert metrics.sortino_ratio == 0.0


def test_sortino_ratio_positive_on_steady_growth_with_small_dips():
    rng_values = [100.0]
    for i in range(200):
        rng_values.append(rng_values[-1] * (1.002 if i % 5 != 0 else 0.999))
    curve = pd.Series(rng_values)
    metrics = compute_metrics(curve)
    assert metrics.sortino_ratio > 0


def test_calmar_ratio_zero_when_no_drawdown():
    curve = pd.Series([100.0 * (1.001**i) for i in range(300)])
    metrics = compute_metrics(curve)
    # Croissance strictement monotone : pas de drawdown, mais compute_metrics
    # doit rester robuste (pas de division par zéro) même si un epsilon de
    # drawdown apparaît par arrondi ; dans tous les cas le ratio est fini.
    assert metrics.calmar_ratio >= 0


def test_calmar_ratio_relates_return_to_drawdown():
    curve = pd.Series([100, 110, 90, 130, 140])
    metrics = compute_metrics(curve)
    expected = metrics.annualized_return_pct / abs(metrics.max_drawdown_pct)
    assert metrics.calmar_ratio == pytest.approx(expected)


def test_profit_factor_from_trades_overrides_equity_approximation():
    curve = pd.Series([100.0, 105.0, 103.0, 108.0])
    trades = [_make_trade(pnl=100.0), _make_trade(pnl=-40.0)]
    metrics = compute_metrics(curve, trades=trades)
    # profit factor = gains bruts / pertes brutes = 100 / 40
    assert metrics.profit_factor == pytest.approx(2.5)


def test_profit_factor_falls_back_to_equity_when_no_trades_given():
    curve = pd.Series([100.0, 110.0, 90.0, 120.0])
    metrics = compute_metrics(curve)  # pas de `trades` fourni
    assert metrics.profit_factor > 0


def test_trade_stats_computed_from_winning_and_losing_trades():
    curve = pd.Series([100.0, 105.0, 110.0])
    trades = [
        _make_trade(pnl=200.0, entry_value=1000.0),  # +20%
        _make_trade(pnl=-100.0, entry_value=1000.0),  # -10%
        _make_trade(pnl=50.0, entry_value=1000.0),  # +5%
    ]
    metrics = compute_metrics(curve, trades=trades)

    assert metrics.num_trades == 3
    assert metrics.trade_win_rate_pct == pytest.approx(2 / 3 * 100)
    assert metrics.avg_win_pct == pytest.approx((20.0 + 5.0) / 2)
    assert metrics.avg_loss_pct == pytest.approx(-10.0)
    assert metrics.win_loss_ratio == pytest.approx((12.5) / 10.0)


def test_trade_stats_are_zero_when_no_trades():
    curve = pd.Series([100.0, 105.0, 110.0])
    metrics = compute_metrics(curve, trades=[])
    assert metrics.num_trades == 0
    assert metrics.trade_win_rate_pct == 0.0
    assert metrics.win_loss_ratio == 0.0


def test_summary_includes_new_metrics_sections():
    curve = pd.Series([100.0, 105.0, 110.0])
    metrics = compute_metrics(curve, trades=[_make_trade(pnl=50.0)])
    text = metrics.summary()
    assert "Sortino" in text
    assert "Calmar" in text
    assert "Profit factor" in text
    assert "trade" in text.lower()

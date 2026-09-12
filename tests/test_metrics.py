from __future__ import annotations

import pandas as pd

from trading_bot.backtest.metrics import compute_metrics


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

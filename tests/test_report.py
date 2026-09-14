from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")  # extra optionnel, voir pyproject.toml [report]

from trading_bot.backtest.engine import BacktestResult
from trading_bot.backtest.metrics import compute_metrics
from trading_bot.backtest.report import generate_html_report
from trading_bot.backtest.trades import Trade


def _make_result() -> BacktestResult:
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    equity_curve = pd.Series(100_000 + (index.dayofyear - index.dayofyear[0]) * 50.0, index=index)
    trades = [
        Trade(
            symbol="UP",
            direction=1,
            entry_date=index[0],
            exit_date=index[5],
            entry_price=10.0,
            exit_price=12.0,
            qty=100.0,
            entry_value=1000.0,
            pnl=200.0,
            exit_reason="rebalance",
        ),
        Trade(
            symbol="UP",
            direction=1,
            entry_date=index[6],
            exit_date=index[10],
            entry_price=12.0,
            exit_price=11.0,
            qty=100.0,
            entry_value=1200.0,
            pnl=-100.0,
            exit_reason="stop",
        ),
    ]
    metrics = compute_metrics(equity_curve, trades=trades)
    return BacktestResult(equity_curve=equity_curve, metrics=metrics, trades=trades)


def test_generate_html_report_writes_a_valid_standalone_file(tmp_path: Path):
    result = _make_result()
    output_path = tmp_path / "report.html"

    generate_html_report(result, output_path, title="Test de rapport")

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "<html" in content
    assert "Test de rapport" in content
    # Les métriques du résumé texte doivent être présentes.
    assert "Ratio de Sortino" in content
    # Les 3 graphiques doivent être embarqués en base64 (pas de fichiers séparés).
    assert content.count("data:image/png;base64,") == 3
    # Aucune dépendance réseau/CDN à l'ouverture.
    assert "http://" not in content
    assert "https://" not in content


def test_generate_html_report_handles_no_trades(tmp_path: Path):
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    equity_curve = pd.Series(100_000.0, index=index)
    result = BacktestResult(equity_curve=equity_curve, metrics=compute_metrics(equity_curve), trades=[])
    output_path = tmp_path / "empty_report.html"

    generate_html_report(result, output_path)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "<html" in content
    assert content.count("data:image/png;base64,") == 3

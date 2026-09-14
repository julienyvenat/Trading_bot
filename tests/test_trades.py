from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest.trades import TradeTracker


def _dt(day: int) -> pd.Timestamp:
    return pd.Timestamp("2020-01-01") + pd.Timedelta(days=day)


def test_full_open_then_full_close_realizes_expected_pnl():
    tracker = TradeTracker()
    tracker.record_fill("UP", _dt(0), old_qty=0.0, new_qty=100.0, price=10.0, commission=0.0)
    assert tracker.completed_trades == []  # ouverture seule : rien de réalisé

    tracker.record_fill("UP", _dt(5), old_qty=100.0, new_qty=0.0, price=12.0, commission=0.0, exit_reason="stop")

    assert len(tracker.completed_trades) == 1
    trade = tracker.completed_trades[0]
    assert trade.symbol == "UP"
    assert trade.direction == 1
    assert trade.qty == pytest.approx(100.0)
    assert trade.entry_price == pytest.approx(10.0)
    assert trade.exit_price == pytest.approx(12.0)
    assert trade.pnl == pytest.approx(100.0 * (12.0 - 10.0))
    assert trade.pnl_pct == pytest.approx(0.2)
    assert trade.exit_reason == "stop"


def test_commission_reduces_realized_pnl():
    tracker = TradeTracker()
    tracker.record_fill("UP", _dt(0), old_qty=0.0, new_qty=100.0, price=10.0, commission=5.0)
    tracker.record_fill("UP", _dt(1), old_qty=100.0, new_qty=0.0, price=12.0, commission=3.0)

    trade = tracker.completed_trades[0]
    # Gain brut 200 (100 * (12-10)) moins les commissions d'entrée (5) et de sortie (3).
    assert trade.pnl == pytest.approx(200.0 - 5.0 - 3.0)


def test_adding_to_position_uses_weighted_average_entry_price():
    tracker = TradeTracker()
    tracker.record_fill("UP", _dt(0), old_qty=0.0, new_qty=100.0, price=10.0, commission=0.0)
    # Ajoute 100 titres de plus à 20 : moyenne pondérée = (100*10 + 100*20) / 200 = 15
    tracker.record_fill("UP", _dt(1), old_qty=100.0, new_qty=200.0, price=20.0, commission=0.0)
    tracker.record_fill("UP", _dt(2), old_qty=200.0, new_qty=0.0, price=25.0, commission=0.0)

    trade = tracker.completed_trades[0]
    assert trade.entry_price == pytest.approx(15.0)
    assert trade.qty == pytest.approx(200.0)


def test_partial_reduction_realizes_only_the_closed_portion():
    tracker = TradeTracker()
    tracker.record_fill("UP", _dt(0), old_qty=0.0, new_qty=100.0, price=10.0, commission=0.0)
    # Réduit la position de moitié sans la flatten : doit réaliser un trade
    # sur les 50 titres vendus, en gardant les 50 restants ouverts.
    tracker.record_fill("UP", _dt(1), old_qty=100.0, new_qty=50.0, price=12.0, commission=0.0)

    assert len(tracker.completed_trades) == 1
    partial = tracker.completed_trades[0]
    assert partial.qty == pytest.approx(50.0)
    assert partial.pnl == pytest.approx(50.0 * (12.0 - 10.0))

    # Clôture du solde restant plus tard, toujours au même prix d'entrée.
    tracker.record_fill("UP", _dt(2), old_qty=50.0, new_qty=0.0, price=14.0, commission=0.0)
    assert len(tracker.completed_trades) == 2
    remainder = tracker.completed_trades[1]
    assert remainder.qty == pytest.approx(50.0)
    assert remainder.entry_price == pytest.approx(10.0)
    assert remainder.pnl == pytest.approx(50.0 * (14.0 - 10.0))


def test_direction_flip_closes_old_lot_and_opens_new_one():
    tracker = TradeTracker()
    tracker.record_fill("UP", _dt(0), old_qty=0.0, new_qty=100.0, price=10.0, commission=0.0)
    # Inversion directe long -> short en un seul fill.
    tracker.record_fill("UP", _dt(1), old_qty=100.0, new_qty=-50.0, price=11.0, commission=0.0)

    assert len(tracker.completed_trades) == 1
    closed = tracker.completed_trades[0]
    assert closed.direction == 1
    assert closed.qty == pytest.approx(100.0)
    assert closed.pnl == pytest.approx(100.0 * (11.0 - 10.0))

    # Le nouveau lot short doit être suivi : le clôturer produit un 2e trade.
    tracker.record_fill("UP", _dt(2), old_qty=-50.0, new_qty=0.0, price=9.0, commission=0.0)
    assert len(tracker.completed_trades) == 2
    short_trade = tracker.completed_trades[1]
    assert short_trade.direction == -1
    assert short_trade.pnl == pytest.approx(50.0 * (11.0 - 9.0))  # short gagnant si le prix baisse


def test_short_lived_position_never_recorded_if_still_open():
    tracker = TradeTracker()
    tracker.record_fill("UP", _dt(0), old_qty=0.0, new_qty=100.0, price=10.0, commission=0.0)
    assert tracker.completed_trades == []


def test_independent_symbols_do_not_interfere():
    tracker = TradeTracker()
    tracker.record_fill("UP", _dt(0), old_qty=0.0, new_qty=100.0, price=10.0, commission=0.0)
    tracker.record_fill("DOWN", _dt(0), old_qty=0.0, new_qty=50.0, price=20.0, commission=0.0)
    tracker.record_fill("UP", _dt(1), old_qty=100.0, new_qty=0.0, price=11.0, commission=0.0)

    assert len(tracker.completed_trades) == 1
    assert tracker.completed_trades[0].symbol == "UP"

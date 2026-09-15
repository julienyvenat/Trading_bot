from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.execution.broker_base import Position, PositionSnapshot
from trading_bot.live.trade_realization import detect_realized_trades, snapshot_positions

NOW = pd.Timestamp("2024-06-01T10:00:00Z")


def test_no_previous_position_produces_no_trade():
    current = {"TSLA": Position(symbol="TSLA", qty=10.0, market_value=1000.0, avg_entry_price=100.0)}
    trades = detect_realized_trades({}, current, {"TSLA": 100.0}, NOW)
    assert trades == []


def test_position_maintained_produces_no_trade():
    previous = {"TSLA": PositionSnapshot(qty=10.0, avg_entry_price=100.0)}
    current = {"TSLA": Position(symbol="TSLA", qty=10.0, market_value=1100.0, avg_entry_price=100.0)}
    trades = detect_realized_trades(previous, current, {"TSLA": 110.0}, NOW)
    assert trades == []


def test_position_increased_produces_no_trade():
    previous = {"TSLA": PositionSnapshot(qty=10.0, avg_entry_price=100.0)}
    current = {"TSLA": Position(symbol="TSLA", qty=15.0, market_value=1650.0, avg_entry_price=100.0)}
    trades = detect_realized_trades(previous, current, {"TSLA": 110.0}, NOW)
    assert trades == []


def test_full_close_produces_one_realized_trade():
    previous = {"TSLA": PositionSnapshot(qty=10.0, avg_entry_price=100.0)}
    current: dict[str, Position] = {}  # position entièrement fermée côté broker

    trades = detect_realized_trades(previous, current, {"TSLA": 120.0}, NOW)

    assert len(trades) == 1
    trade = trades[0]
    assert trade.symbol == "TSLA"
    assert trade.direction == 1
    assert trade.qty == 10.0
    assert trade.entry_price == 100.0
    assert trade.exit_price == 120.0
    assert trade.pnl == pytest.approx(200.0)  # 10 * (120 - 100)
    assert trade.exit_date == NOW


def test_partial_reduction_realizes_only_the_reduced_quantity():
    previous = {"TSLA": PositionSnapshot(qty=10.0, avg_entry_price=100.0)}
    current = {"TSLA": Position(symbol="TSLA", qty=4.0, market_value=480.0, avg_entry_price=100.0)}

    trades = detect_realized_trades(previous, current, {"TSLA": 120.0}, NOW)

    assert len(trades) == 1
    assert trades[0].qty == 6.0  # 10 - 4 réalisés, 4 restent ouverts
    assert trades[0].pnl == pytest.approx(6.0 * (120.0 - 100.0))


def test_direction_flip_closes_the_whole_previous_lot():
    previous = {"TSLA": PositionSnapshot(qty=10.0, avg_entry_price=100.0)}
    current = {"TSLA": Position(symbol="TSLA", qty=-5.0, market_value=-550.0, avg_entry_price=110.0)}

    trades = detect_realized_trades(previous, current, {"TSLA": 110.0}, NOW)

    assert len(trades) == 1
    assert trades[0].qty == 10.0  # tout l'ancien lot long est clôturé
    assert trades[0].direction == 1


def test_missing_exit_price_skips_the_symbol_without_crashing():
    previous = {"TSLA": PositionSnapshot(qty=10.0, avg_entry_price=100.0)}
    trades = detect_realized_trades(previous, {}, {}, NOW)
    assert trades == []


def test_short_position_closed_computes_pnl_with_correct_sign():
    previous = {"TSLA": PositionSnapshot(qty=-10.0, avg_entry_price=100.0)}
    trades = detect_realized_trades(previous, {}, {"TSLA": 80.0}, NOW)

    assert len(trades) == 1
    assert trades[0].direction == -1
    # Short clôturé à un prix plus bas que l'entrée = gain.
    assert trades[0].pnl == pytest.approx(10.0 * (100.0 - 80.0))


def test_snapshot_positions_excludes_flat_positions():
    positions = {
        "TSLA": Position(symbol="TSLA", qty=10.0, market_value=1000.0, avg_entry_price=100.0),
        "FLAT": Position(symbol="FLAT", qty=0.0, market_value=0.0, avg_entry_price=0.0),
    }
    snapshot = snapshot_positions(positions)
    assert set(snapshot) == {"TSLA"}
    assert snapshot["TSLA"] == PositionSnapshot(qty=10.0, avg_entry_price=100.0)


def test_position_snapshot_roundtrip():
    snap = PositionSnapshot(qty=5.0, avg_entry_price=42.5)
    assert PositionSnapshot.from_dict(snap.to_dict()) == snap

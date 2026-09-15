from __future__ import annotations

from datetime import date
from pathlib import Path

from trading_bot.execution.broker_base import PositionSnapshot
from trading_bot.portfolio.circuit_breaker import RiskState
from trading_bot.portfolio.stops import StopLevel
from trading_bot.state import LiveState, load_state, save_state


def test_load_state_missing_file_returns_empty(tmp_path: Path):
    state = load_state(tmp_path / "does_not_exist.json")
    assert state.trailing_stops == {}
    assert state.risk_state is None


def test_save_then_load_roundtrip(tmp_path: Path):
    path = tmp_path / "state" / "live_state.json"
    original = LiveState(
        trailing_stops={"AAPL": StopLevel(direction=1, stop_price=150.0)},
        stop_order_ids={"AAPL": "abc-123"},
        stop_order_dates={"AAPL": "2024-06-01"},
        risk_state=RiskState(
            equity_peak=110_000,
            session_start_equity=105_000,
            current_date=date(2024, 6, 1),
            daily_halted=True,
            drawdown_halted=False,
        ),
    )

    save_state(path, original)
    assert path.exists()  # les répertoires parents doivent être créés automatiquement

    loaded = load_state(path)
    assert loaded.trailing_stops["AAPL"] == StopLevel(direction=1, stop_price=150.0)
    assert loaded.stop_order_ids == {"AAPL": "abc-123"}
    # Nécessaire pour savoir qu'un stop DAY posé la veille a expiré côté
    # broker et doit être reposé (voir trading_bot.live.engine).
    assert loaded.stop_order_dates == {"AAPL": "2024-06-01"}
    assert loaded.risk_state == original.risk_state


def test_save_preserves_sticky_drawdown_halt(tmp_path: Path):
    path = tmp_path / "live_state.json"
    state = LiveState(risk_state=RiskState(equity_peak=100_000, session_start_equity=100_000, drawdown_halted=True))
    save_state(path, state)

    reloaded = load_state(path)
    assert reloaded.risk_state.drawdown_halted is True


def test_last_known_positions_roundtrip(tmp_path: Path):
    path = tmp_path / "live_state.json"
    original = LiveState(last_known_positions={"TSLA": PositionSnapshot(qty=10.0, avg_entry_price=250.5)})

    save_state(path, original)
    loaded = load_state(path)

    assert loaded.last_known_positions == {"TSLA": PositionSnapshot(qty=10.0, avg_entry_price=250.5)}

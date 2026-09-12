from __future__ import annotations

from datetime import date, timedelta

from trading_bot.config import RiskConfig
from trading_bot.portfolio.circuit_breaker import CircuitBreaker, RiskState, apply_halt, new_entries_allowed, should_flatten
from trading_bot.portfolio.risk import PositionSizing

DAY1 = date(2024, 1, 2)
DAY2 = DAY1 + timedelta(days=1)


def make_breaker(**overrides) -> CircuitBreaker:
    defaults = dict(
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
    defaults.update(overrides)
    return CircuitBreaker(RiskConfig(**defaults))


def test_no_halt_when_equity_stable():
    breaker = make_breaker()
    state = RiskState.initial(100_000, today=DAY1)
    state = breaker.update(state, 100_000, DAY1)
    assert new_entries_allowed(state)
    assert not should_flatten(state)


def test_daily_halt_triggers_on_intraday_loss():
    breaker = make_breaker(max_daily_loss_pct=0.03)
    state = RiskState.initial(100_000, today=DAY1)
    state = breaker.update(state, 96_000, DAY1)  # -4% depuis le début de séance
    assert state.daily_halted is True
    assert not new_entries_allowed(state)
    assert not should_flatten(state)  # le halt journalier ne force pas un flatten


def test_daily_halt_resets_on_new_session():
    breaker = make_breaker(max_daily_loss_pct=0.03)
    state = RiskState.initial(100_000, today=DAY1)
    state = breaker.update(state, 96_000, DAY1)
    assert state.daily_halted is True

    # Nouvelle séance : le halt journalier doit se réinitialiser.
    state = breaker.update(state, 96_500, DAY2)
    assert state.daily_halted is False
    assert state.session_start_equity == 96_500


def test_drawdown_halt_is_sticky_across_sessions():
    breaker = make_breaker(max_drawdown_pct=0.20)
    state = RiskState.initial(100_000, today=DAY1)
    state = breaker.update(state, 79_000, DAY1)  # -21% depuis le plus haut
    assert state.drawdown_halted is True
    assert should_flatten(state)

    # Même en repartant sur une nouvelle séance avec une equity stable, le
    # drawdown halt ne doit PAS se lever tout seul.
    state = breaker.update(state, 79_500, DAY2)
    assert state.drawdown_halted is True


def test_apply_halt_flattens_everything_on_drawdown():
    sizings = {"A": PositionSizing("A", target_weight=0.3, stop_loss_price=None)}
    state = RiskState(equity_peak=100_000, session_start_equity=100_000, current_date=DAY1, drawdown_halted=True)
    result = apply_halt(sizings, current_weights={"A": 0.2}, state=state)
    assert result == {}


def test_apply_halt_blocks_new_entries_but_keeps_existing():
    sizings = {
        "A": PositionSizing("A", target_weight=0.3, stop_loss_price=None),  # déjà détenu, stratégie veut augmenter
        "B": PositionSizing("B", target_weight=0.2, stop_loss_price=None),  # nouvelle position
    }
    state = RiskState(equity_peak=100_000, session_start_equity=100_000, current_date=DAY1, daily_halted=True)
    result = apply_halt(sizings, current_weights={"A": 0.15, "B": 0.0}, state=state)

    assert "B" not in result  # pas de nouvelle entrée
    assert "A" in result
    assert result["A"].target_weight == 0.15  # gelé à la taille actuelle, pas d'ajout


def test_apply_halt_allows_reducing_existing_position():
    sizings = {"A": PositionSizing("A", target_weight=0.05, stop_loss_price=None)}  # la stratégie veut réduire
    state = RiskState(equity_peak=100_000, session_start_equity=100_000, current_date=DAY1, daily_halted=True)
    result = apply_halt(sizings, current_weights={"A": 0.15}, state=state)
    assert result["A"].target_weight == 0.05  # réduction autorisée, pas gelée


def test_apply_halt_noop_when_not_halted():
    sizings = {"A": PositionSizing("A", target_weight=0.3, stop_loss_price=None)}
    state = RiskState.initial(100_000, today=DAY1)
    result = apply_halt(sizings, current_weights={"A": 0.0}, state=state)
    assert result == sizings

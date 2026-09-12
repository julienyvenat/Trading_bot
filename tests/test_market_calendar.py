from __future__ import annotations

import pandas as pd

from trading_bot.market_calendar import MarketCalendar


def test_thanksgiving_is_not_a_trading_day():
    cal = MarketCalendar("NYSE")
    assert cal.is_trading_day("2024-11-28") is False  # Thanksgiving
    assert cal.is_trading_day("2024-11-27") is True


def test_day_after_thanksgiving_has_early_close():
    cal = MarketCalendar("NYSE")
    session = cal.session_for(pd.Timestamp("2024-11-29 16:00", tz="UTC"))
    assert session is not None
    _, market_close = session
    # Fermeture anticipée à 13h ET = 18h UTC (au lieu de 21h UTC un jour normal)
    assert market_close.hour == 18


def test_is_open_now_during_regular_session():
    cal = MarketCalendar("NYSE")
    # Un mardi normal, 15h UTC = 10h ET, en pleine séance.
    assert cal.is_open_now(pd.Timestamp("2024-06-04 15:00", tz="UTC")) is True


def test_is_open_now_outside_session():
    cal = MarketCalendar("NYSE")
    assert cal.is_open_now(pd.Timestamp("2024-06-04 03:00", tz="UTC")) is False


def test_is_open_now_on_weekend():
    cal = MarketCalendar("NYSE")
    assert cal.is_open_now(pd.Timestamp("2024-06-08 15:00", tz="UTC")) is False  # samedi


def test_is_actionable_respects_close_buffer():
    cal = MarketCalendar("NYSE")
    # En juin (heure d'été), clôture à 20h00 UTC. À 19h50, il reste 10 min :
    # pas actionnable avec un buffer de 15 min, mais oui avec un buffer de 5 min.
    almost_close = pd.Timestamp("2024-06-04 19:50", tz="UTC")
    assert cal.is_actionable(almost_close, close_buffer_minutes=15) is False
    assert cal.is_actionable(almost_close, close_buffer_minutes=5) is True


def test_next_session_open_skips_weekend():
    cal = MarketCalendar("NYSE")
    friday_evening = pd.Timestamp("2024-06-07 22:00", tz="UTC")
    next_open = cal.next_session_open(friday_evening)
    assert next_open.dayofweek == 0  # lundi


def test_seconds_until_actionable_is_zero_when_actionable():
    cal = MarketCalendar("NYSE")
    mid_session = pd.Timestamp("2024-06-04 15:00", tz="UTC")
    assert cal.seconds_until_actionable(mid_session, close_buffer_minutes=15) == 0.0


def test_seconds_until_actionable_positive_outside_hours():
    cal = MarketCalendar("NYSE")
    night = pd.Timestamp("2024-06-04 03:00", tz="UTC")
    assert cal.seconds_until_actionable(night, close_buffer_minutes=15) > 0

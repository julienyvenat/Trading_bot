from __future__ import annotations

from trading_bot.portfolio.stops import StopLevel, is_triggered, update_stop


def test_update_stop_creates_initial_stop_for_new_long():
    stop = update_stop(None, direction=1, close=100.0, atr_value=2.0, atr_stop_multiple=2.0)
    assert stop is not None
    assert stop.direction == 1
    assert stop.stop_price == 96.0  # 100 - 2*2


def test_update_stop_creates_initial_stop_for_new_short():
    stop = update_stop(None, direction=-1, close=100.0, atr_value=2.0, atr_stop_multiple=2.0)
    assert stop.stop_price == 104.0  # 100 + 2*2


def test_long_stop_ratchets_up_never_down():
    stop = update_stop(None, direction=1, close=100.0, atr_value=2.0, atr_stop_multiple=2.0)  # stop=96
    # Le prix monte : le stop doit suivre (monter).
    stop = update_stop(stop, direction=1, close=110.0, atr_value=2.0, atr_stop_multiple=2.0)  # candidat=106
    assert stop.stop_price == 106.0

    # Le prix redescend un peu : le stop ne doit PAS redescendre.
    stop = update_stop(stop, direction=1, close=105.0, atr_value=2.0, atr_stop_multiple=2.0)  # candidat=101
    assert stop.stop_price == 106.0


def test_short_stop_ratchets_down_never_up():
    stop = update_stop(None, direction=-1, close=100.0, atr_value=2.0, atr_stop_multiple=2.0)  # stop=104
    stop = update_stop(stop, direction=-1, close=90.0, atr_value=2.0, atr_stop_multiple=2.0)  # candidat=94
    assert stop.stop_price == 94.0

    stop = update_stop(stop, direction=-1, close=95.0, atr_value=2.0, atr_stop_multiple=2.0)  # candidat=99
    assert stop.stop_price == 94.0


def test_update_stop_resets_on_direction_change():
    stop = update_stop(None, direction=1, close=100.0, atr_value=2.0, atr_stop_multiple=2.0)
    stop = update_stop(stop, direction=-1, close=100.0, atr_value=2.0, atr_stop_multiple=2.0)
    assert stop.direction == -1
    assert stop.stop_price == 104.0


def test_update_stop_keeps_previous_when_no_atr():
    stop = update_stop(None, direction=1, close=100.0, atr_value=2.0, atr_stop_multiple=2.0)
    unchanged = update_stop(stop, direction=1, close=105.0, atr_value=None, atr_stop_multiple=2.0)
    assert unchanged == stop


def test_is_triggered_long():
    stop = StopLevel(direction=1, stop_price=96.0)
    assert is_triggered(stop, low=95.0, high=100.0) is True
    assert is_triggered(stop, low=97.0, high=100.0) is False


def test_is_triggered_short():
    stop = StopLevel(direction=-1, stop_price=104.0)
    assert is_triggered(stop, low=100.0, high=105.0) is True
    assert is_triggered(stop, low=100.0, high=103.0) is False


def test_is_triggered_none_is_false():
    assert is_triggered(None, low=0.0, high=1000.0) is False

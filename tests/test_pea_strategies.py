from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.portfolio.stops import PctTrailingStop, entry_trail_pct, pct_stop_exit_price, validate_stop_mode
from trading_bot.strategies.buy_and_hold import BuyAndHoldStrategy
from trading_bot.strategies.dual_momentum import DualMomentumStrategy
from trading_bot.strategies.registry import STRATEGY_REGISTRY

from conftest import make_ohlcv


def test_new_strategies_are_registered():
    assert STRATEGY_REGISTRY["buy_and_hold"] is BuyAndHoldStrategy
    assert STRATEGY_REGISTRY["dual_momentum"] is DualMomentumStrategy


def test_buy_and_hold_is_always_fully_long():
    df = make_ohlcv(np.linspace(100, 50, 30))
    assert (BuyAndHoldStrategy().generate_signals(df) == 1.0).all()
    assert BuyAndHoldStrategy().generate_universe_signals({"A": df}) is None


def test_buy_and_hold_restricted_to_symbols():
    df = make_ohlcv(np.linspace(100, 110, 30))
    signals = BuyAndHoldStrategy(symbols=["A"]).generate_universe_signals({"A": df, "B": df})
    assert (signals["A"] == 1.0).all() and (signals["B"] == 0.0).all()


def _universe(n: int = 300) -> dict[str, pd.DataFrame]:
    days = np.arange(n)
    return {
        "STRONG": make_ohlcv(100 * 1.002**days),
        "WEAK": make_ohlcv(100 * 1.0005**days),
        "DOWN": make_ohlcv(100 * 0.999**days),
    }


def test_dual_momentum_holds_strongest_and_changes_only_at_month_start():
    data = _universe()
    signals = DualMomentumStrategy(lookback_window=60, top_n=1).generate_universe_signals(data)
    strong = signals["STRONG"]
    # Pas de position tant que le momentum n'est pas calculable, puis 100 % sur le plus fort.
    assert strong.iloc[:60].eq(0).all()
    assert strong.iloc[-1] == 1.0 and signals["WEAK"].iloc[-1] == 0.0 and signals["DOWN"].iloc[-1] == 0.0
    changes = strong.index[strong.diff().fillna(0) != 0]
    month_starts = set(DualMomentumStrategy.rebalance_dates(strong.index))
    assert set(changes) <= month_starts


def test_dual_momentum_top2_splits_weight_and_absolute_filter_goes_to_cash():
    data = _universe()
    signals = DualMomentumStrategy(lookback_window=60, top_n=2).generate_universe_signals(data)
    assert signals["STRONG"].iloc[-1] == pytest.approx(0.5)
    assert signals["WEAK"].iloc[-1] == pytest.approx(0.5)

    only_down = {"DOWN": data["DOWN"], "DOWN2": make_ohlcv(100 * 0.998 ** np.arange(300))}
    signals = DualMomentumStrategy(lookback_window=60, top_n=1).generate_universe_signals(only_down)
    assert signals["DOWN"].iloc[-1] == 0.0 and signals["DOWN2"].iloc[-1] == 0.0


def test_dual_momentum_risk_off_symbol_fills_empty_slots():
    days = np.arange(300)
    data = {"DOWN": make_ohlcv(100 * 0.999**days), "SAFE": make_ohlcv(np.full(300, 10.0))}
    signals = DualMomentumStrategy(lookback_window=60, top_n=1, risk_off_symbol="SAFE").generate_universe_signals(data)
    assert signals["SAFE"].iloc[-1] == 1.0 and signals["DOWN"].iloc[-1] == 0.0


def test_dual_momentum_sma_filter():
    data = {"DOWN": make_ohlcv(100 * 0.999 ** np.arange(300))}
    signals = DualMomentumStrategy(lookback_window=60, top_n=1, absolute_filter="sma", sma_window=50).generate_universe_signals(data)
    assert signals["DOWN"].iloc[-1] == 0.0


def test_dual_momentum_rejects_bad_params():
    with pytest.raises(ValueError):
        DualMomentumStrategy(absolute_filter="foo")
    with pytest.raises(ValueError):
        DualMomentumStrategy(top_n=0)


def test_pct_trailing_stop_ratchets_up_only():
    stop = PctTrailingStop(trail_pct=0.2, high_water=100.0)
    assert stop.stop_price == pytest.approx(80.0)
    assert stop.ratchet(90.0) is stop
    assert stop.ratchet(110.0).stop_price == pytest.approx(88.0)


def test_entry_trail_pct_fixed_or_from_atr():
    assert entry_trail_pct(0.15, 100.0, 2.0, 6.0) == 0.15
    assert entry_trail_pct(None, 100.0, 2.0, 6.0) == pytest.approx(0.12)
    assert entry_trail_pct(None, 100.0, None, 6.0) is None
    with pytest.raises(ValueError):
        entry_trail_pct(1.5, 100.0, 2.0, 6.0)


def test_pct_stop_exit_price_handles_gap_down():
    stop = PctTrailingStop(trail_pct=0.1, high_water=100.0)
    assert pct_stop_exit_price(stop, 95.0) == pytest.approx(90.0)
    assert pct_stop_exit_price(stop, 85.0) == pytest.approx(85.0)


def test_validate_stop_mode():
    assert validate_stop_mode("none") == "none"
    with pytest.raises(ValueError):
        validate_stop_mode("foo")

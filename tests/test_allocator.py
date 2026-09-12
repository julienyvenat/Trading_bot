from __future__ import annotations

from trading_bot.portfolio.allocator import SignalAllocator, StrategySignal, combine_signals
from trading_bot.strategies.sma_crossover import SmaCrossoverStrategy


def test_combine_signals_weighted_average():
    signals = [
        StrategySignal("a", weight=1.0, signal=1.0),
        StrategySignal("a", weight=1.0, signal=0.0),
    ]
    assert combine_signals(signals) == 0.5


def test_combine_signals_clips_negative_when_no_short():
    signals = [StrategySignal("a", weight=1.0, signal=-1.0)]
    assert combine_signals(signals, allow_short=False) == 0.0
    assert combine_signals(signals, allow_short=True) == -1.0


def test_combine_signals_empty_is_zero():
    assert combine_signals([]) == 0.0


def test_allocator_latest_target_exposures(trending_up_df, trending_down_df):
    strategy = SmaCrossoverStrategy(fast_window=10, slow_window=30)
    allocator = SignalAllocator([(strategy, 1.0)], allow_short=False)

    exposures = allocator.latest_target_exposures(
        {"UP": trending_up_df, "DOWN": trending_down_df}
    )
    assert exposures["UP"] == 1.0
    assert exposures["DOWN"] == 0.0


def test_allocator_target_exposure_series_bounded(trending_up_df):
    strategy = SmaCrossoverStrategy(fast_window=10, slow_window=30)
    allocator = SignalAllocator([(strategy, 1.0)], allow_short=False)

    series = allocator.target_exposure_series({"UP": trending_up_df})["UP"]
    assert series.between(0.0, 1.0).all()

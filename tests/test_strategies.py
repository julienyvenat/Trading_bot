from __future__ import annotations

from trading_bot.strategies.momentum_breakout import MomentumBreakoutStrategy
from trading_bot.strategies.rsi_mean_reversion import RsiMeanReversionStrategy
from trading_bot.strategies.sma_crossover import SmaCrossoverStrategy


def test_sma_crossover_goes_long_in_uptrend(trending_up_df):
    strategy = SmaCrossoverStrategy(fast_window=10, slow_window=30)
    signal = strategy.latest_signal(trending_up_df)
    assert signal == 1.0


def test_sma_crossover_flat_in_downtrend(trending_down_df):
    strategy = SmaCrossoverStrategy(fast_window=10, slow_window=30)
    signal = strategy.latest_signal(trending_down_df)
    assert signal == 0.0


def test_sma_crossover_signal_bounded(trending_up_df):
    strategy = SmaCrossoverStrategy(fast_window=10, slow_window=30)
    signals = strategy.generate_signals(trending_up_df)
    assert signals.isin([0.0, 1.0]).all()


def test_rsi_mean_reversion_flat_on_flat_market(flat_df):
    strategy = RsiMeanReversionStrategy(rsi_window=14)
    signal = strategy.latest_signal(flat_df)
    # RSI neutre en marché plat -> pas d'entrée en survente
    assert signal == 0.0


def test_momentum_breakout_goes_long_in_uptrend(trending_up_df):
    strategy = MomentumBreakoutStrategy(lookback_window=20, exit_window=10)
    signals = strategy.generate_signals(trending_up_df)
    # Une tendance haussière soutenue doit finir par déclencher un breakout long.
    assert signals.iloc[-1] == 1.0


def test_momentum_breakout_signal_bounded(trending_up_df):
    strategy = MomentumBreakoutStrategy(lookback_window=20, exit_window=10)
    signals = strategy.generate_signals(trending_up_df)
    assert signals.isin([0.0, 1.0]).all()

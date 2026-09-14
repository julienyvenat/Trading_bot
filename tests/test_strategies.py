from __future__ import annotations

from trading_bot.strategies.defensive_rotation import DefensiveRotationStrategy
from trading_bot.strategies.momentum_breakout import MomentumBreakoutStrategy
from trading_bot.strategies.relative_strength import RelativeStrengthStrategy
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


def test_rsi_mean_reversion_trend_filter_blocks_entry_in_downtrend(trending_down_df):
    """Sans historique suffisant au-dessus de la SMA 200, ou en tendance
    baissière franche, le filtre de tendance doit bloquer les entrées même si
    le RSI passe en survente."""
    strategy = RsiMeanReversionStrategy(rsi_window=14, oversold=70, trend_filter_window=50)
    # oversold=70 force presque toujours des "entrées" côté RSI seul, pour
    # isoler l'effet du filtre de tendance : en tendance baissière, le prix
    # reste sous sa SMA 50, donc le filtre doit rester à 0 malgré tout.
    signal = strategy.latest_signal(trending_down_df)
    assert signal == 0.0


def test_rsi_mean_reversion_trend_filter_disabled_keeps_old_behavior(trending_down_df):
    with_filter = RsiMeanReversionStrategy(oversold=70, trend_filter_window=None)
    signals = with_filter.generate_signals(trending_down_df)
    # oversold=70 en tendance baissière déclenche presque toujours l'entrée
    # dès que le filtre de tendance est désactivé.
    assert signals.iloc[-1] == 1.0


def test_momentum_breakout_goes_long_in_uptrend(trending_up_df):
    strategy = MomentumBreakoutStrategy(lookback_window=20, exit_window=10)
    signals = strategy.generate_signals(trending_up_df)
    # Une tendance haussière soutenue doit finir par déclencher un breakout long.
    assert signals.iloc[-1] == 1.0


def test_momentum_breakout_signal_bounded(trending_up_df):
    strategy = MomentumBreakoutStrategy(lookback_window=20, exit_window=10)
    signals = strategy.generate_signals(trending_up_df)
    assert signals.isin([0.0, 1.0]).all()


def test_relative_strength_selects_top_performer(trending_up_df, flat_df):
    strategy = RelativeStrengthStrategy(lookback_window=60, top_n=1)
    signals = strategy.generate_universe_signals({"UP": trending_up_df, "FLAT": flat_df})

    assert signals["UP"].iloc[-1] == 1.0
    assert signals["FLAT"].iloc[-1] == 0.0


def test_relative_strength_generate_signals_alone_is_neutral(trending_up_df):
    """Appelée isolément (hors allocateur), sans comparaison possible avec
    d'autres symboles, cette stratégie doit rester neutre plutôt que
    d'inventer un signal absolu qui n'a pas de sens pour elle."""
    strategy = RelativeStrengthStrategy()
    signals = strategy.generate_signals(trending_up_df)
    assert (signals == 0.0).all()


def test_defensive_rotation_goes_long_when_benchmark_bearish(trending_down_df, flat_df):
    strategy = DefensiveRotationStrategy(defensive_symbol="TLT", benchmark_symbol="SPY", sma_window=20)
    signals = strategy.generate_universe_signals({"SPY": trending_down_df, "TLT": flat_df, "OTHER": flat_df})

    assert signals["TLT"].iloc[-1] == 1.0
    assert (signals["OTHER"] == 0.0).all()  # ne s'applique qu'au symbole défensif


def test_defensive_rotation_flat_when_benchmark_bullish(trending_up_df, flat_df):
    strategy = DefensiveRotationStrategy(defensive_symbol="TLT", benchmark_symbol="SPY", sma_window=20)
    signals = strategy.generate_universe_signals({"SPY": trending_up_df, "TLT": flat_df})
    assert signals["TLT"].iloc[-1] == 0.0


def test_defensive_rotation_neutral_when_symbols_missing_from_universe():
    strategy = DefensiveRotationStrategy(defensive_symbol="TLT", benchmark_symbol="SPY")
    assert strategy.generate_universe_signals({}) is None

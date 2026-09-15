from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.config import UniverseRotationConfig
from trading_bot.portfolio.allocator import SignalAllocator, StrategySignal, combine_signals
from trading_bot.strategies.base import Strategy
from trading_bot.strategies.relative_strength import RelativeStrengthStrategy
from trading_bot.strategies.sma_crossover import SmaCrossoverStrategy


class _AlwaysLongStrategy(Strategy):
    """Stratégie factice : toujours pleinement long sur tout symbole reçu,
    utile pour isoler l'effet du masquage de rotation d'univers."""

    name = "always_long"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)


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


def test_allocator_supports_cross_sectional_strategy(trending_up_df, flat_df):
    """Une stratégie cross-sectionnelle (`generate_universe_signals`) doit se
    combiner avec les stratégies classiques par symbole exactement comme
    elles, sans traitement spécial côté appelant."""
    strategy = RelativeStrengthStrategy(lookback_window=60, top_n=1)
    allocator = SignalAllocator([(strategy, 1.0)], allow_short=False)

    exposures = allocator.latest_target_exposures({"UP": trending_up_df, "FLAT": flat_df})
    assert exposures["UP"] == 1.0
    assert exposures["FLAT"] == 0.0


def _flat_ohlcv(price: float, periods: int) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=periods, freq="B")
    close = pd.Series(price, index=index)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1_000_000.0}
    )


def test_rotation_restricts_strategy_to_its_own_candidate_pool():
    """Une stratégie avec rotation active ne doit jamais s'exposer à un
    symbole hors de son propre `candidates`, même s'il est présent dans
    `data_by_symbol` (ex: candidat d'une AUTRE stratégie, ou symbole de
    l'univers principal non listé comme candidat de celle-ci)."""
    strategy = _AlwaysLongStrategy()
    rng = np.random.default_rng(0)
    volatile = 100 + np.cumsum(rng.normal(0, 5.0, size=20))
    calm = np.full(20, 50.0)

    index = pd.date_range("2020-01-01", periods=20, freq="B")
    data_by_symbol = {
        "CANDIDATE_A": pd.DataFrame(
            {
                "open": volatile,
                "high": volatile * 1.001,
                "low": volatile * 0.999,
                "close": volatile,
                "volume": 1_000_000.0,
            },
            index=index,
        ),
        "CANDIDATE_B": _flat_ohlcv(50.0, 20),
        "NOT_A_CANDIDATE": _flat_ohlcv(75.0, 20),
    }

    rotation = UniverseRotationConfig(
        enabled=True,
        candidates=["CANDIDATE_A", "CANDIDATE_B"],
        metric="volatility",
        min_confidence=0.75,
        stability_window=1,
        lookback_window=5,
        rebalance_every=5,
    )
    allocator = SignalAllocator(
        [(strategy, 1.0)],
        allow_short=False,
        rotation_configs={"always_long": rotation},
        base_symbols=[],
    )

    series = allocator.target_exposure_series(data_by_symbol)

    # Jamais exposé à un symbole absent de son pool de candidats.
    assert (series["NOT_A_CANDIDATE"] == 0.0).all()
    # Avant historique suffisant, pas d'exposition.
    assert (series["CANDIDATE_A"].iloc[:4] == 0.0).all()
    # Une fois l'historique suffisant, seul le plus volatil des deux candidats est retenu.
    assert (series["CANDIDATE_A"].iloc[5:] == 1.0).all()
    assert (series["CANDIDATE_B"] == 0.0).all()


def test_base_symbols_isolates_non_rotation_strategy_from_other_strategies_candidates():
    """Des candidats de rotation ajoutés pour la stratégie B ne doivent pas
    fuiter dans le calcul de la stratégie A (sans rotation), même si les deux
    partagent le même `data_by_symbol` fusionné."""
    strategy_a = _AlwaysLongStrategy()
    strategy_a.name = "a"  # noqa: SLF001 - juste pour distinguer des instances dans ce test
    strategy_b = _AlwaysLongStrategy()
    strategy_b.name = "b"

    data_by_symbol = {
        "MAIN": _flat_ohlcv(100.0, 10),
        "EXTRA_CANDIDATE": _flat_ohlcv(50.0, 10),
    }

    rotation_b = UniverseRotationConfig(
        enabled=True,
        candidates=["MAIN", "EXTRA_CANDIDATE"],
        metric="volatility",
        min_confidence=0.7,  # les deux candidats sont à égalité (rang percentile 0.75 chacun) : les deux passent
        stability_window=1,
        lookback_window=1,
        rebalance_every=1,
    )
    allocator = SignalAllocator(
        [(strategy_a, 1.0), (strategy_b, 1.0)],
        allow_short=False,
        rotation_configs={"b": rotation_b},
        base_symbols=["MAIN"],
    )

    series = allocator.target_exposure_series(data_by_symbol)

    # Stratégie A (pas de rotation) : ne voit que MAIN (`base_symbols`), donc
    # ne contribue jamais à EXTRA_CANDIDATE.
    # Stratégie B (rotation, seuil atteint par les deux candidats à égalité) :
    # sélectionne les deux, donc EXTRA_CANDIDATE est exposé uniquement via B
    # (poids 0.5 avec 2 stratégies de poids égal).
    assert series["EXTRA_CANDIDATE"].iloc[-1] == 0.5
    assert series["MAIN"].iloc[-1] == 1.0

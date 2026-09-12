"""Combine les signaux de plusieurs stratégies en une allocation cible unique."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.strategies.base import Strategy


@dataclass
class StrategySignal:
    strategy_name: str
    weight: float
    signal: float  # dans [-1, 1]


def combine_signals(signals: list[StrategySignal], allow_short: bool = False) -> float:
    """Combine les signaux de plusieurs stratégies pour un symbole en une
    exposition cible unique, pondérée par le poids de chaque stratégie.

    Le résultat est la moyenne pondérée des signaux, bornée à [-1, 1] (ou
    [0, 1] si le short n'est pas autorisé).
    """
    if not signals:
        return 0.0

    total_weight = sum(s.weight for s in signals) or 1.0
    combined = sum(s.weight * s.signal for s in signals) / total_weight

    if not allow_short:
        combined = max(combined, 0.0)
    return max(-1.0, min(1.0, combined))


class SignalAllocator:
    """Calcule, pour chaque symbole, le signal combiné de toutes les stratégies actives."""

    def __init__(self, strategies_with_weights: list[tuple[Strategy, float]], allow_short: bool = False) -> None:
        self.strategies_with_weights = strategies_with_weights
        self.allow_short = allow_short

    def latest_target_exposures(self, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Calcule l'exposition cible combinée (dernier point) pour chaque symbole."""
        targets: dict[str, float] = {}
        for symbol, df in data_by_symbol.items():
            signals = [
                StrategySignal(strategy.name, weight, strategy.latest_signal(df))
                for strategy, weight in self.strategies_with_weights
            ]
            targets[symbol] = combine_signals(signals, allow_short=self.allow_short)
        return targets

    def target_exposure_series(self, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        """Calcule la série temporelle complète d'exposition cible combinée
        (utilisé pour le backtest vectorisé)."""
        result: dict[str, pd.Series] = {}
        for symbol, df in data_by_symbol.items():
            weighted_sum = None
            total_weight = sum(weight for _, weight in self.strategies_with_weights) or 1.0

            for strategy, weight in self.strategies_with_weights:
                signal = strategy.generate_signals(df) * weight
                weighted_sum = signal if weighted_sum is None else weighted_sum.add(signal, fill_value=0.0)

            combined = (weighted_sum / total_weight) if weighted_sum is not None else pd.Series(0.0, index=df.index)
            if not self.allow_short:
                combined = combined.clip(lower=0.0)
            result[symbol] = combined.clip(-1.0, 1.0)
        return result

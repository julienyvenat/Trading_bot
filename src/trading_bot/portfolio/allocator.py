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
    """Calcule, pour chaque symbole, le signal combiné de toutes les stratégies actives.

    Une stratégie est soit "par symbole" (`generate_signals(df)`, le cas
    classique), soit "cross-sectionnelle" (`generate_universe_signals`, pour
    une stratégie qui a besoin de comparer les symboles entre eux ou de
    réagir au prix d'un autre symbole — voir `Strategy.generate_universe_signals`).
    Les deux types se combinent de la même façon une fois leur signal calculé.
    """

    def __init__(self, strategies_with_weights: list[tuple[Strategy, float]], allow_short: bool = False) -> None:
        self.strategies_with_weights = strategies_with_weights
        self.allow_short = allow_short

    def _signals_by_strategy(self, data_by_symbol: dict[str, pd.DataFrame]) -> list[tuple[float, dict[str, pd.Series]]]:
        """Pour chaque stratégie active, calcule sa série de signal complète
        pour chaque symbole de l'univers (qu'elle soit par symbole ou
        cross-sectionnelle), une seule fois. Renvoie [(poids, {symbole: série})]."""
        per_strategy: list[tuple[float, dict[str, pd.Series]]] = []
        for strategy, weight in self.strategies_with_weights:
            universe_signals = strategy.generate_universe_signals(data_by_symbol)
            if universe_signals is None:
                universe_signals = {symbol: strategy.generate_signals(df) for symbol, df in data_by_symbol.items()}
            per_strategy.append((weight, universe_signals))
        return per_strategy

    def latest_target_exposures(self, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Calcule l'exposition cible combinée (dernier point) pour chaque symbole."""
        series = self.target_exposure_series(data_by_symbol)
        return {symbol: (float(s.iloc[-1]) if len(s) else 0.0) for symbol, s in series.items()}

    def target_exposure_series(self, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        """Calcule la série temporelle complète d'exposition cible combinée
        (utilisé pour le backtest vectorisé, et par `latest_target_exposures`
        en live — les stratégies étant vectorisées, calculer toute la série
        ou seulement son dernier point coûte la même chose)."""
        per_strategy = self._signals_by_strategy(data_by_symbol)
        total_weight = sum(weight for weight, _ in per_strategy) or 1.0

        result: dict[str, pd.Series] = {}
        for symbol, df in data_by_symbol.items():
            weighted_sum = None
            for weight, signals_by_symbol in per_strategy:
                signal = signals_by_symbol.get(symbol, pd.Series(0.0, index=df.index)) * weight
                weighted_sum = signal if weighted_sum is None else weighted_sum.add(signal, fill_value=0.0)

            combined = (weighted_sum / total_weight) if weighted_sum is not None else pd.Series(0.0, index=df.index)
            if not self.allow_short:
                combined = combined.clip(lower=0.0)
            result[symbol] = combined.clip(-1.0, 1.0)
        return result

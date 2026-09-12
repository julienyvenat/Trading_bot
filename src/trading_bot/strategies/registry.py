"""Registre des stratégies disponibles + fabrique à partir de la config."""

from __future__ import annotations

from trading_bot.config import StrategyConfig
from trading_bot.strategies.base import Strategy
from trading_bot.strategies.momentum_breakout import MomentumBreakoutStrategy
from trading_bot.strategies.rsi_mean_reversion import RsiMeanReversionStrategy
from trading_bot.strategies.sma_crossover import SmaCrossoverStrategy

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    SmaCrossoverStrategy.name: SmaCrossoverStrategy,
    RsiMeanReversionStrategy.name: RsiMeanReversionStrategy,
    MomentumBreakoutStrategy.name: MomentumBreakoutStrategy,
}


def register_strategy(strategy_cls: type[Strategy]) -> None:
    """Permet d'ajouter facilement une nouvelle stratégie custom au registre."""
    STRATEGY_REGISTRY[strategy_cls.name] = strategy_cls


def build_strategy(config: StrategyConfig) -> Strategy:
    if config.name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Stratégie inconnue '{config.name}'. Disponibles : {available}")
    return STRATEGY_REGISTRY[config.name](**config.params)


def build_enabled_strategies(configs: list[StrategyConfig]) -> list[tuple[Strategy, float]]:
    """Construit les stratégies actives avec leur poids (normalisé)."""
    enabled = [c for c in configs if c.enabled]
    total_weight = sum(c.weight for c in enabled) or 1.0
    return [(build_strategy(c), c.weight / total_weight) for c in enabled]

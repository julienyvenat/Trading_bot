"""Stratégie `core_satellite` : allocation passive à poids cibles fixes.

Aucun signal de marché : la décision (quand et quoi acheter/vendre) est
entièrement portée par `trading_bot.portfolio.core_satellite.plan_rebalance`,
que le backtest (`run_backtest`) et le live (`run_once`) appellent
directement dès que cette stratégie est la seule active. Cette classe sert
au registre (validation des paramètres au chargement) et renvoie, si un
appelant générique la sollicite quand même, les poids cibles comme
expositions constantes.
"""

from __future__ import annotations

import pandas as pd

from trading_bot.strategies.base import Strategy


class CoreSatelliteStrategy(Strategy):
    name = "core_satellite"

    def __init__(self, **params) -> None:
        # Import différé : `trading_bot.portfolio.core_satellite` dépend de
        # `trading_bot.execution`, qui importe (indirectement) ce registre.
        from trading_bot.portfolio.core_satellite import CoreSatelliteParams

        super().__init__(**params)
        self.config = CoreSatelliteParams.from_params(params)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)

    def generate_universe_signals(self, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        return {
            symbol: pd.Series(self.config.targets.get(symbol, 0.0), index=df.index)
            for symbol, df in data_by_symbol.items()
        }

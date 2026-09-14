"""Rotation défensive vers un actif peu corrélé aux actions — stratégie cross-sectionnelle."""

from __future__ import annotations

import pandas as pd

from trading_bot.indicators import sma
from trading_bot.strategies.base import Strategy


class DefensiveRotationStrategy(Strategy):
    """Passe long sur un actif défensif (obligations, or...) quand un indice
    de référence actions clôture sous sa moyenne mobile longue (régime
    baissier), flat sinon.

    Contrairement aux stratégies par symbole, celle-ci ne réagit PAS au prix
    du symbole sur lequel elle émet un signal : elle réagit à la tendance
    d'un *autre* symbole (`benchmark_symbol`, ex: SPY) et ne s'applique qu'au
    symbole défensif configuré (`defensive_symbol`, ex: TLT ou GLD) — signal
    nul sur tous les autres symboles de l'univers. C'est une vraie
    diversification de classe d'actif : le signal ne dépend pas du
    comportement de prix des actions suivies par les autres stratégies, mais
    de leur régime global.

    `defensive_symbol` ET `benchmark_symbol` doivent tous deux faire partie de
    `universe.symbols` dans config.yaml (sinon leurs données ne sont pas
    disponibles et la stratégie reste neutre partout).

    Attention à l'interaction avec `market.regime_filter` (voir
    `trading_bot.portfolio.regime`) : celui-ci réduit l'exposition de TOUS les
    symboles en régime baissier, y compris `defensive_symbol`, ce qui
    annulerait en grande partie l'intérêt de la rotation défensive. Pense à
    ajouter `defensive_symbol` à `market.regime_filter.exempt_symbols`.
    """

    name = "defensive_rotation"

    def __init__(self, defensive_symbol: str, benchmark_symbol: str = "SPY", sma_window: int = 200, **kwargs) -> None:
        super().__init__(
            defensive_symbol=defensive_symbol, benchmark_symbol=benchmark_symbol, sma_window=sma_window, **kwargs
        )
        self.defensive_symbol = defensive_symbol
        self.benchmark_symbol = benchmark_symbol
        self.sma_window = sma_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Cette stratégie est cross-sectionnelle : voir generate_universe_signals.
        return pd.Series(0.0, index=df.index)

    def generate_universe_signals(self, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.Series] | None:
        benchmark_df = data_by_symbol.get(self.benchmark_symbol)
        defensive_df = data_by_symbol.get(self.defensive_symbol)
        if benchmark_df is None or defensive_df is None:
            return None  # symboles absents de l'univers configuré : neutre partout

        benchmark_close = benchmark_df["close"]
        benchmark_sma = sma(benchmark_close, self.sma_window)
        is_bearish = benchmark_close <= benchmark_sma
        is_bearish[benchmark_sma.isna()] = False  # historique insuffisant -> pas de rotation défensive
        defensive_signal = is_bearish.astype(float).reindex(defensive_df.index).fillna(0.0)

        return {
            symbol: (defensive_signal if symbol == self.defensive_symbol else pd.Series(0.0, index=df.index))
            for symbol, df in data_by_symbol.items()
        }

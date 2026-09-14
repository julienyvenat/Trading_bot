"""Rotation sectorielle / force relative — stratégie cross-sectionnelle."""

from __future__ import annotations

import pandas as pd

from trading_bot.strategies.base import Strategy


class RelativeStrengthStrategy(Strategy):
    """Classe les symboles de l'univers entre eux par performance passée et ne
    reste investi que sur les `top_n` les plus forts *relativement aux
    autres*.

    Mécanisme fondamentalement différent des stratégies par symbole
    (`sma_crossover`, `rsi_mean_reversion`, `momentum_breakout`) : celles-ci
    jugent chaque symbole dans l'absolu ("est-il en tendance ?"), celle-ci les
    compare entre eux ("lequel est le plus fort ?"). Un symbole peut être en
    tendance haussière absolue mais sorti du portefeuille s'il est
    relativement plus faible que le reste de l'univers, et inversement rester
    investi en marché globalement plat s'il surperforme les autres. C'est ce
    changement de mécanisme (relatif plutôt qu'absolu) qui la rend
    effectivement décorrélée des trois autres, plutôt qu'une simple variante
    de plus du même principe.

    Nécessite au moins deux symboles dans l'univers pour être pertinente.
    """

    name = "relative_strength"

    def __init__(self, lookback_window: int = 90, top_n: int = 2, **kwargs) -> None:
        super().__init__(lookback_window=lookback_window, top_n=top_n, **kwargs)
        self.lookback_window = lookback_window
        self.top_n = top_n

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Cette stratégie est cross-sectionnelle : voir generate_universe_signals.
        # Si jamais elle était appelée isolément sur un seul symbole (aucune
        # comparaison possible), on renvoie un signal neutre par sécurité.
        return pd.Series(0.0, index=df.index)

    def generate_universe_signals(self, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        close_df = pd.DataFrame({sym: df["close"] for sym, df in data_by_symbol.items()}).sort_index().ffill()
        trailing_return = close_df / close_df.shift(self.lookback_window) - 1.0

        top_n = min(self.top_n, len(data_by_symbol))
        # Pour chaque date, en concurrence (rang <= top_n) parmi les symboles
        # disposant d'un historique suffisant ce jour-là -> exposition 1.0,
        # sinon 0.0 (pas de position, ou historique insuffisant).
        ranks = trailing_return.rank(axis=1, ascending=False, method="first")
        signals = (ranks <= top_n).astype(float)
        signals[trailing_return.isna()] = 0.0

        return {symbol: signals[symbol].reindex(df.index).fillna(0.0) for symbol, df in data_by_symbol.items()}

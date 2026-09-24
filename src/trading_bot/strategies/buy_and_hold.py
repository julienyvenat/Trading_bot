"""Buy & hold : toujours pleinement investi sur ses symboles."""

from __future__ import annotations

import pandas as pd

from trading_bot.strategies.base import Strategy


class BuyAndHoldStrategy(Strategy):
    """Signal constant à 1.0 (pleinement long) sur chaque symbole de
    l'univers, dès la première bougie disponible. Sert de référence honnête
    (et, sur un PEA, de mode d'investissement à part entière : voir
    `config/config_pea_buyhold.example.yaml`) : aucune décision de timing,
    donc quasiment aucun ordre à passer.

    `symbols` (optionnel) : restreint le signal à ces symboles (0.0 ailleurs),
    pour mélanger buy & hold et autres stratégies dans un même univers.
    """

    name = "buy_and_hold"

    def __init__(self, symbols: list[str] | None = None, **kwargs) -> None:
        super().__init__(symbols=symbols, **kwargs)
        self.symbols = list(symbols) if symbols else None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)

    def generate_universe_signals(self, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.Series] | None:
        if self.symbols is None:
            return None
        return {
            symbol: pd.Series(1.0 if symbol in self.symbols else 0.0, index=df.index)
            for symbol, df in data_by_symbol.items()
        }

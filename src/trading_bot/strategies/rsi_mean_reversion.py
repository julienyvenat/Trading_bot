"""Stratégie de retour à la moyenne basée sur le RSI."""

from __future__ import annotations

import pandas as pd

from trading_bot.indicators import rsi
from trading_bot.strategies.base import Strategy


class RsiMeanReversionStrategy(Strategy):
    """Achète en survente (RSI bas), sort en zone neutre/surachat.

    Signal binaire : long tant que le RSI n'est pas repassé au-dessus du
    seuil de surachat après être descendu sous le seuil de survente.
    """

    name = "rsi_mean_reversion"

    def __init__(
        self,
        rsi_window: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        **kwargs,
    ) -> None:
        super().__init__(rsi_window=rsi_window, oversold=oversold, overbought=overbought, **kwargs)
        self.rsi_window = rsi_window
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi_values = rsi(df["close"], self.rsi_window)

        entries = rsi_values < self.oversold
        exits = rsi_values > self.overbought

        # Construit un signal on/off : passe à 1 sur une entrée, revient à 0 sur
        # une sortie, sinon conserve l'état précédent (logique état-machine).
        state = pd.Series(0.0, index=df.index)
        position = 0.0
        values = []
        for is_entry, is_exit in zip(entries, exits, strict=True):
            if is_entry:
                position = 1.0
            elif is_exit:
                position = 0.0
            values.append(position)
        state[:] = values
        return state

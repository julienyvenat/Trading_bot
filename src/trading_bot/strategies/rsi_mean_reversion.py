"""Stratégie de retour à la moyenne basée sur le RSI."""

from __future__ import annotations

import pandas as pd

from trading_bot.indicators import rsi, sma
from trading_bot.strategies.base import Strategy


class RsiMeanReversionStrategy(Strategy):
    """Achète en survente (RSI bas), sort en zone neutre/surachat.

    Signal binaire : long tant que le RSI n'est pas repassé au-dessus du
    seuil de surachat après être descendu sous le seuil de survente.

    Filtre de tendance (`trend_filter_window`) : une entrée en survente n'est
    prise que si le prix est au-dessus de sa moyenne mobile longue. Sans ce
    filtre, la stratégie achète *tout* creux de RSI, y compris en pleine
    tendance baissière ("attraper un couteau qui tombe") — ce qui est
    précisément le régime où le retour à la moyenne fonctionne le moins bien
    (le prix continue de baisser au lieu de rebondir). Avec le filtre, on ne
    cherche des rebonds que dans un marché déjà haussier sur le fond, ce qui
    est le contexte où le "buy the dip" a historiquement le meilleur ratio
    gain/risque. Une fois en position, la sortie (`overbought`) reste
    inconditionnelle : le filtre ne s'applique qu'aux nouvelles entrées, pas
    aux sorties. Mettre `trend_filter_window=None` pour désactiver le filtre
    et retrouver le comportement d'origine.
    """

    name = "rsi_mean_reversion"

    def __init__(
        self,
        rsi_window: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        trend_filter_window: int | None = 200,
        **kwargs,
    ) -> None:
        super().__init__(
            rsi_window=rsi_window,
            oversold=oversold,
            overbought=overbought,
            trend_filter_window=trend_filter_window,
            **kwargs,
        )
        self.rsi_window = rsi_window
        self.oversold = oversold
        self.overbought = overbought
        self.trend_filter_window = trend_filter_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi_values = rsi(df["close"], self.rsi_window)

        entries = rsi_values < self.oversold
        exits = rsi_values > self.overbought

        if self.trend_filter_window:
            trend_sma = sma(df["close"], self.trend_filter_window)
            # Historique insuffisant (trend_sma NaN) -> filtre non concluant,
            # on n'entre pas (même logique prudente que sma_crossover/momentum_breakout).
            in_uptrend = df["close"] > trend_sma
            entries = entries & in_uptrend

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

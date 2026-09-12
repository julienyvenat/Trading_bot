"""Stratégie de suivi de tendance : croisement de moyennes mobiles (SMA)."""

from __future__ import annotations

import pandas as pd

from trading_bot.indicators import sma
from trading_bot.strategies.base import Strategy


class SmaCrossoverStrategy(Strategy):
    """Long quand la SMA rapide est au-dessus de la SMA lente, flat sinon.

    Stratégie de suivi de tendance classique : simple, robuste, bonne base
    pour un système multi-stratégies.
    """

    name = "sma_crossover"

    def __init__(self, fast_window: int = 20, slow_window: int = 50, **kwargs) -> None:
        super().__init__(fast_window=fast_window, slow_window=slow_window, **kwargs)
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = sma(df["close"], self.fast_window)
        slow = sma(df["close"], self.slow_window)

        signal = pd.Series(0.0, index=df.index)
        signal[fast > slow] = 1.0
        signal[fast <= slow] = 0.0
        signal[fast.isna() | slow.isna()] = float("nan")
        return signal.ffill().fillna(0.0)

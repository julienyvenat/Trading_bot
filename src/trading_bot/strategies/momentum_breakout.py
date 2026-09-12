"""Stratégie de breakout de momentum (type turtle / canal de Donchian)."""

from __future__ import annotations

import pandas as pd

from trading_bot.indicators import rolling_max, rolling_min
from trading_bot.strategies.base import Strategy


class MomentumBreakoutStrategy(Strategy):
    """Entre long sur un nouveau plus-haut à N jours, sort sur un plus-bas à M jours.

    `lookback_window` (N) doit être > `exit_window` (M) pour créer une bande
    de sortie plus proche du prix que l'entrée (moins de mouvement requis
    pour sortir que pour entrer), ce qui limite le nombre d'allers-retours.
    """

    name = "momentum_breakout"

    def __init__(self, lookback_window: int = 55, exit_window: int = 20, **kwargs) -> None:
        super().__init__(lookback_window=lookback_window, exit_window=exit_window, **kwargs)
        self.lookback_window = lookback_window
        self.exit_window = exit_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        entry_level = rolling_max(close, self.lookback_window).shift(1)
        exit_level = rolling_min(close, self.exit_window).shift(1)

        entries = close >= entry_level
        exits = close <= exit_level

        state = pd.Series(0.0, index=df.index)
        position = 0.0
        values = []
        for is_entry, is_exit, has_data in zip(
            entries, exits, ~(entry_level.isna() | exit_level.isna()), strict=True
        ):
            if not has_data:
                values.append(0.0)
                continue
            if is_entry:
                position = 1.0
            elif is_exit:
                position = 0.0
            values.append(position)
        state[:] = values
        return state

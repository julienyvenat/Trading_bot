"""Classe de base pour toutes les stratégies de trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """Une stratégie transforme un historique OHLCV en un signal de position.

    Le signal est une série alignée sur l'index du DataFrame d'entrée, avec des
    valeurs dans [-1, 1] représentant l'exposition cible désirée par la
    stratégie pour ce symbole :
      1.0  -> pleinement long
      0.0  -> flat / pas de position
     -1.0  -> pleinement short

    Les stratégies sont volontairement *vectorisées* (calculent le signal sur
    toute la série) : le même code sert au backtest et au live (on prend alors
    juste la dernière valeur).
    """

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Calcule le signal d'exposition cible pour un symbole.

        `df` doit contenir au moins les colonnes open/high/low/close/volume,
        indexées par date croissante.
        """
        raise NotImplementedError

    def generate_universe_signals(self, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.Series] | None:
        """Optionnel : pour une stratégie *cross-sectionnelle* qui a besoin de
        comparer les symboles entre eux (ex: rotation sectorielle/force
        relative) ou de réagir au prix d'un *autre* symbole (ex: rotation
        défensive pilotée par un indice de référence), calcule directement un
        signal par symbole à partir de tout l'univers en une fois.

        Renvoie `None` par défaut : une stratégie "classique" par symbole
        (la majorité) n'a pas besoin de surcharger cette méthode, l'allocateur
        utilisera alors `generate_signals` symbole par symbole comme
        d'habitude.
        """
        return None

    def latest_signal(self, df: pd.DataFrame) -> float:
        """Renvoie le signal du dernier point disponible (utilisé en live)."""
        signals = self.generate_signals(df)
        if signals.empty:
            return 0.0
        value = signals.iloc[-1]
        return 0.0 if pd.isna(value) else float(value)

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        return f"{self.__class__.__name__}({self.params})"

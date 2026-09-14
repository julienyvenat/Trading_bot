"""Stratégie de scalping par retour à la moyenne sur bandes de Bollinger.

Pensée pour des bougies INTRADAY (typiquement 5 min, voir `universe.timeframe`
dans `config/config_scalping.example.yaml`) plutôt que quotidiennes comme les
autres stratégies du bot : `window` est un nombre de bougies, pas de jours,
et les entrées/sorties visent des mouvements rapides et fréquents plutôt
qu'un mouvement ample sur plusieurs séances.

Logique : entre quand le prix clôture sous la bande inférieure (anormalement
bas par rapport à sa moyenne/volatilité récente), ressort dès que le prix
retrouve la bande médiane — pas besoin d'attendre la bande supérieure : viser
des gains modestes mais fréquents est le principe même du scalping, attendre
un mouvement ample irait à l'encontre de l'objectif (et laisserait plus de
temps à un retournement de tendance d'invalider le pari de retour à la
moyenne).

Filtre de volatilité optionnel (`min_band_width_pct`) : ignore les entrées
quand les bandes sont trop resserrées (marché quasi plat), signe d'un
mouvement trop faible pour couvrir les coûts de transaction (commission +
spread), qui pèsent proportionnellement bien plus lourd sur des gains visés
petits et fréquents qu'en swing trading quotidien.
"""

from __future__ import annotations

import pandas as pd

from trading_bot.indicators import bollinger_bands
from trading_bot.strategies.base import Strategy


class BollingerScalpingStrategy(Strategy):
    name = "bollinger_scalping"

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        min_band_width_pct: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(window=window, num_std=num_std, min_band_width_pct=min_band_width_pct, **kwargs)
        self.window = window
        self.num_std = num_std
        self.min_band_width_pct = min_band_width_pct

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        mid, upper, lower = bollinger_bands(close, self.window, self.num_std)

        band_width_pct = (upper - lower) / mid
        enough_volatility = band_width_pct >= self.min_band_width_pct

        entries = (close < lower) & enough_volatility
        exits = close >= mid

        # Construit un signal on/off : passe à 1 sur une entrée, revient à 0 sur
        # une sortie, sinon conserve l'état précédent (même logique état-machine
        # que rsi_mean_reversion).
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

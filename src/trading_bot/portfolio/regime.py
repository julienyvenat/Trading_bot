"""Filtre de régime de marché.

Réduit l'exposition du portefeuille quand un indice de référence (par défaut
SPY) clôture sous sa moyenne mobile longue, pour atténuer les pertes en
marché baissier généralisé. Le bot est long-only et suit des signaux
individuels par symbole (tendance/retour à la moyenne/momentum) : sans ce
filtre, rien ne l'empêche de rester pleinement investi pendant une baisse
large du marché — constaté sur le backtest 2022 (rendement -16%, Sharpe
-1.71) avant l'ajout de ce filtre.

Volontairement une réduction d'exposition (`bearish_exposure_scale`) plutôt
qu'un flatten total à 0 par défaut : permet de garder un peu d'exposition si
on estime que les signaux individuels restent pertinents, tout en réduisant
le risque global. Mettre `bearish_exposure_scale` à 0.0 pour un flatten total
en régime baissier.

Limite connue : un simple seuil SMA peut "whipsaw" (bascules répétées) si le
prix oscille autour de sa moyenne mobile ; pas de zone morte ni de
confirmation multi-jours pour l'instant.
"""

from __future__ import annotations

import pandas as pd

from trading_bot.indicators import sma


def regime_scale_series(benchmark_close: pd.Series, sma_window: int, bearish_scale: float) -> pd.Series:
    """Facteur multiplicatif à appliquer aux expositions cibles, par date.

    Vaut 1.0 en régime haussier (clôture > SMA) et `bearish_scale` en régime
    baissier (clôture <= SMA). Avant que la SMA ne soit calculable (historique
    insuffisant), vaut 1.0 (pas de filtre, on ne pénalise pas le début de
    l'historique faute de recul suffisant).
    """
    benchmark_sma = sma(benchmark_close, sma_window)
    scale = pd.Series(1.0, index=benchmark_close.index)
    scale[benchmark_close <= benchmark_sma] = bearish_scale
    scale[benchmark_sma.isna()] = 1.0
    return scale


def latest_regime_scale(benchmark_close: pd.Series, sma_window: int, bearish_scale: float) -> float:
    """Facteur multiplicatif pour le cycle live courant (dernier point connu)."""
    if len(benchmark_close) == 0:
        return 1.0
    series = regime_scale_series(benchmark_close, sma_window, bearish_scale)
    return float(series.iloc[-1])

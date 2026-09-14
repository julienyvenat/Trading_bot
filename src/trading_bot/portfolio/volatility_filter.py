"""Filtre de volatilité.

Réduit l'exposition du portefeuille quand un proxy de volatilité négociable
(par défaut VIXY, qui réplique des futures VIX court terme) s'envole
au-dessus de sa moyenne mobile récente, signe d'un pic de stress de marché
où les stratégies de tendance/retour à la moyenne tendent à mal performer.

Contrairement au filtre de régime (`trading_bot.portfolio.regime`), qui
réagit à la DIRECTION du marché (tendance haussière/baissière sur une longue
fenêtre), celui-ci réagit à l'AMPLITUDE des mouvements récents : les deux
filtres se combinent (multiplicativement, voir `trading_bot.backtest.engine`
et `trading_bot.live.engine`) plutôt que de se remplacer, un pic de
volatilité pouvant survenir même en tendance haussière (ex. un choc macro
soudain).

Volontairement basé sur un ETF négociable (VIXY) plutôt que l'indice ^VIX
lui-même : Alpaca ne fournit pas de données de marché pour les indices,
seulement pour les actions/ETF, donc ^VIX serait inutilisable en live (il ne
fonctionnerait qu'en backtest via yfinance) alors que VIXY se comporte à
l'identique dans les deux environnements.
"""

from __future__ import annotations

import pandas as pd

from trading_bot.indicators import sma


def volatility_scale_series(
    proxy_close: pd.Series, sma_window: int, spike_threshold_pct: float, spike_scale: float
) -> pd.Series:
    """Facteur multiplicatif à appliquer aux expositions cibles, par date.

    Vaut `spike_scale` quand `proxy_close` dépasse sa SMA de plus de
    `spike_threshold_pct` (pic de volatilité), 1.0 sinon. Avant que la SMA ne
    soit calculable (historique insuffisant), vaut 1.0 (pas de pénalité faute
    de recul suffisant).
    """
    proxy_sma = sma(proxy_close, sma_window)
    ratio = proxy_close / proxy_sma - 1.0
    scale = pd.Series(1.0, index=proxy_close.index)
    scale[ratio > spike_threshold_pct] = spike_scale
    scale[proxy_sma.isna()] = 1.0
    return scale


def latest_volatility_scale(
    proxy_close: pd.Series, sma_window: int, spike_threshold_pct: float, spike_scale: float
) -> float:
    """Facteur multiplicatif pour le cycle live courant (dernier point connu)."""
    if len(proxy_close) == 0:
        return 1.0
    series = volatility_scale_series(proxy_close, sma_window, spike_threshold_pct, spike_scale)
    return float(series.iloc[-1])

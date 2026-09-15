"""Rotation périodique de l'univers tradé par UNE stratégie.

Complète le dimensionnement du risque et les filtres de régime/volatilité :
là où ceux-ci modulent l'AMPLEUR de l'exposition à un univers fixe, ce module
décide QUELS symboles, parmi un pool de candidats propre à une stratégie
(`UniverseRotationConfig.candidates`), sont effectivement éligibles à cette
stratégie pour la période courante — pour éviter de rester indéfiniment sur
des titres devenus peu intéressants (ex: une volatilité qui s'est tarie)
simplement parce qu'ils figuraient dans la config au départ.

La sélection se fait par SEUIL de confiance (`min_confidence`), pas par
compte fixe : le nombre de candidats retenus varie naturellement dans le
temps selon les opportunités disponibles, plutôt que de forcer un nombre
constant de positions même quand peu de candidats sont réellement
intéressants (voir `compute_confidence`). Le nombre de positions réellement
ouvertes reste de toute façon borné en aval par
`risk.max_open_positions`/`max_gross_exposure_pct`
(`trading_bot.portfolio.risk.RiskManager.apply_portfolio_caps`).

Utilisé par `trading_bot.portfolio.allocator.SignalAllocator` : le signal
d'une stratégie ayant une rotation active est multiplié, symbole par
symbole, par le résultat de `compute_membership` avant d'être combiné aux
autres stratégies.
"""

from __future__ import annotations

import pandas as pd

from trading_bot.config import UniverseRotationConfig

SUPPORTED_METRICS = ("volatility", "momentum", "dollar_volume")


def _compute_metric(close_df: pd.DataFrame, volume_df: pd.DataFrame | None, metric: str, lookback_window: int) -> pd.DataFrame:
    """Un score brut par symbole et par date, où "plus haut = plus
    intéressant" dans les 3 cas (permet un classement percentile uniforme,
    voir `compute_confidence`)."""
    if metric == "volatility":
        # Écart-type des rendements : favorise les titres qui bougent le
        # plus, pertinent pour une stratégie de scalping / retour à la
        # moyenne qui a besoin d'amplitude pour dépasser ses coûts.
        return close_df.pct_change().rolling(window=lookback_window, min_periods=lookback_window).std(ddof=0)
    if metric == "momentum":
        # Rendement glissant : pertinent pour une stratégie de suivi de tendance.
        return close_df / close_df.shift(lookback_window) - 1.0
    if metric == "dollar_volume":
        # Liquidité (prix x volume) : écarte les titres trop étroits pour
        # être tradés proprement, indépendamment de leur mouvement de prix.
        if volume_df is None:
            raise ValueError("La métrique 'dollar_volume' nécessite une colonne 'volume'.")
        return (close_df * volume_df).rolling(window=lookback_window, min_periods=lookback_window).mean()
    raise ValueError(f"Métrique de rotation inconnue '{metric}'. Valeurs supportées : {', '.join(SUPPORTED_METRICS)}.")


def _checkpoint_index(full_index: pd.DatetimeIndex, rebalance_every: int) -> pd.DatetimeIndex:
    positions = range(0, len(full_index), max(1, rebalance_every))
    return full_index[list(positions)]


def compute_confidence(candidates_data: dict[str, pd.DataFrame], config: UniverseRotationConfig) -> dict[str, pd.Series]:
    """Score de confiance par candidat, dans [0, 1] (`NaN` pendant le
    warmup, avant `stability_window` réévaluations complètes).

    À chaque point de réévaluation (tous les `rebalance_every` bougies), un
    candidat est classé par PERCENTILE parmi les autres candidats sur
    `metric` (1.0 = le meilleur du pool ce jour-là, proche de 0 = le pire).
    Le score de confiance retenu est la MOYENNE de ce percentile sur les
    `stability_window` dernières réévaluations, pas la dernière seule : un
    candidat qui n'aurait été intéressant qu'une fois par hasard (ex: un pic
    de volatilité isolé causé par une actualité ponctuelle) n'obtient une
    confiance élevée que s'il l'est resté de façon répétée.

    Entre deux réévaluations, la confiance est maintenue telle quelle
    (forward-fill), comme le reste de la mécanique de rotation.

    Pas de biais de lookahead : le percentile à la date de réévaluation T
    n'utilise que les données jusqu'à T inclus — même convention que le
    reste du moteur (signal calculé à la clôture de T, exécuté à
    l'ouverture de T+1, voir `trading_bot.backtest.engine`).
    """
    if not candidates_data:
        return {}

    close_df = pd.DataFrame({sym: df["close"] for sym, df in candidates_data.items()}).sort_index().ffill()
    volume_df = None
    if config.metric == "dollar_volume":
        volume_df = pd.DataFrame({sym: df["volume"] for sym, df in candidates_data.items()}).sort_index().ffill()

    metric_df = _compute_metric(close_df, volume_df, config.metric, config.lookback_window)
    percentile_df = metric_df.rank(axis=1, pct=True, ascending=True)

    checkpoints = _checkpoint_index(close_df.index, config.rebalance_every)
    stability_window = max(1, config.stability_window)
    percentile_at_checkpoints = percentile_df.loc[checkpoints]
    confidence_at_checkpoints = percentile_at_checkpoints.rolling(
        window=stability_window, min_periods=stability_window
    ).mean()

    confidence = pd.DataFrame(index=close_df.index, columns=close_df.columns, dtype=float)
    confidence.loc[checkpoints] = confidence_at_checkpoints
    confidence = confidence.ffill()

    return {symbol: confidence[symbol].reindex(df.index) for symbol, df in candidates_data.items()}


def compute_membership(candidates_data: dict[str, pd.DataFrame], config: UniverseRotationConfig) -> dict[str, pd.Series]:
    """Calcule, pour chaque symbole candidat, une série (alignée sur son
    propre index) valant 1.0 quand sa confiance (voir `compute_confidence`)
    atteint `config.min_confidence`, 0.0 sinon (y compris pendant le warmup,
    où la confiance est encore `NaN`)."""
    confidence_by_symbol = compute_confidence(candidates_data, config)
    return {symbol: (series >= config.min_confidence).astype(float) for symbol, series in confidence_by_symbol.items()}

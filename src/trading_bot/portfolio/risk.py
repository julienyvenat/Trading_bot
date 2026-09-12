"""Gestion du risque : transforme des expositions cibles en tailles de position.

Applique les règles définies dans config.yaml (risk:) :
  - risque max par trade, dimensionné via l'ATR (distance de stop)
  - poids max par position
  - exposition brute max du portefeuille
  - nombre max de positions ouvertes simultanément

Le calcul par symbole (`_size_single`) est une fonction pure (pas d'accès à un
DataFrame) afin de pouvoir être réutilisée telle quelle :
  - en live, à partir de la dernière bougie connue (`size_positions`)
  - en backtest, à partir de séries précalculées jour par jour, sans
    recalculer l'ATR à chaque itération (voir `trading_bot.backtest.engine`)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.config import RiskConfig
from trading_bot.indicators import atr


@dataclass
class PositionSizing:
    symbol: str
    target_weight: float  # part de l'equity totale à allouer à ce symbole, signée
    stop_loss_price: float | None


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def size_single(self, symbol: str, exposure: float, last_close: float, last_atr: float | None) -> PositionSizing:
        """Calcule le poids cible pour UN symbole, sans appliquer les caps
        globaux de portefeuille (gross exposure / max positions) — ceux-ci
        sont appliqués séparément sur l'ensemble des positions candidates,
        voir `size_positions` et `trading_bot.backtest.engine`.
        """
        risk_based_weight = self.config.max_position_weight_pct
        stop_price = None

        if last_atr and last_atr > 0 and last_close > 0:
            stop_distance = last_atr * self.config.atr_stop_multiple
            direction = 1.0 if exposure > 0 else -1.0
            stop_price = last_close - direction * stop_distance
            # weight * equity * (stop_distance / last_close) = risk_per_trade_pct * equity
            risk_based_weight = self.config.risk_per_trade_pct * last_close / stop_distance

        weight = min(self.config.max_position_weight_pct, risk_based_weight)
        weight *= abs(exposure)  # la conviction du signal module la taille
        signed_weight = weight if exposure > 0 else -weight
        return PositionSizing(symbol=symbol, target_weight=signed_weight, stop_loss_price=stop_price)

    def apply_portfolio_caps(self, sizings: dict[str, PositionSizing]) -> dict[str, PositionSizing]:
        """Applique max_open_positions puis max_gross_exposure_pct sur un jeu
        de PositionSizing candidats déjà dimensionnés individuellement."""
        ranked = sorted(sizings.values(), key=lambda s: abs(s.target_weight), reverse=True)
        kept = {s.symbol: s for s in ranked[: self.config.max_open_positions]}

        gross = sum(abs(s.target_weight) for s in kept.values())
        if gross > self.config.max_gross_exposure_pct and gross > 0:
            scale = self.config.max_gross_exposure_pct / gross
            for sizing in kept.values():
                sizing.target_weight *= scale

        return kept

    def size_positions(
        self,
        target_exposures: dict[str, float],
        data_by_symbol: dict[str, pd.DataFrame],
    ) -> dict[str, PositionSizing]:
        """Convertit des expositions cibles ([-1,1] par symbole) en poids de
        portefeuille concrets, en respectant les contraintes de risque.
        Utilisé en live, à partir de la dernière bougie connue de chaque symbole.
        """
        candidates = {sym: exp for sym, exp in target_exposures.items() if abs(exp) > 1e-9}

        sizings: dict[str, PositionSizing] = {}
        for symbol, exposure in candidates.items():
            df = data_by_symbol.get(symbol)
            last_close = float(df["close"].iloc[-1]) if df is not None and len(df) > 0 else 0.0
            last_atr = None
            if df is not None and len(df) > self.config.atr_window:
                last_atr = float(atr(df, self.config.atr_window).iloc[-1])

            sizings[symbol] = self.size_single(symbol, exposure, last_close, last_atr)

        return self.apply_portfolio_caps(sizings)

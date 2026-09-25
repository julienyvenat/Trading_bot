"""Modèle de frais de courtage et garde-fous d'exécution (actions entières,
cash disponible, ordres trop petits pour valoir leurs frais).

Partagé par le backtest (`trading_bot.backtest.engine`) et le live
(`trading_bot.live.engine`, broker manuel) pour que ce qui est testé soit ce
qui est tradé : même barème, mêmes arrondis, mêmes ordres ignorés.

Barème par paliers (`backtest.commission_schedule`) : chaque palier couvre
les ordres de valeur <= `up_to` (None = sans limite, dernier palier) et
facture `fixed + pct x valeur`, avec un plancher `min`. Exemple, barème
standard Fortuneo sur Euronext (tarifs 2026, à revérifier sur la grille
tarifaire en vigueur) :

    tiers:
      - {up_to: 500, pct: 0.005, min: 1.95}   # <= 500 € : 0,50 %, min 1,95 €
      - {up_to: 2000, fixed: 1.95}            # 500-2000 € : 1,95 € fixe
      - {up_to: null, pct: 0.002}             # > 2000 € : 0,20 %

Sans barème, repli sur `commission_pct` proportionnel pur (comportement
historique inchangé).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from trading_bot.config import BacktestConfig


@dataclass(frozen=True)
class FeeTier:
    up_to: float | None
    pct: float = 0.0
    fixed: float = 0.0
    min: float = 0.0


@dataclass(frozen=True)
class CommissionModel:
    """Calcule les frais d'un ordre à partir de sa valeur absolue."""

    pct: float = 0.0
    tiers: tuple[FeeTier, ...] = field(default_factory=tuple)
    minimum: float = 0.0

    @classmethod
    def from_schedule(cls, schedule: dict[str, Any] | None, fallback_pct: float = 0.0) -> CommissionModel:
        if not schedule:
            return cls(pct=fallback_pct)
        raw_tiers = schedule.get("tiers") or []
        tiers = tuple(
            FeeTier(
                up_to=None if t.get("up_to") is None else float(t["up_to"]),
                pct=float(t.get("pct", 0.0)),
                fixed=float(t.get("fixed", 0.0)),
                min=float(t.get("min", 0.0)),
            )
            for t in raw_tiers
        )
        bounded = [t.up_to for t in tiers if t.up_to is not None]
        if bounded != sorted(bounded) or any(t.up_to is None for t in tiers[:-1]):
            raise ValueError(
                "`commission_schedule.tiers` doit être trié par `up_to` croissant, seul le dernier palier "
                "pouvant avoir `up_to: null`."
            )
        return cls(pct=fallback_pct, tiers=tiers, minimum=float(schedule.get("minimum", 0.0)))

    @classmethod
    def from_backtest_config(cls, config: BacktestConfig) -> CommissionModel:
        return cls.from_schedule(config.commission_schedule, config.commission_pct)

    def fee(self, notional: float) -> float:
        value = abs(notional)
        if value <= 0:
            return 0.0
        if not self.tiers:
            return max(value * self.pct, self.minimum) if self.minimum else value * self.pct
        tier = next((t for t in self.tiers if t.up_to is None or value <= t.up_to), self.tiers[-1])
        return max(tier.fixed + tier.pct * value, tier.min, self.minimum)


@dataclass(frozen=True)
class ExecutionRules:
    """Contraintes d'exécution communes backtest/live (voir `BacktestConfig`)."""

    commission: CommissionModel = field(default_factory=CommissionModel)
    whole_shares: bool = False
    min_order_value: float = 0.0
    max_fee_pct: float | None = None
    rebalance_tolerance_pct: float = 0.0

    @classmethod
    def from_backtest_config(cls, config: BacktestConfig) -> ExecutionRules:
        return cls(
            commission=CommissionModel.from_backtest_config(config),
            whole_shares=config.whole_shares,
            min_order_value=config.min_order_value,
            max_fee_pct=config.max_fee_pct,
            rebalance_tolerance_pct=config.rebalance_tolerance_pct,
        )


def max_affordable_qty(cash: float, price: float, commission: CommissionModel, whole_shares: bool) -> float:
    """Plus grande quantité achetable avec `cash`, frais compris."""
    if cash <= 0 or price <= 0:
        return 0.0
    qty = cash / price
    if whole_shares:
        qty = math.floor(qty)
        while qty > 0 and qty * price + commission.fee(qty * price) > cash + 1e-9:
            qty -= 1
        return float(max(qty, 0))
    # Fractionnaire : quelques itérations suffisent (frais ~linéaires par palier).
    for _ in range(20):
        total = qty * price + commission.fee(qty * price)
        if total <= cash + 1e-9:
            break
        qty = max(0.0, qty - (total - cash) / price)
    return qty


def decide_order_qty(
    current_qty: float,
    target_qty: float,
    price: float,
    equity: float,
    cash_available: float | None,
    rules: ExecutionRules,
) -> tuple[float, str | None]:
    """Nouvelle quantité à détenir après l'ordre (et raison si l'ordre est
    ignoré ou réduit), à partir d'une quantité cible "idéale" éventuellement
    fractionnaire. Renvoie (quantité finale, raison) : quantité finale ==
    `current_qty` si aucun ordre ne doit être passé.

    - actions entières : cible arrondie vers zéro ;
    - `cash_available` (None = pas de contrainte, ex: marge Alpaca) : achat
      plafonné au cash disponible frais compris ;
    - une clôture complète (cible 0) passe toujours, sans garde-fou de frais ;
    - sinon : ignoré si écart de poids < `rebalance_tolerance_pct` (sauf
      entrée depuis 0), valeur < `min_order_value`, ou frais > `max_fee_pct`.
    """
    if rules.whole_shares:
        target_qty = float(math.floor(target_qty)) if target_qty >= 0 else float(math.ceil(target_qty))

    delta = target_qty - current_qty
    if delta > 0 and cash_available is not None:
        affordable = max_affordable_qty(cash_available, price, rules.commission, rules.whole_shares)
        if delta > affordable:
            delta = affordable
            target_qty = current_qty + delta

    if abs(delta) < 1e-12:
        return current_qty, None

    full_exit = abs(target_qty) < 1e-12 and abs(current_qty) > 0
    if full_exit:
        return target_qty, None

    notional = abs(delta) * price
    is_entry = abs(current_qty) < 1e-12
    if not is_entry and equity > 0 and rules.rebalance_tolerance_pct > 0:
        if notional / equity < rules.rebalance_tolerance_pct:
            return current_qty, (
                f"écart de poids {notional / equity:.1%} < tolérance de rééquilibrage "
                f"{rules.rebalance_tolerance_pct:.0%}"
            )
    if rules.min_order_value > 0 and notional < rules.min_order_value:
        return current_qty, f"valeur {notional:.2f} € < min_order_value {rules.min_order_value:.2f} €"
    if rules.max_fee_pct is not None:
        fee = rules.commission.fee(notional)
        if fee > rules.max_fee_pct * notional:
            return current_qty, f"frais {fee:.2f} € > {rules.max_fee_pct:.1%} de la valeur ({notional:.2f} €)"
    return target_qty, None

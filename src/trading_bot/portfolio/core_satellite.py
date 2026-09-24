"""Allocation passive cœur-satellite (mode `core_satellite`) : poids cibles
fixes par poche (ex: 55 % MSCI World, 20 % S&P 500, 25 % levier x2 USA),
aucun signal de marché, aucun stop.

Logique de décision, partagée à l'identique par le backtest
(`trading_bot.backtest.core_satellite`) et le live (`trading_bot.live.
engine`, broker manuel) — `plan_rebalance` est une fonction pure :

1. Rien à faire (aucun ordre) tant que :
   - aucune poche ne dérive de plus de `drift_threshold_pts` points de son
     poids cible (poids = valeur de la poche / valeur totale, cash compris) ;
   - on n'est pas à la date de rééquilibrage annuel (premier jour de bourse
     du mois `annual_rebalance_month`, voir `calendar_rebalance_due`) ;
   - le cash non investi (au-delà du coussin `cash_buffer_*`) reste sous le
     seuil d'apport max(`contribution_min_eur`, `contribution_min_pct` x
     valeur totale).
2. Sinon, on investit d'abord le cash disponible dans les poches en retard
   (« rééquilibrage par les apports », achats uniquement : pas de vente,
   donc pas de frais de vente ni de fiscalité à la sortie du PEA). Le cash
   va à la poche la plus en retard, puis à la suivante quand elles sont au
   même niveau de remplissage (remplissage « par niveau »).
3. On ne vend que si la dérive dépasse ENCORE le seuil après ces achats (ou,
   au rééquilibrage annuel, si elle dépasse encore `calendar_min_drift_pts`) :
   les poches en excès sont ramenées à leur cible, le produit des ventes
   finance les poches en retard.

Contraintes d'exécution (`trading_bot.portfolio.fees`) : actions entières,
achats plafonnés au cash frais compris, ordres sous `min_order_value` ou
dont les frais dépassent `max_fee_pct` écartés (le cash correspondant est
reporté sur les autres poches, ou attend le prochain apport).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_bot.execution.rebalancer import PlannedOrder
from trading_bot.portfolio.fees import ExecutionRules, max_affordable_qty

STRATEGY_NAME = "core_satellite"


@dataclass(frozen=True)
class CoreSatelliteParams:
    targets: dict[str, float]
    drift_threshold_pts: float = 5.0
    # Mois (1-12) du rééquilibrage calendaire annuel, exécuté au premier jour
    # de bourse de ce mois. None : pas de rééquilibrage calendaire.
    annual_rebalance_month: int | None = 1
    # Au rééquilibrage annuel, dérive résiduelle en dessous de laquelle on ne
    # vend rien (évite de payer des frais pour corriger 0,3 point).
    calendar_min_drift_pts: float = 1.0
    contribution_min_eur: float = 200.0
    contribution_min_pct: float = 0.01
    # Coussin de cash jamais investi : max(`cash_buffer_eur`, `cash_buffer_pct`
    # x cash disponible). Absorbe l'écart entre le cours de la veille (qui
    # dimensionne l'ordre) et le cours d'exécution du lendemain.
    cash_buffer_eur: float = 10.0
    cash_buffer_pct: float = 0.01

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> CoreSatelliteParams:
        raw = dict(params or {})
        targets = {str(k): float(v) for k, v in (raw.pop("targets", None) or {}).items()}
        if not targets:
            raise ValueError("core_satellite : `params.targets` (poids cible par symbole) est obligatoire.")
        if any(w < 0 for w in targets.values()):
            raise ValueError("core_satellite : les poids cibles doivent être positifs ou nuls.")
        total = sum(targets.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"core_satellite : la somme des poids cibles doit valoir 1 (actuellement {total:.4f}).")
        month = raw.get("annual_rebalance_month", 1)
        if month is not None and not 1 <= int(month) <= 12:
            raise ValueError("core_satellite : `annual_rebalance_month` doit être entre 1 et 12, ou null.")
        unknown = set(raw) - {f for f in cls.__dataclass_fields__ if f != "targets"}
        if unknown:
            raise ValueError(f"core_satellite : paramètre(s) inconnu(s) : {', '.join(sorted(unknown))}.")
        if month is not None:
            raw["annual_rebalance_month"] = int(month)
        return cls(targets=targets, **raw)

    def cash_buffer(self, cash: float) -> float:
        return min(max(cash, 0.0), max(self.cash_buffer_eur, self.cash_buffer_pct * cash))

    def contribution_threshold(self, equity: float) -> float:
        return max(self.contribution_min_eur, self.contribution_min_pct * equity)


def core_satellite_params(config) -> CoreSatelliteParams | None:
    """Paramètres du mode cœur-satellite si c'est la stratégie active de
    `config` (AppConfig), None sinon. Ce mode gère seul tout le portefeuille :
    le combiner à d'autres stratégies est refusé."""
    enabled = config.enabled_strategies()
    names = [s.name for s in enabled]
    if STRATEGY_NAME not in names:
        return None
    if len(enabled) > 1:
        raise ValueError("La stratégie core_satellite gère tout le portefeuille : elle doit être la seule active.")
    params = CoreSatelliteParams.from_params(enabled[0].params)
    missing = set(params.targets) - set(config.symbols)
    if missing:
        raise ValueError(f"core_satellite : poche(s) absente(s) de universe.symbols : {', '.join(sorted(missing))}.")
    return params


@dataclass
class RebalancePlan:
    orders: list[PlannedOrder] = field(default_factory=list)
    # "contribution" (cash investi, achats seuls), "drift" (dérive > seuil),
    # "calendar" (rééquilibrage annuel) ; None si rien à faire.
    reason: str | None = None
    sells: bool = False
    weights_before: dict[str, float] = field(default_factory=dict)
    weights_after: dict[str, float] = field(default_factory=dict)
    cash_before: float = 0.0
    cash_after: float = 0.0
    fees: float = 0.0
    max_drift_before_pts: float = 0.0


def weights_of(values: dict[str, float], cash: float) -> dict[str, float]:
    """Poids de chaque poche dans la valeur totale (cash compris)."""
    total = cash + sum(values.values())
    if total <= 0:
        return dict.fromkeys(values, 0.0)
    return {s: v / total for s, v in values.items()}


def max_drift_pts(weights: dict[str, float], targets: dict[str, float]) -> float:
    return max((abs(weights.get(s, 0.0) - t) * 100 for s, t in targets.items()), default=0.0)


def _fill_level(values: dict[str, float], targets: dict[str, float], cash: float) -> dict[str, float]:
    """Remplissage par niveau : montant à ajouter à chaque poche pour amener
    les plus en retard au même taux de remplissage valeur/cible, en
    dépensant exactement `cash` (achats seulement, aucune poche ne baisse)."""
    eligible = sorted((s for s in targets if targets[s] > 0), key=lambda s: values.get(s, 0.0) / targets[s])
    if cash <= 0 or not eligible:
        return {}
    for k in range(1, len(eligible) + 1):
        group = eligible[:k]
        level = (cash + sum(values.get(s, 0.0) for s in group)) / sum(targets[s] for s in group)
        nxt = eligible[k] if k < len(eligible) else None
        if nxt is None or level <= values.get(nxt, 0.0) / targets[nxt]:
            return {s: max(0.0, targets[s] * level - values.get(s, 0.0)) for s in group}
    return {}  # pragma: no cover - la boucle renvoie toujours


def _order_ok(notional: float, rules: ExecutionRules) -> bool:
    if notional <= 0:
        return False
    if rules.min_order_value > 0 and notional < rules.min_order_value:
        return False
    if rules.max_fee_pct is not None and rules.commission.fee(notional) > rules.max_fee_pct * notional:
        return False
    return True


def allocate_cash(
    values: dict[str, float],
    prices: dict[str, float],
    targets: dict[str, float],
    cash: float,
    rules: ExecutionRules,
) -> dict[str, float]:
    """Quantités à ACHETER par poche pour investir au mieux `cash` (frais
    compris) dans les poches en retard. Une poche dont l'achat serait trop
    petit (min_order_value / max_fee_pct) est écartée et son montant reporté
    sur les autres ; le reliquat d'arrondi (actions entières) complète, une
    action à la fois, la poche la plus en retard déjà achetée."""
    eligible = {s: t for s, t in targets.items() if t > 0 and prices.get(s, 0) > 0}
    buys: dict[str, float] = {}
    while eligible:
        amounts = _fill_level(values, eligible, cash)
        buys = {}
        rejected = None
        for symbol, amount in sorted(amounts.items(), key=lambda kv: -kv[1]):
            if amount <= 0:
                continue
            qty = max_affordable_qty(amount, prices[symbol], rules.commission, rules.whole_shares)
            if qty <= 0 or not _order_ok(qty * prices[symbol], rules):
                rejected = symbol if rejected is None or amount < amounts[rejected] else rejected
                continue
            buys[symbol] = qty
        if rejected is None:
            break
        # On écarte la plus petite poche refusée et on recalcule : son montant
        # revient aux autres (sinon ce cash dormirait jusqu'au prochain apport).
        eligible.pop(rejected)
    if not eligible:
        return {}

    def spent(b: dict[str, float]) -> float:
        return sum(q * prices[s] + rules.commission.fee(q * prices[s]) for s, q in b.items())

    step = 1.0 if rules.whole_shares else 0.0
    if step:
        while True:
            left = cash - spent(buys)
            candidates = []
            for symbol, qty in buys.items():
                price = prices[symbol]
                extra = (qty + 1) * price + rules.commission.fee((qty + 1) * price) - (
                    qty * price + rules.commission.fee(qty * price)
                )
                if extra <= left + 1e-9:
                    fill = (values.get(symbol, 0.0) + qty * price) / targets[symbol]
                    candidates.append((fill, symbol))
            if not candidates:
                break
            _, symbol = min(candidates)
            buys[symbol] += 1
        buys = _refine_whole_shares(values, prices, targets, cash, rules, buys, spent)
    return buys


def _refine_whole_shares(values, prices, targets, cash, rules, buys, spent) -> dict[str, float]:
    """Recherche locale autour de la solution arrondie : quelques actions de
    plus ou de moins par poche, pour la combinaison finançable (frais
    compris, garde-fous respectés) qui colle le mieux aux cibles (somme des
    écarts au carré), puis celle qui laisse le moins de cash dormir."""
    symbols = list(buys)
    total = sum(values.values()) + cash
    best_key, best = None, buys
    ranges = [range(max(0, int(buys[s]) - 3), int(buys[s]) + 2) for s in symbols]
    for combo in itertools.product(*ranges):
        candidate = {s: float(q) for s, q in zip(symbols, combo) if q > 0}
        if any(not _order_ok(q * prices[s], rules) for s, q in candidate.items()):
            continue
        cost = spent(candidate)
        if cost > cash + 1e-9:
            continue
        deviation = sum(
            (values.get(s, 0.0) + candidate.get(s, 0.0) * prices[s] - t * total) ** 2 for s, t in targets.items()
        )
        key = (round(deviation, 6), round(cash - cost, 6))
        if best_key is None or key < best_key:
            best_key, best = key, candidate
    return best


def _sell_down(
    values: dict[str, float], prices: dict[str, float], targets: dict[str, float], total: float, rules: ExecutionRules
) -> dict[str, float]:
    """Quantités à VENDRE pour ramener chaque poche en excès à sa cible
    (arrondi vers le bas : on reste juste au-dessus de la cible)."""
    sells: dict[str, float] = {}
    for symbol, target in targets.items():
        price = prices.get(symbol, 0.0)
        excess = values.get(symbol, 0.0) - target * total
        if price <= 0 or excess <= 0:
            continue
        qty = math.floor(excess / price) if rules.whole_shares else excess / price
        if qty > 0 and _order_ok(qty * price, rules):
            sells[symbol] = float(qty)
    return sells


def plan_rebalance(
    qty: dict[str, float],
    prices: dict[str, float],
    cash: float,
    params: CoreSatelliteParams,
    rules: ExecutionRules,
    calendar_due: bool = False,
) -> RebalancePlan:
    """Ordres à passer (ventes d'abord) pour ce cycle, voir la docstring du
    module. Aucune donnée n'est modifiée."""
    targets = params.targets
    values = {s: qty.get(s, 0.0) * prices[s] for s in targets}
    equity = cash + sum(values.values())
    weights = weights_of(values, cash)
    plan = RebalancePlan(weights_before=weights, weights_after=weights, cash_before=cash, cash_after=cash)
    if equity <= 0:
        return plan
    plan.max_drift_before_pts = drift = max_drift_pts(weights, targets)
    deployable = max(0.0, cash - params.cash_buffer(cash))
    drift_due = drift > params.drift_threshold_pts + 1e-9
    contribution_due = deployable >= params.contribution_threshold(equity) - 1e-9
    if not (drift_due or calendar_due or contribution_due):
        return plan

    def finish(sells: dict[str, float], buys: dict[str, float], reason: str) -> RebalancePlan:
        new_values = dict(values)
        new_cash = cash
        orders: list[PlannedOrder] = []
        fees = 0.0
        for side, book in (("sell", sells), ("buy", buys)):
            for symbol in targets:
                q = book.get(symbol, 0.0)
                if q <= 0:
                    continue
                notional = q * prices[symbol]
                fee = rules.commission.fee(notional)
                fees += fee
                sign = 1 if side == "buy" else -1
                new_values[symbol] += sign * notional
                new_cash -= sign * notional + fee
                orders.append(PlannedOrder(symbol=symbol, side=side, qty=q, notional_value=notional))
        plan.orders = orders
        plan.reason = reason if orders else None
        plan.sells = bool(sells)
        plan.weights_after = weights_of(new_values, new_cash)
        plan.cash_after = new_cash
        plan.fees = fees
        return plan

    # 1) Achats seuls avec le cash disponible.
    buys = allocate_cash(values, prices, targets, deployable, rules)
    after = {s: values[s] + buys.get(s, 0.0) * prices[s] for s in targets}
    cash_left = cash - sum(q * prices[s] + rules.commission.fee(q * prices[s]) for s, q in buys.items())
    residual = max_drift_pts(weights_of(after, cash_left), targets)
    if drift_due:
        band = params.drift_threshold_pts
    elif calendar_due:
        band = params.calendar_min_drift_pts
    else:
        band = math.inf
    if residual <= band + 1e-9:
        reason = "drift" if drift_due and not contribution_due else ("contribution" if buys else "calendar")
        return finish({}, buys, reason)

    # 2) La dérive persiste : ventes des poches en excès, puis achats.
    investable = equity - params.cash_buffer(cash)
    sells = _sell_down(values, prices, targets, investable, rules)
    post_values = {s: values[s] - sells.get(s, 0.0) * prices[s] for s in targets}
    proceeds = sum(q * prices[s] - rules.commission.fee(q * prices[s]) for s, q in sells.items())
    buys = allocate_cash(post_values, prices, targets, deployable + proceeds, rules)
    return finish(sells, buys, "drift" if drift_due else "calendar")


def first_trading_day_of_month(calendar, year: int, month: int) -> pd.Timestamp | None:
    start = pd.Timestamp(year=year, month=month, day=1)
    days = calendar.trading_days(start, start + pd.offsets.MonthEnd(0))
    return days[0] if len(days) else None


def calendar_rebalance_due(
    params: CoreSatelliteParams, today: pd.Timestamp, last_year: int | None, calendar
) -> bool:
    """Rééquilibrage annuel dû : on a atteint (ou dépassé, si un cycle a été
    manqué) le premier jour de bourse du mois configuré cette année, et il
    n'a pas encore été fait cette année."""
    month = params.annual_rebalance_month
    if month is None:
        return False
    today = pd.Timestamp(today).normalize()
    if last_year is not None and last_year >= today.year:
        return False
    first = first_trading_day_of_month(calendar, today.year, month)
    return first is not None and today >= first

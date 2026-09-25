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
dont les frais dépassent `max_fee_pct` écartés. Une poche n'est jamais
achetée au-delà de son écart à la cible (+ 1 action d'arrondi), ni si elle
est déjà à sa cible ; parmi les combinaisons d'achats finançables, on
retient celle qui colle le mieux aux cibles (voir `allocate_cash`). Un plan
qui ne réduit pas la dérive n'est jamais émis : le cash attend alors le
prochain apport (raison "waiting").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
    # "contribution" (apport : cash investi, achats seuls), "drift" (dérive
    # des poches > seuil), "calendar" (rééquilibrage annuel), "waiting" (cash
    # à investir mais aucun achat possible ne rapproche des cibles : il
    # attend le prochain apport) ; None si rien à faire.
    reason: str | None = None
    sells: bool = False
    weights_before: dict[str, float] = field(default_factory=dict)
    weights_after: dict[str, float] = field(default_factory=dict)
    cash_before: float = 0.0
    cash_after: float = 0.0
    fees: float = 0.0
    max_drift_before_pts: float = 0.0
    max_drift_after_pts: float = 0.0
    waiting_cash: float = 0.0


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


# Taille max de la recherche exhaustive des combinaisons d'achats en actions
# entières ; au-delà (gros montants), recherche locale de ±LOCAL_RADIUS
# actions par poche autour du remplissage par niveau.
EXHAUSTIVE_MAX_COMBOS = 60_000
LOCAL_RADIUS = 12


def _candidates(
    gap: float, center: int, price: float, cash: float, rules: ExecutionRules, radius: int | None
) -> list[int]:
    """Quantités envisageables pour une poche : 0, ou de la plus petite
    quantité qui vaut ses frais jusqu'à l'écart à la cible + 1 action (jamais
    au-delà), dans la limite du cash."""
    if gap <= 0 or price <= 0:
        return [0]
    cap = min(int(math.floor(gap / price)) + 1, int(max_affordable_qty(cash, price, rules.commission, True)))
    low = max(1, int(math.ceil(rules.min_order_value / price)) if rules.min_order_value > 0 else 1)
    while low <= cap and not _order_ok(low * price, rules):
        low += 1
    if low > cap:
        return [0]
    if radius is not None:
        base = max(center, low)
        low, cap = max(low, base - radius), min(cap, base + radius)
    return [0, *range(low, cap + 1)]


def allocate_cash(
    values: dict[str, float],
    prices: dict[str, float],
    targets: dict[str, float],
    cash: float,
    rules: ExecutionRules,
    cash_total: float | None = None,
) -> dict[str, float]:
    """Quantités à ACHETER par poche pour investir au mieux `cash` (frais
    compris) sans jamais dépasser la cible d'une poche de plus d'une action
    ni acheter une poche déjà à sa cible (cible = poids x (valeur investie +
    `cash`)). `cash_total` : cash réellement détenu (>= `cash`, coussin
    compris), pour mesurer les poids après achat.

    Actions entières : parmi les combinaisons finançables qui respectent les
    garde-fous (min_order_value, max_fee_pct), on retient celle qui minimise
    la dérive max après achat (poids cash compris), puis la somme des écarts
    au carré, puis le cash laissé. « Ne rien acheter » fait partie des
    candidates : si aucun achat ne rapproche des cibles, le cash attend."""
    cash_total = cash if cash_total is None else cash_total
    investable = sum(values.values()) + cash
    gaps = {s: t * investable - values.get(s, 0.0) for s, t in targets.items()}
    eligible = [s for s, t in targets.items() if t > 0 and prices.get(s, 0) > 0 and gaps[s] > 0]
    if cash <= 0 or not eligible:
        return {}
    ideal = _fill_level({s: values.get(s, 0.0) for s in eligible}, {s: targets[s] for s in eligible}, cash)
    ideal = {s: min(a, gaps[s]) for s, a in ideal.items()}

    if not rules.whole_shares:
        buys = {}
        for s, amount in ideal.items():
            qty = max_affordable_qty(amount, prices[s], rules.commission, False)
            if qty > 0 and _order_ok(qty * prices[s], rules):
                buys[s] = qty
        return buys

    centers = {s: int(math.floor(ideal.get(s, 0.0) / prices[s])) for s in eligible}
    cands = [_candidates(gaps[s], centers[s], prices[s], cash, rules, None) for s in eligible]
    if math.prod(len(c) for c in cands) > EXHAUSTIVE_MAX_COMBOS:
        cands = [_candidates(gaps[s], centers[s], prices[s], cash, rules, LOCAL_RADIUS) for s in eligible]
    grids = np.meshgrid(*[np.array(c, dtype=float) for c in cands], indexing="ij")
    qty = [g.ravel() for g in grids]
    cost = np.zeros_like(qty[0])
    fees = np.zeros_like(qty[0])
    for s, c, q in zip(eligible, cands, qty):
        fee_of = {n: rules.commission.fee(n * prices[s]) for n in c}
        f = np.vectorize(fee_of.get)(q) if len(c) > 1 else np.zeros_like(q)
        fees += f
        cost += q * prices[s] + f
    equity_after = sum(values.values()) + cash_total - fees
    drift = np.zeros_like(cost)
    sq = np.zeros_like(cost)
    added = dict(zip(eligible, qty))
    for s, t in targets.items():
        value = values.get(s, 0.0) + (added[s] * prices[s] if s in added else 0.0)
        drift = np.maximum(drift, np.abs(value / equity_after - t))
        sq += (value - t * equity_after) ** 2
    feasible = cost <= cash + 1e-9
    if not feasible.any():
        return {}
    idx = np.flatnonzero(feasible)
    order = np.lexsort((cash - cost[idx], np.round(sq[idx], 4), np.round(drift[idx], 9)))
    best = idx[order[0]]
    return {s: float(q[best]) for s, q in zip(eligible, qty) if q[best] > 0}


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


def _apply(values, cash, prices, rules, sells, buys):
    new_values = dict(values)
    new_cash = cash
    fees = 0.0
    for side, book in (("sell", sells), ("buy", buys)):
        for symbol, q in book.items():
            notional = q * prices[symbol]
            fee = rules.commission.fee(notional)
            fees += fee
            sign = 1 if side == "buy" else -1
            new_values[symbol] += sign * notional
            new_cash -= sign * notional + fee
    return new_values, new_cash, fees


def plan_rebalance(
    qty: dict[str, float],
    prices: dict[str, float],
    cash: float,
    params: CoreSatelliteParams,
    rules: ExecutionRules,
    calendar_due: bool = False,
) -> RebalancePlan:
    """Ordres à passer (ventes d'abord) pour ce cycle, voir la docstring du
    module. Aucune donnée n'est modifiée. Garantie : la dérive max après
    ordres (poids cash compris) n'est jamais pire qu'avant ; sinon aucun
    ordre."""
    targets = params.targets
    values = {s: qty.get(s, 0.0) * prices[s] for s in targets}
    equity = cash + sum(values.values())
    weights = weights_of(values, cash)
    plan = RebalancePlan(weights_before=weights, weights_after=weights, cash_before=cash, cash_after=cash)
    if equity <= 0:
        return plan
    plan.max_drift_before_pts = plan.max_drift_after_pts = drift = max_drift_pts(weights, targets)
    deployable = max(0.0, cash - params.cash_buffer(cash))
    drift_due = drift > params.drift_threshold_pts + 1e-9
    contribution_due = deployable >= params.contribution_threshold(equity) - 1e-9
    if not (drift_due or calendar_due or contribution_due):
        return plan

    def outcome(sells, buys):
        new_values, new_cash, fees = _apply(values, cash, prices, rules, sells, buys)
        return max_drift_pts(weights_of(new_values, new_cash), targets), new_values, new_cash, fees

    # 1) Achats seuls avec le cash disponible.
    buys = allocate_cash(values, prices, targets, deployable, rules, cash_total=cash)
    sells: dict[str, float] = {}
    residual, new_values, new_cash, fees = outcome({}, buys)
    band = (
        params.drift_threshold_pts if drift_due else (params.calendar_min_drift_pts if calendar_due else math.inf)
    )
    # 2) La dérive persiste : ventes des poches en excès, puis achats. Retenu
    # seulement si le résultat est meilleur que les achats seuls.
    if residual > band + 1e-9:
        sell_q = _sell_down(values, prices, targets, equity - params.cash_buffer(cash), rules)
        if sell_q:
            post_values = {s: values[s] - sell_q.get(s, 0.0) * prices[s] for s in targets}
            proceeds = sum(q * prices[s] - rules.commission.fee(q * prices[s]) for s, q in sell_q.items())
            buys2 = allocate_cash(post_values, prices, targets, deployable + proceeds, rules, cash_total=cash + proceeds)
            candidate = outcome(sell_q, buys2)
            if candidate[0] < residual - 1e-9:
                sells, buys = sell_q, buys2
                residual, new_values, new_cash, fees = candidate

    if (not sells and not buys) or residual > drift + 1e-9:
        # Rien d'utile (ou pire qu'avant) : aucun ordre, le cash attend.
        if contribution_due:
            plan.reason, plan.waiting_cash = "waiting", deployable
        return plan

    orders = [
        PlannedOrder(symbol=s, side=side, qty=book[s], notional_value=book[s] * prices[s])
        for side, book in (("sell", sells), ("buy", buys))
        for s in targets
        if book.get(s, 0.0) > 0
    ]
    invested_drift = max_drift_pts(weights_of(values, 0.0), targets) if sum(values.values()) > 0 else 0.0
    if sells:
        reason = "drift" if drift_due else "calendar"
    elif contribution_due or (drift_due and invested_drift <= params.drift_threshold_pts + 1e-9):
        reason = "contribution"  # la « dérive » ne vient que du cash non investi : c'est un apport
    elif drift_due:
        reason = "drift"
    else:
        reason = "calendar"
    plan.orders, plan.reason, plan.sells = orders, reason, bool(sells)
    plan.weights_after = weights_of(new_values, new_cash)
    plan.cash_after, plan.fees, plan.max_drift_after_pts = new_cash, fees, residual
    return plan


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

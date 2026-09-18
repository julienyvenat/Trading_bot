"""Traduit des poids de portefeuille cibles en ordres, puis les soumet au broker."""

from __future__ import annotations

import time
from dataclasses import dataclass

from trading_bot.execution.broker_base import Broker
from trading_bot.logger import get_logger
from trading_bot.portfolio.risk import PositionSizing

logger = get_logger()

# En dessous de ce delta (en valeur, $), on ne passe pas d'ordre : évite le
# bruit de micro-rebalancements pour des variations négligeables.
MIN_ORDER_VALUE = 1.0

# Nombre de tentatives et délai avant retry en cas de rejet transitoire par
# Alpaca ("potential wash trade detected") : `run_once`
# (`trading_bot.live.engine`) annule le stop natif d'un symbole juste avant
# de soumettre son ordre de rebalancement, mais cette annulation n'est pas
# forcément déjà reflétée côté broker au moment de la soumission — Alpaca
# voit alors encore l'ancien stop comme un ordre ouvert de sens opposé et
# rejette le nouvel ordre.
_ORDER_MAX_ATTEMPTS = 3
_ORDER_RETRY_DELAY_SECONDS = 2.0


@dataclass
class PlannedOrder:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    notional_value: float


def plan_orders(
    target_sizings: dict[str, PositionSizing],
    current_positions: dict[str, float],  # symbol -> qty (signé)
    equity: float,
    last_prices: dict[str, float],
) -> list[PlannedOrder]:
    """Calcule les ordres nécessaires pour passer des positions actuelles aux
    positions cibles (targets exprimés en poids d'equity, signés)."""
    orders: list[PlannedOrder] = []

    all_symbols = set(target_sizings) | set(current_positions)
    for symbol in all_symbols:
        price = last_prices.get(symbol)
        if not price or price <= 0:
            logger.warning("Pas de prix disponible pour %s, ordre ignoré.", symbol)
            continue

        target_weight = target_sizings[symbol].target_weight if symbol in target_sizings else 0.0
        target_qty = (target_weight * equity) / price
        current_qty = current_positions.get(symbol, 0.0)

        delta_qty = target_qty - current_qty
        delta_value = abs(delta_qty) * price
        if delta_value < MIN_ORDER_VALUE:
            continue

        side = "buy" if delta_qty > 0 else "sell"
        orders.append(PlannedOrder(symbol=symbol, side=side, qty=abs(delta_qty), notional_value=delta_value))

    return orders


def execute_orders(orders: list[PlannedOrder], broker: Broker, dry_run: bool = False) -> None:
    """Soumet chaque ordre planifié. Un rejet sur UN symbole (voir
    `_submit_market_order_with_retry`) ne doit jamais empêcher la soumission
    des AUTRES ordres du cycle, ni la suite du cycle côté appelant (pose des
    stops, mise à jour du suivi par symbole...)."""
    for order in orders:
        logger.info(
            "%s %s %.4f (%s $%.2f)",
            "DRY-RUN" if dry_run else "ORDRE",
            order.side.upper(),
            order.qty,
            order.symbol,
            order.notional_value,
        )
        if not dry_run:
            _submit_market_order_with_retry(order, broker)


def _submit_market_order_with_retry(order: PlannedOrder, broker: Broker) -> None:
    last_error: Exception | None = None
    for attempt in range(1, _ORDER_MAX_ATTEMPTS + 1):
        try:
            broker.submit_market_order(order.symbol, order.qty, order.side)
            return
        except Exception as exc:  # noqa: BLE001 - on catégorise via retry, pas via type
            last_error = exc
            if attempt < _ORDER_MAX_ATTEMPTS:
                logger.warning(
                    "Échec de l'ordre de rebalancement sur %s (tentative %d/%d), nouvelle tentative dans %.0fs : %s",
                    order.symbol,
                    attempt,
                    _ORDER_MAX_ATTEMPTS,
                    _ORDER_RETRY_DELAY_SECONDS,
                    exc,
                )
                time.sleep(_ORDER_RETRY_DELAY_SECONDS)

    logger.error(
        "Impossible de soumettre l'ordre de rebalancement sur %s après %d tentatives, ordre abandonné ce cycle : %s",
        order.symbol,
        _ORDER_MAX_ATTEMPTS,
        last_error,
    )

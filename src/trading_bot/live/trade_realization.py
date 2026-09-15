"""Détection des trades RÉALISÉS en live, par comparaison de deux
instantanés de positions — l'équivalent, côté live, de
`trading_bot.backtest.trades.TradeTracker` pour le backtest, mais en
s'appuyant sur `avg_entry_price` déjà calculé par le BROKER lui-même plutôt
qu'en reconstituant un prix d'entrée moyen pondéré au fil des fills (le
broker tient déjà cette moyenne à jour pour chaque position ouverte, pas
besoin de la dupliquer côté bot).

Alimente `trading_bot.portfolio.symbol_track_record` à chaque cycle live
(voir `trading_bot.live.engine.run_once`) : c'est la seule façon, dans ce
projet, d'accumuler une base de confiance par symbole sur des périodes
RÉELLEMENT nouvelles jour après jour, plutôt que de rejouer un backtest sur
une période déjà connue (voir l'avertissement dans
`trading_bot.portfolio.symbol_track_record`).

Limites connues :
  - Le prix de sortie utilisé est le dernier prix connu au moment de la
    comparaison (`exit_prices`, rafraîchi à chaque cycle), pas le prix de
    fill RÉEL du broker (nécessiterait de lister l'historique des ordres
    exécutés, pas branché pour l'instant) — approximation dans le même
    esprit que le reste du moteur live (ex: dimensionnement basé sur une
    equity estimée plutôt qu'un fill confirmé, voir la docstring de
    `trading_bot.live.engine`).
  - `entry_date`/`exit_date` valent tous deux l'instant de la détection : le
    broker ne renvoie que le prix d'entrée moyen d'une position, pas sa date
    de constitution. Sans impact sur `trading_bot.portfolio.symbol_track_record`,
    qui n'utilise que `pnl`/`pnl_pct`/`exit_date` (voir sa docstring).
"""

from __future__ import annotations

import pandas as pd

from trading_bot.backtest.trades import Trade
from trading_bot.execution.broker_base import Position, PositionSnapshot


def snapshot_positions(positions: dict[str, Position]) -> dict[str, PositionSnapshot]:
    """Réduit les `Position` (complètes, côté broker) à ce qui doit être
    persisté entre deux cycles (`LiveState.last_known_positions`) pour
    détecter plus tard ce qui a été réalisé. Ignore les positions déjà
    nulles (rien à comparer plus tard)."""
    return {symbol: PositionSnapshot.from_position(p) for symbol, p in positions.items() if p.qty != 0}


def detect_realized_trades(
    previous: dict[str, PositionSnapshot],
    current: dict[str, Position],
    exit_prices: dict[str, float],
    now: pd.Timestamp,
) -> list[Trade]:
    """Compare deux instantanés de positions et renvoie les `Trade` RÉALISÉS
    entre les deux (réduction partielle ou clôture totale) — jamais pour une
    position maintenue ou augmentée, jamais pour une nouvelle position sans
    instantané `previous` pour comparaison.

    `previous` : dernier instantané connu (généralement
    `LiveState.last_known_positions`, persisté à la fin du cycle précédent).
    `current` : positions fraîchement lues chez le broker.
    `exit_prices` : dernier prix connu par symbole (voir les limites en tête
    de module sur cette approximation).
    """
    trades: list[Trade] = []
    for symbol, prev in previous.items():
        if prev.qty == 0:
            continue
        current_position = current.get(symbol)
        current_qty = current_position.qty if current_position else 0.0

        same_direction = current_qty != 0 and (current_qty > 0) == (prev.qty > 0)
        closed_qty = (abs(prev.qty) - abs(current_qty)) if same_direction else abs(prev.qty)
        if closed_qty <= 1e-9:
            continue  # position maintenue ou augmentée : rien de réalisé

        exit_price = exit_prices.get(symbol)
        if not exit_price or exit_price <= 0:
            continue  # pas de prix disponible pour valoriser la sortie : on ne peut rien enregistrer

        direction = 1 if prev.qty > 0 else -1
        entry_value = closed_qty * prev.avg_entry_price
        pnl = direction * closed_qty * (exit_price - prev.avg_entry_price)
        trades.append(
            Trade(
                symbol=symbol,
                direction=direction,
                entry_date=now,
                exit_date=now,
                entry_price=prev.avg_entry_price,
                exit_price=exit_price,
                qty=closed_qty,
                entry_value=entry_value,
                pnl=pnl,
                exit_reason="live",
            )
        )
    return trades

"""Suivi des trades individuels (ouverture/ajout/réduction/clôture d'une
position) pendant un backtest, pour alimenter les métriques par trade (voir
`trading_bot.backtest.metrics`).

Modèle retenu : un "lot" ouvert par symbole, avec un prix d'entrée moyenné
pondéré si la position est augmentée en plusieurs fois. Toute RÉDUCTION de
la position (partielle ou totale) réalise le P&L sur la quantité réduite,
au prorata de la commission d'entrée déjà payée sur ce lot — pas seulement
un retour exact à zéro : le rebalancement quotidien de ce bot ajuste
souvent une position sans la flatten complètement (signal qui faiblit sans
disparaître, caps de portefeuille qui rognent une taille cible...), et s'en
tenir aux seules clôtures totales sous-compterait largement les trades
réels et fausserait le win rate / gain moyen par trade.

Un changement de SENS (long -> short ou l'inverse) en un seul mouvement est
traité comme une clôture totale de l'ancien lot suivie de l'ouverture d'un
nouveau lot dans l'autre sens (peut survenir si `risk.allow_short` est
activé et qu'un signal s'inverse d'un coup).

Les positions encore ouvertes à la toute fin du backtest ne sont jamais
comptées comme des trades : seul le P&L RÉALISÉ alimente les métriques par
trade (convention standard).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Trade:
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float  # moyenne pondérée si la position a été augmentée en plusieurs fois
    exit_price: float
    qty: float  # quantité RÉALISÉE par ce trade (clôture totale ou partielle du lot)
    entry_value: float  # abs(qty * entry_price) : base du rendement en %
    pnl: float  # $ net de commissions (part d'entrée au prorata + commission de sortie)
    exit_reason: str  # "stop" | "rebalance"

    @property
    def pnl_pct(self) -> float:
        return self.pnl / self.entry_value if self.entry_value else 0.0


@dataclass
class _OpenLot:
    direction: int
    qty: float  # toujours positif (la direction porte le signe)
    avg_entry_price: float
    entry_date: pd.Timestamp
    entry_commission: float  # commission cumulée sur la partie encore ouverte du lot


class TradeTracker:
    """État mutable tenu par `trading_bot.backtest.engine` : un appel à
    `record_fill` par exécution d'ordre (rebalancement à l'ouverture, ou
    sortie sur stop suiveur)."""

    def __init__(self) -> None:
        self._open_lots: dict[str, _OpenLot] = {}
        self.completed_trades: list[Trade] = []

    def record_fill(
        self,
        symbol: str,
        dt: pd.Timestamp,
        old_qty: float,
        new_qty: float,
        price: float,
        commission: float,
        exit_reason: str = "rebalance",
    ) -> None:
        if old_qty == 0 or abs(old_qty - new_qty) < 1e-12:
            if abs(old_qty) < 1e-12 and abs(new_qty) > 1e-12:
                self._open_new_lot(symbol, dt, new_qty, price, commission)
            return

        lot = self._open_lots.get(symbol)
        if lot is None:
            # Ne devrait pas arriver (old_qty != 0 implique normalement un lot
            # suivi), mais on ouvre quand même un nouveau lot plutôt que de
            # planter sur un état incohérent.
            if abs(new_qty) > 1e-12:
                self._open_new_lot(symbol, dt, new_qty, price, commission)
            return

        same_direction = (new_qty > 0) == (lot.direction > 0)

        if new_qty != 0 and same_direction:
            if abs(new_qty) > lot.qty + 1e-12:
                # Ajout à la position existante : prix d'entrée moyenné pondéré.
                added_qty = abs(new_qty) - lot.qty
                total_qty = lot.qty + added_qty
                lot.avg_entry_price = (lot.avg_entry_price * lot.qty + price * added_qty) / total_qty
                lot.qty = total_qty
                lot.entry_commission += commission
                return
            if abs(new_qty) < lot.qty - 1e-12:
                # Réduction partielle : réalise le P&L sur la quantité réduite,
                # garde le reste du lot ouvert au même prix d'entrée moyen.
                closed_qty = lot.qty - abs(new_qty)
                self.completed_trades.append(
                    self._realize(symbol, lot, dt, price, closed_qty, commission, exit_reason)
                )
                lot.qty = abs(new_qty)
                return
            return  # quantité inchangée (ne devrait pas arriver, garde par sécurité)

        # Clôture totale (new_qty == 0) ou inversion de sens : on clôture tout
        # le lot existant, puis on ouvre un nouveau lot si `new_qty` n'est pas nul.
        self.completed_trades.append(self._realize(symbol, lot, dt, price, lot.qty, commission, exit_reason))
        del self._open_lots[symbol]
        if abs(new_qty) > 1e-12:
            # La commission de ce fill a déjà été imputée à la clôture de
            # l'ancien lot ci-dessus ; le nouveau lot démarre sans commission
            # d'entrée déjà comptée (légère simplification en cas d'inversion
            # de sens en un seul fill, cas rare avec `allow_short` uniquement).
            self._open_new_lot(symbol, dt, new_qty, price, 0.0)

    def _open_new_lot(self, symbol: str, dt: pd.Timestamp, qty: float, price: float, commission: float) -> None:
        self._open_lots[symbol] = _OpenLot(
            direction=1 if qty > 0 else -1,
            qty=abs(qty),
            avg_entry_price=price,
            entry_date=dt,
            entry_commission=commission,
        )

    def _realize(
        self,
        symbol: str,
        lot: _OpenLot,
        exit_date: pd.Timestamp,
        exit_price: float,
        closed_qty: float,
        exit_commission: float,
        exit_reason: str,
    ) -> Trade:
        entry_value = closed_qty * lot.avg_entry_price
        gross_pnl = lot.direction * closed_qty * (exit_price - lot.avg_entry_price)
        entry_commission_share = lot.entry_commission * (closed_qty / lot.qty) if lot.qty else 0.0
        lot.entry_commission -= entry_commission_share
        net_pnl = gross_pnl - entry_commission_share - exit_commission
        return Trade(
            symbol=symbol,
            direction=lot.direction,
            entry_date=lot.entry_date,
            exit_date=exit_date,
            entry_price=lot.avg_entry_price,
            exit_price=exit_price,
            qty=closed_qty,
            entry_value=entry_value,
            pnl=net_pnl,
            exit_reason=exit_reason,
        )

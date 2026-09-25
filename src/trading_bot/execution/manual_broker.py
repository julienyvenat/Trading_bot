"""Broker "manuel" pour un compte sans API de courtage (ex: PEA Fortuneo) :
ne passe jamais d'ordre lui-même, affiche l'instruction exacte à exécuter à
la main sur le site du courtier, et tient sa propre comptabilité (cash,
positions) dans un fichier JSON local plutôt que d'interroger un broker en
direct.

Bootstrap obligatoire avant le premier cycle : créer `account_file` (voir
`trading_bot.config.ManualBrokerConfig`) avec le solde de cash et les
positions réelles du compte, par exemple :

    {"cash": 5000.0, "positions": {}}

ou, avec des positions déjà ouvertes :

    {"cash": 1200.0, "positions": {"MC.PA": {"qty": 3, "avg_entry_price": 780.0}}}

Le bot suppose qu'un ordre affiché est exécuté au prix affiché dès que
l'utilisateur l'a réellement passé sur son courtier : `account_file` est mis
à jour en conséquence immédiatement après chaque `submit_market_order`. Si le
fill réel diffère (prix, ou ordre pas encore passé), éditer `account_file` à
la main avant le prochain cycle pour garder la comptabilité du bot alignée
sur la réalité du compte.

Les prix (dont ceux utilisés pour calculer l'equity) viennent de yfinance,
pas d'un flux temps réel du courtier : suffisant pour une utilisation à la
séance (quelques cycles par jour), pas pour de l'intraday serré.

Format des instructions : pensé pour Fortuneo (mnémonique sans le suffixe
yfinance ".PA", montants au format français), ex :

    ACHETER 38 PSP5 — ordre au marché (ou à cours limité 59,77 €) · ≈ 2 259,86 € au cours de 59,47 €, frais ≈ 4,52 €
    Une fois l'achat exécuté, poser un STOP SUIVEUR : vendre 38 PSP5, écart 12,5 % (≈ 7,43 €), seuil de départ 52,04 €

Les frais estimés (barème `backtest.commission_schedule`, voir
`trading_bot.portfolio.fees`) sont aussi déduits du cash de `account_file`.

Chaque instruction affichée est aussi mémorisée (`drain_instructions`) pour
que le moteur live puisse en envoyer un récapitulatif unique par cycle (push
Pushover, voir `trading_bot.notify.pushover`) plutôt que de devoir surveiller
les logs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from trading_bot.execution.broker_base import AccountInfo, Broker, Position
from trading_bot.logger import get_logger
from trading_bot.portfolio.fees import CommissionModel

logger = get_logger()


@dataclass
class ManualInstruction:
    """Instruction à exécuter à la main sur le courtier. `kind` : "order"
    (ordre au marché), "stop" (stop à poser), "cancel" (ancien stop à
    annuler/remplacer), "info" (information, rien à faire) ou "alert"
    (événement à vérifier : stop probablement déclenché, alerte de marché)."""

    kind: str
    text: str


def fmt_eur(value: float) -> str:
    """Montant au format français : 2 259,86 €."""
    return f"{value:,.2f}".replace(",", " ").replace(".", ",") + " €"


def fmt_pct(value: float) -> str:
    """Pourcentage au format français, 1 décimale : 12,5 %."""
    return f"{value * 100:.1f}".replace(".", ",") + " %"


def broker_ticker(symbol: str) -> str:
    """Mnémonique à saisir chez le courtier : ticker yfinance sans suffixe de
    place (ex: "PSP5.PA" -> "PSP5")."""
    return symbol.split(".")[0]


class ManualBroker(Broker):
    def __init__(
        self,
        account_file: str,
        calendar_name: str = "XPAR",
        commission: CommissionModel | None = None,
        limit_offset_pct: float = 0.005,
    ) -> None:
        self._path = Path(account_file)
        if not self._path.exists():
            raise RuntimeError(
                f"Fichier de compte manuel introuvable : {account_file}. Crée-le avec le solde de cash "
                'et les positions réelles du compte avant de démarrer, ex: {"cash": 5000.0, "positions": '
                '{}} (voir la docstring de trading_bot.execution.manual_broker).'
            )
        self._calendar_name = calendar_name
        self._price_cache: dict[str, float] = {}
        self._pending_instructions: list[ManualInstruction] = []
        self._commission = commission or CommissionModel()
        self._limit_offset_pct = limit_offset_pct

    def add_note(self, text: str, kind: str = "info") -> None:
        """Ajoute une ligne d'information (ou d'alerte) au récapitulatif du
        cycle, sans rien comptabiliser."""
        logger.warning("NOTE POUR LE COURTIER MANUEL : %s", text)
        self._pending_instructions.append(ManualInstruction(kind, text))

    def clear_price_cache(self) -> None:
        """Oublie les prix mis en cache (appelé au début de chaque cycle)."""
        self._price_cache.clear()

    def estimated_fee(self, notional: float) -> float:
        return self._commission.fee(notional)

    def drain_instructions(self) -> list[ManualInstruction]:
        """Renvoie puis oublie les instructions affichées depuis le dernier
        appel (utilisé par le moteur live pour notifier une fois par cycle)."""
        instructions, self._pending_instructions = self._pending_instructions, []
        return instructions

    def _load(self) -> dict:
        with open(self._path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_account(self) -> AccountInfo:
        data = self._load()
        cash = float(data.get("cash", 0.0))
        positions_value = sum(
            float(pos["qty"]) * self.get_last_price(symbol) for symbol, pos in data.get("positions", {}).items()
        )
        equity = cash + positions_value
        # Pas de marge sur un PEA : le pouvoir d'achat se limite au cash disponible.
        return AccountInfo(equity=equity, cash=cash, buying_power=cash)

    def get_positions(self) -> dict[str, Position]:
        data = self._load()
        positions: dict[str, Position] = {}
        for symbol, pos in data.get("positions", {}).items():
            qty = float(pos["qty"])
            price = self.get_last_price(symbol)
            positions[symbol] = Position(
                symbol=symbol, qty=qty, market_value=qty * price, avg_entry_price=float(pos["avg_entry_price"])
            )
        return positions

    def get_last_price(self, symbol: str) -> float:
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        import yfinance as yf  # import différé : dépendance réseau, inutile pour les tests unitaires

        history = yf.Ticker(symbol).history(period="5d")
        closes = history["Close"].dropna()
        # yfinance renvoie parfois une dernière ligne (jour le plus récent)
        # entièrement à NaN quand la clôture n'est pas encore disponible côté
        # Yahoo Finance : on retombe alors sur la dernière clôture connue
        # plutôt que de propager un prix NaN (qui corromprait silencieusement
        # tout le dimensionnement des ordres en aval, voir `plan_orders`).
        if closes.empty:
            raise RuntimeError(f"Impossible de récupérer le dernier prix de {symbol} via yfinance.")
        price = float(closes.iloc[-1])
        self._price_cache[symbol] = price
        return price

    def submit_market_order(self, symbol: str, qty: float, side: str) -> None:
        # Contrairement à Alpaca (qui accepte des quantités fractionnaires,
        # voir `trading_bot.execution.alpaca_broker`), un PEA ne se négocie
        # qu'en actions entières : le dimensionnement basé sur l'ATR
        # (`RiskManager`) produit pourtant quasi systématiquement des
        # quantités fractionnaires, qu'il faut donc arrondir ici. On arrondit
        # toujours VERS LE BAS (jamais au plus proche) pour ne jamais afficher
        # un ordre qui dépasserait le cash/la position réellement disponible.
        qty = math.floor(qty)
        if qty <= 0:
            logger.info(
                "Ordre sur %s ignoré : quantité cible (%s) arrondie à 0 action entière.", symbol, side
            )
            return

        price = self.get_last_price(symbol)
        notional = qty * price
        fee = self._commission.fee(notional)
        is_buy = side == "buy"
        limit = price * (1 + self._limit_offset_pct) if is_buy else price * (1 - self._limit_offset_pct)
        instruction = "%s %d %s — ordre au marché (ou à cours limité %s) · ≈ %s au cours de %s, frais ≈ %s" % (
            "ACHETER" if is_buy else "VENDRE",
            qty,
            broker_ticker(symbol),
            fmt_eur(limit),
            fmt_eur(notional),
            fmt_eur(price),
            fmt_eur(fee),
        )
        logger.warning("ORDRE MANUEL À PASSER SUR TON COURTIER : %s", instruction)
        self._pending_instructions.append(ManualInstruction("order", instruction))
        self._apply_fill(symbol, qty, side, price, fee)

    def _apply_fill(self, symbol: str, qty: float, side: str, price: float, fee: float) -> None:
        """Comptabilise dans `account_file` un fill supposé au prix `price`,
        frais `fee` déduits du cash."""
        notional = qty * price
        data = self._load()
        cash = float(data.get("cash", 0.0))
        positions = data.setdefault("positions", {})
        current = positions.get(symbol, {"qty": 0.0, "avg_entry_price": 0.0})
        current_qty = float(current["qty"])
        new_qty = current_qty + (qty if side == "buy" else -qty)

        if side == "buy":
            cash -= notional + fee
            if new_qty != 0:
                total_cost = current_qty * float(current["avg_entry_price"]) + notional
                current["avg_entry_price"] = total_cost / new_qty
        else:
            cash += notional - fee

        if new_qty == 0:
            positions.pop(symbol, None)
        else:
            current["qty"] = new_qty
            positions[symbol] = current

        data["cash"] = cash
        self._save(data)

    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float) -> str:
        instruction = "Poser un ordre STOP : %s %d %s, seuil de déclenchement %s" % (
            "acheter" if side == "buy" else "vendre",
            math.floor(qty),
            broker_ticker(symbol),
            fmt_eur(stop_price),
        )
        logger.warning("STOP MANUEL À POSER SUR TON COURTIER : %s", instruction)
        self._pending_instructions.append(ManualInstruction("stop", instruction))
        # Pas d'ordre réel côté courtier à référencer : identifiant purement
        # interne, pour que `trading_bot.state.LiveState.stop_order_ids`
        # continue de fonctionner (détection de stop "posé"/"à reposer").
        return f"manual-{symbol}-{int(time.time())}"

    def submit_trailing_stop_order(
        self, symbol: str, qty: float, trail_pct: float, start_stop_price: float, after_buy: bool = False
    ) -> None:
        """Instruction de pose d'un ordre "Stop Suiveur" NATIF (vente) : le
        courtier remonte lui-même le seuil en continu, le bot ne redemandera
        donc jamais de le "remplacer" à chaque cycle (voir
        `trading_bot.live.engine`, mode `live.manual.native_trailing_stop`).
        `after_buy` : l'achat correspondant figure dans le même récapitulatif,
        le stop ne peut être posé qu'une fois cet achat exécuté."""
        price = self.get_last_price(symbol)
        prefix = "Une fois l'achat exécuté, poser" if after_buy else "Poser"
        instruction = prefix + " un STOP SUIVEUR : vendre %d %s, écart %s (≈ %s), seuil de départ %s" % (
            math.floor(qty),
            broker_ticker(symbol),
            fmt_pct(trail_pct),
            fmt_eur(price * trail_pct),
            fmt_eur(start_stop_price),
        )
        logger.warning("STOP SUIVEUR NATIF À POSER SUR TON COURTIER : %s", instruction)
        self._pending_instructions.append(ManualInstruction("stop", instruction))

    def cancel_trailing_stop(self, symbol: str, reason: str) -> None:
        text = f"Annuler le STOP SUIVEUR sur {broker_ticker(symbol)} ({reason})"
        logger.warning("À FAIRE SUR TON COURTIER : %s", text)
        self._pending_instructions.append(ManualInstruction("cancel", text))

    def record_stop_exit(self, symbol: str, qty: float, exit_price: float, stop_price: float, low: float) -> None:
        """Le Stop Suiveur natif a vraisemblablement été exécuté par le
        courtier (plus bas du jour <= seuil) : comptabilise la vente au prix
        estimé et prévient l'utilisateur de vérifier/corriger."""
        qty = math.floor(qty)
        if qty <= 0:
            return
        fee = self._commission.fee(qty * exit_price)
        text = (
            "STOP SUIVEUR %s probablement déclenché (plus bas %s <= seuil %s) : vérifie sur Fortuneo. "
            "Compte du bot mis à jour : vente de %d à ≈ %s (corrige %s si le prix réel diffère)."
            % (broker_ticker(symbol), fmt_eur(low), fmt_eur(stop_price), qty, fmt_eur(exit_price), self._path.name)
        )
        logger.warning(text)
        self._pending_instructions.append(ManualInstruction("alert", text))
        self._apply_fill(symbol, qty, "sell", exit_price, fee)

    def cancel_order(self, order_id: str) -> None:
        logger.info(
            "Pense à annuler ou remplacer à la main le stop précédent sur ton courtier (référence interne %s).",
            order_id,
        )
        # Référence interne au format `manual-<symbole>-<timestamp>` (voir
        # `submit_stop_order`) : on en extrait le symbole pour un message lisible.
        symbol = order_id[len("manual-") :].rsplit("-", 1)[0] if order_id.startswith("manual-") else order_id
        self._pending_instructions.append(ManualInstruction("cancel", f"Annuler le stop précédent sur {symbol}"))

    def is_market_open(self) -> bool:
        from trading_bot.market_calendar import MarketCalendar

        return MarketCalendar(self._calendar_name).is_open_now()

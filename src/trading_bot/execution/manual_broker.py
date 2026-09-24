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

logger = get_logger()


@dataclass
class ManualInstruction:
    """Instruction à exécuter à la main sur le courtier. `kind` : "order"
    (ordre au marché), "stop" (stop à poser) ou "cancel" (ancien stop à
    annuler/remplacer)."""

    kind: str
    text: str


class ManualBroker(Broker):
    def __init__(self, account_file: str, calendar_name: str = "XPAR") -> None:
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
        instruction = "%s %d %s (≈ %.2f € au dernier cours de %.2f €)" % (
            "ACHETER" if side == "buy" else "VENDRE",
            qty,
            symbol,
            notional,
            price,
        )
        logger.warning("ORDRE MANUEL À PASSER SUR TON COURTIER : %s", instruction)
        self._pending_instructions.append(ManualInstruction("order", instruction))

        data = self._load()
        cash = float(data.get("cash", 0.0))
        positions = data.setdefault("positions", {})
        current = positions.get(symbol, {"qty": 0.0, "avg_entry_price": 0.0})
        current_qty = float(current["qty"])
        new_qty = current_qty + (qty if side == "buy" else -qty)

        if side == "buy":
            cash -= notional
            if new_qty != 0:
                total_cost = current_qty * float(current["avg_entry_price"]) + notional
                current["avg_entry_price"] = total_cost / new_qty
        else:
            cash += notional

        if new_qty == 0:
            positions.pop(symbol, None)
        else:
            current["qty"] = new_qty
            positions[symbol] = current

        data["cash"] = cash
        self._save(data)

    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float) -> str:
        instruction = "%s %d %s si le cours atteint %.2f €" % (
            "ACHETER" if side == "buy" else "VENDRE",
            math.floor(qty),
            symbol,
            stop_price,
        )
        logger.warning("STOP MANUEL À POSER SUR TON COURTIER : %s", instruction)
        self._pending_instructions.append(ManualInstruction("stop", instruction))
        # Pas d'ordre réel côté courtier à référencer : identifiant purement
        # interne, pour que `trading_bot.state.LiveState.stop_order_ids`
        # continue de fonctionner (détection de stop "posé"/"à reposer").
        return f"manual-{symbol}-{int(time.time())}"

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

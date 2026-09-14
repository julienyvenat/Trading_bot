"""Interface abstraite d'un broker (permet de changer de courtier facilement)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float


@dataclass
class Position:
    symbol: str
    qty: float  # positif = long, négatif = short
    market_value: float
    avg_entry_price: float


class Broker(ABC):
    @abstractmethod
    def get_account(self) -> AccountInfo: ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def get_last_price(self, symbol: str) -> float: ...

    @abstractmethod
    def submit_market_order(self, symbol: str, qty: float, side: str) -> None:
        """side: 'buy' ou 'sell'. qty toujours positif."""
        ...

    @abstractmethod
    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float) -> str:
        """Envoie un ordre stop natif (déclenché en continu par le broker,
        contrairement à un stop suiveur vérifié uniquement à chaque cycle
        Python). `side` : 'buy' ou 'sell' (sens de la sortie, pas de la
        position). `qty` toujours positif. Renvoie l'identifiant d'ordre du
        broker, à conserver pour pouvoir l'annuler/le remplacer plus tard."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Annule un ordre existant. Ne doit pas lever si l'ordre n'existe
        plus (déjà exécuté ou déjà annulé) : c'est un cas normal, à gérer par
        l'appelant plutôt que de faire planter le cycle live."""
        ...

    @abstractmethod
    def is_market_open(self) -> bool: ...

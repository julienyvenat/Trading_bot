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
    def is_market_open(self) -> bool: ...

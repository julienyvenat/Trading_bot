"""Persistance de l'état du moteur live entre deux redémarrages.

Deux choses doivent survivre à un redémarrage du bot :
  - les stops suiveurs en cours (sinon on perdrait le "ratchet" et repartirait
    d'un stop plus large que ce qu'il devrait être) ;
  - l'état des coupe-circuits, en particulier le coupe-circuit de drawdown,
    qui est volontairement "sticky" : si le bot crashe puis redémarre après
    l'avoir déclenché, il DOIT rester arrêté plutôt que de repartir comme si
    de rien n'était.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from trading_bot.portfolio.circuit_breaker import RiskState
from trading_bot.portfolio.stops import StopLevel


@dataclass
class LiveState:
    trailing_stops: dict[str, StopLevel] = field(default_factory=dict)
    risk_state: RiskState | None = None

    def to_dict(self) -> dict:
        return {
            "trailing_stops": {
                symbol: {"direction": stop.direction, "stop_price": stop.stop_price}
                for symbol, stop in self.trailing_stops.items()
            },
            "risk_state": self.risk_state.to_dict() if self.risk_state else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LiveState:
        stops = {
            symbol: StopLevel(direction=v["direction"], stop_price=v["stop_price"])
            for symbol, v in data.get("trailing_stops", {}).items()
        }
        risk_state = RiskState.from_dict(data["risk_state"]) if data.get("risk_state") else None
        return cls(trailing_stops=stops, risk_state=risk_state)


def load_state(path: str | Path) -> LiveState:
    file_path = Path(path)
    if not file_path.exists():
        return LiveState()
    with open(file_path, encoding="utf-8") as f:
        return LiveState.from_dict(json.load(f))


def save_state(path: str | Path, state: LiveState) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)

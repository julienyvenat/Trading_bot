"""Coupe-circuits de risque au niveau du portefeuille.

Deux mécanismes indépendants, tous deux pilotés par `config.yaml -> risk:` :

- **Perte journalière max** (`max_daily_loss_pct`) : bloque toute NOUVELLE
  entrée (ou augmentation de position) pour le reste de la séance en cours si
  la perte depuis le début de la séance dépasse le seuil. Les positions déjà
  ouvertes restent gérées normalement (stop suiveur actif). Se réinitialise
  automatiquement au début de la séance suivante.

- **Drawdown max** (`max_drawdown_pct`) : si l'equity chute de plus de ce
  pourcentage depuis son plus haut historique, on liquide TOUT et on arrête
  complètement le trading. Contrairement au précédent, celui-ci est
  volontairement "sticky" : il ne se réinitialise jamais tout seul, pour
  forcer une intervention humaine avant de reprendre (voir README).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from trading_bot.config import RiskConfig
from trading_bot.portfolio.risk import PositionSizing


@dataclass
class RiskState:
    equity_peak: float
    session_start_equity: float
    current_date: date | None = None
    daily_halted: bool = False
    drawdown_halted: bool = False

    @classmethod
    def initial(cls, equity: float, today: date | None = None) -> RiskState:
        return cls(equity_peak=equity, session_start_equity=equity, current_date=today)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["current_date"] = self.current_date.isoformat() if self.current_date else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> RiskState:
        raw_date = data.get("current_date")
        return cls(
            equity_peak=data["equity_peak"],
            session_start_equity=data["session_start_equity"],
            current_date=date.fromisoformat(raw_date) if raw_date else None,
            daily_halted=data.get("daily_halted", False),
            drawdown_halted=data.get("drawdown_halted", False),
        )


class CircuitBreaker:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def update(self, state: RiskState, equity: float, today: date) -> RiskState:
        """Met à jour l'état des coupe-circuits pour le pas de temps courant.
        Renvoie un nouvel état (ne mute pas `state` en place)."""
        current_date = state.current_date
        session_start_equity = state.session_start_equity
        daily_halted = state.daily_halted

        if current_date != today:
            current_date = today
            session_start_equity = equity
            daily_halted = False  # nouvelle séance : on ré-autorise les entrées

        equity_peak = max(state.equity_peak, equity)

        if session_start_equity > 0:
            daily_loss_pct = (session_start_equity - equity) / session_start_equity
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                daily_halted = True

        drawdown_halted = state.drawdown_halted
        if equity_peak > 0:
            drawdown_pct = (equity_peak - equity) / equity_peak
            if drawdown_pct >= self.config.max_drawdown_pct:
                drawdown_halted = True  # sticky, voir docstring du module

        return RiskState(
            equity_peak=equity_peak,
            session_start_equity=session_start_equity,
            current_date=current_date,
            daily_halted=daily_halted,
            drawdown_halted=drawdown_halted,
        )


def new_entries_allowed(state: RiskState) -> bool:
    return not (state.daily_halted or state.drawdown_halted)


def should_flatten(state: RiskState) -> bool:
    return state.drawdown_halted


def apply_halt(
    sizings: dict[str, PositionSizing],
    current_weights: dict[str, float],
    state: RiskState,
) -> dict[str, PositionSizing]:
    """Applique l'effet des coupe-circuits sur les tailles de position calculées.

    - Coupe-circuit de drawdown déclenché -> flatten total (renvoie {}).
    - Coupe-circuit journalier déclenché -> aucune nouvelle position, et les
      positions existantes ne peuvent pas être agrandies (mais peuvent être
      réduites ou fermées normalement par la stratégie ou le stop suiveur).
    - Sinon -> aucun changement.
    """
    if should_flatten(state):
        return {}

    if new_entries_allowed(state):
        return sizings

    adjusted: dict[str, PositionSizing] = {}
    for symbol, sizing in sizings.items():
        current_weight = current_weights.get(symbol, 0.0)
        if current_weight == 0.0:
            continue  # pas de nouvelle position pendant le halt journalier

        same_direction = (sizing.target_weight > 0) == (current_weight > 0)
        if same_direction and abs(sizing.target_weight) > abs(current_weight):
            # Gèle la taille actuelle : pas d'ajout de risque, mais pas de
            # fermeture forcée non plus.
            sizing = PositionSizing(
                symbol=symbol, target_weight=current_weight, stop_loss_price=sizing.stop_loss_price
            )
        adjusted[symbol] = sizing

    return adjusted

"""Stop-loss suiveur basé sur l'ATR (trailing stop) — logique pure.

Réutilisée telle quelle par le backtest (jour par jour) et par le moteur live
(cycle par cycle), pour garantir le même comportement des deux côtés.

Principe : le stop est recalculé à chaque pas de temps à partir de l'ATR
courant, mais ne peut jamais se déplacer en défaveur de la position (il ne
fait que "ratchet" dans le sens favorable) — c'est ce qui protège les gains
au fur et à mesure qu'une position progresse, plutôt qu'un stop fixe qui
resterait au niveau d'entrée.

Limite connue : le stop n'est vérifié qu'une fois par pas de temps (jour en
backtest, `loop_interval_seconds` en live) — un mouvement violent *intra-cycle*
qui reviendrait avant le prochain contrôle ne serait pas capturé. Une
amélioration future possible est de déléguer le stop au broker (ordre stop
natif), qui réagit en continu.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopLevel:
    direction: int  # +1 = position longue, -1 = position courte
    stop_price: float


def compute_candidate_stop(direction: int, close: float, atr_value: float, atr_stop_multiple: float) -> float:
    """Niveau de stop "brut" du jour, avant application du ratchet."""
    return close - direction * atr_value * atr_stop_multiple


def update_stop(
    previous: StopLevel | None,
    direction: int,
    close: float,
    atr_value: float | None,
    atr_stop_multiple: float,
) -> StopLevel | None:
    """Calcule le nouveau stop suiveur pour une position ouverte.

    - Si pas d'ATR disponible (warmup), conserve le stop précédent tel quel.
    - Si la position change de sens (ou n'existait pas), repart d'un nouveau stop.
    - Sinon, ratchet : un long ne peut que remonter son stop, un short ne peut
      que le baisser.
    """
    if direction == 0:
        return None
    if atr_value is None or atr_value <= 0 or close <= 0:
        return previous

    candidate = compute_candidate_stop(direction, close, atr_value, atr_stop_multiple)

    if previous is None or previous.direction != direction:
        return StopLevel(direction=direction, stop_price=candidate)

    if direction > 0:
        return StopLevel(direction=direction, stop_price=max(previous.stop_price, candidate))
    return StopLevel(direction=direction, stop_price=min(previous.stop_price, candidate))


def is_triggered(stop: StopLevel | None, low: float, high: float) -> bool:
    """True si le range [low, high] du pas de temps courant a touché le stop."""
    if stop is None:
        return False
    if stop.direction > 0:
        return low <= stop.stop_price
    return high >= stop.stop_price

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


# --- Stop suiveur en % (sémantique "Stop Suiveur" natif de courtier) --------
#
# Contrairement au stop ATR ci-dessus (recalculé à chaque pas de temps à
# partir de l'ATR courant), un ordre "Stop Suiveur" natif (ex: Fortuneo) est
# défini une fois pour toutes à la pose par un ÉCART (ici en % du cours) :
# c'est ensuite le courtier qui remonte le seuil en continu, à
# `plus haut atteint depuis la pose x (1 - écart)`. Modélisé à l'identique
# côté backtest et live pour ne jamais notifier "remplace ton stop" à chaque
# cycle alors que le courtier le fait déjà tout seul.

STOP_MODES = ("atr", "trailing_pct", "none")


@dataclass(frozen=True)
class PctTrailingStop:
    trail_pct: float  # écart figé à l'entrée, ex 0.125 = 12,5 %
    high_water: float  # plus haut atteint depuis la pose

    @property
    def stop_price(self) -> float:
        return self.high_water * (1.0 - self.trail_pct)

    def ratchet(self, high: float) -> PctTrailingStop:
        """Remonte le plus haut de référence (jamais à la baisse)."""
        if high is None or high <= self.high_water:
            return self
        return PctTrailingStop(trail_pct=self.trail_pct, high_water=float(high))


def validate_stop_mode(mode: str) -> str:
    if mode not in STOP_MODES:
        raise ValueError(f"`risk.stop_mode` inconnu '{mode}'. Valeurs supportées : {', '.join(STOP_MODES)}.")
    return mode


def entry_trail_pct(
    fixed_pct: float | None, price: float, atr_value: float | None, atr_stop_multiple: float
) -> float | None:
    """Écart (en %) à figer à l'entrée : `fixed_pct` s'il est fourni, sinon
    `atr_stop_multiple` x ATR / prix (converti une seule fois, à l'entrée).
    None si aucun des deux n'est calculable (ATR en warmup)."""
    if fixed_pct is not None:
        if not 0 < fixed_pct < 1:
            raise ValueError(f"`risk.trailing_stop_pct` doit être dans ]0, 1[ (reçu {fixed_pct}).")
        return float(fixed_pct)
    if atr_value is None or atr_value <= 0 or price <= 0:
        return None
    return min(0.95, atr_stop_multiple * atr_value / price)


def pct_stop_exit_price(stop: PctTrailingStop, open_price: float) -> float:
    """Prix d'exécution réaliste d'un stop déclenché : au seuil, ou à
    l'ouverture si le cours a ouvert en gap SOUS le seuil (le stop devient
    alors un ordre au marché exécuté au premier cours disponible)."""
    if open_price is not None and open_price > 0:
        return min(stop.stop_price, open_price)
    return stop.stop_price

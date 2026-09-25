"""Persistance de l'état du moteur live entre deux redémarrages.

Quatre choses doivent survivre à un redémarrage du bot :
  - les stops suiveurs en cours (sinon on perdrait le "ratchet" et repartirait
    d'un stop plus large que ce qu'il devrait être) ;
  - les identifiants des ordres stop natifs posés chez le broker pour chaque
    position (nécessaire pour pouvoir les annuler/remplacer au bon moment,
    voir `trading_bot.live.engine`), ainsi que la date de séance à laquelle
    chacun a été posé (`stop_order_dates`) : Alpaca n'accepte les ordres
    stop/stop_limit en quantité fractionnaire qu'en `TimeInForce.DAY` (jamais
    `GTC`), donc chaque ordre stop natif expire à la clôture et doit être
    reposé à chaque nouvelle séance, même si son prix n'a pas bougé — sans
    cette date, le bot croirait un stop de la veille toujours actif alors
    qu'il a expiré côté broker, laissant la position sans protection ;
  - l'état des coupe-circuits, en particulier le coupe-circuit de drawdown,
    qui est volontairement "sticky" : si le bot crashe puis redémarre après
    l'avoir déclenché, il DOIT rester arrêté plutôt que de repartir comme si
    de rien n'était ;
  - le dernier instantané connu des positions ouvertes (`last_known_positions`,
    voir `trading_bot.live.trade_realization`), pour détecter au cycle
    suivant ce qui a été réalisé entre-temps (clôture sur stop natif, ou
    toute autre sortie de position) et alimenter
    `trading_bot.portfolio.symbol_track_record` — sans ce point de
    comparaison persistant, un redémarrage du bot ferait perdre la trace du
    prix d'entrée des positions closes pendant l'interruption.

Pour le mode PEA manuel (`live.manual.native_trailing_stop`, alertes,
`live.daily_run_after`), s'y ajoutent : les Stops Suiveurs natifs posés chez
le courtier (écart figé, plus haut de référence, quantité, dernière bougie
examinée), les symboles sortis sur stop en attente de ré-entrée, l'état des
alertes d'information (pour ne notifier qu'au franchissement) et la date du
dernier cycle quotidien.

Mode `core_satellite` : `core_satellite` garde la comptabilité par parts
(NAV hors apports, plus haut, dernier cash/quantités vus pour détecter un
apport), l'année du dernier rééquilibrage calendaire et les plus hauts des
poches suivies par les alertes de baisse.

Aperçu du matin (`live.morning_brief`) : `last_morning_brief` (date locale
du dernier aperçu envoyé, pour ne jamais l'envoyer deux fois le même jour,
y compris après un redémarrage) et `last_cycle_orders` (ordres poussés par
le dernier cycle du soir, rappelés le lendemain matin).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from trading_bot.execution.broker_base import PositionSnapshot
from trading_bot.portfolio.circuit_breaker import RiskState
from trading_bot.portfolio.stops import StopLevel


@dataclass
class LiveState:
    trailing_stops: dict[str, StopLevel] = field(default_factory=dict)
    # Identifiant (côté broker) de l'ordre stop natif actuellement posé pour
    # chaque symbole ayant une position ouverte. Absent si aucun ordre stop
    # natif n'est actuellement posé (ex: dry-run, ou pas encore soumis).
    stop_order_ids: dict[str, str] = field(default_factory=dict)
    # Date (ISO, "YYYY-MM-DD") de la séance à laquelle le stop natif de
    # chaque symbole a été (re)posé. Ces ordres étant en `TimeInForce.DAY`
    # (voir `AlpacaBroker.submit_stop_order`), un id présent dans
    # `stop_order_ids` mais dont la date ici est antérieure à la séance en
    # cours correspond à un ordre déjà expiré côté broker : `run_once` doit
    # le reposer même si le prix du stop n'a pas changé.
    stop_order_dates: dict[str, str] = field(default_factory=dict)
    risk_state: RiskState | None = None
    last_known_positions: dict[str, PositionSnapshot] = field(default_factory=dict)
    # {symbole: {"trail_pct", "high_water", "qty", "last_date"}} : Stop
    # Suiveur natif que l'utilisateur a été invité à poser chez le courtier.
    native_stops: dict[str, dict] = field(default_factory=dict)
    # {symbole: date ISO de la sortie sur stop} (voir `risk.stop_reentry`).
    stopped_out: dict[str, str] = field(default_factory=dict)
    alert_flags: dict[str, bool] = field(default_factory=dict)
    last_daily_run: str | None = None
    core_satellite: dict = field(default_factory=dict)
    last_morning_brief: str | None = None
    # {"session": date ISO du cycle du soir, "orders": [texte de chaque ordre]}.
    # Vide : aucun cycle du soir enregistré depuis l'ajout de l'aperçu du matin.
    last_cycle_orders: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trailing_stops": {
                symbol: {"direction": stop.direction, "stop_price": stop.stop_price}
                for symbol, stop in self.trailing_stops.items()
            },
            "stop_order_ids": dict(self.stop_order_ids),
            "stop_order_dates": dict(self.stop_order_dates),
            "risk_state": self.risk_state.to_dict() if self.risk_state else None,
            "last_known_positions": {symbol: snap.to_dict() for symbol, snap in self.last_known_positions.items()},
            "native_stops": {symbol: dict(v) for symbol, v in self.native_stops.items()},
            "stopped_out": dict(self.stopped_out),
            "alert_flags": dict(self.alert_flags),
            "last_daily_run": self.last_daily_run,
            "core_satellite": json.loads(json.dumps(self.core_satellite)),
            "last_morning_brief": self.last_morning_brief,
            "last_cycle_orders": json.loads(json.dumps(self.last_cycle_orders)),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LiveState:
        stops = {
            symbol: StopLevel(direction=v["direction"], stop_price=v["stop_price"])
            for symbol, v in data.get("trailing_stops", {}).items()
        }
        stop_order_ids = dict(data.get("stop_order_ids", {}))
        stop_order_dates = dict(data.get("stop_order_dates", {}))
        risk_state = RiskState.from_dict(data["risk_state"]) if data.get("risk_state") else None
        last_known_positions = {
            symbol: PositionSnapshot.from_dict(v) for symbol, v in data.get("last_known_positions", {}).items()
        }
        return cls(
            trailing_stops=stops,
            stop_order_ids=stop_order_ids,
            stop_order_dates=stop_order_dates,
            risk_state=risk_state,
            last_known_positions=last_known_positions,
            native_stops={symbol: dict(v) for symbol, v in data.get("native_stops", {}).items()},
            stopped_out=dict(data.get("stopped_out", {})),
            alert_flags=dict(data.get("alert_flags", {})),
            last_daily_run=data.get("last_daily_run"),
            core_satellite=dict(data.get("core_satellite", {}) or {}),
            last_morning_brief=data.get("last_morning_brief"),
            last_cycle_orders=dict(data.get("last_cycle_orders", {}) or {}),
        )


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

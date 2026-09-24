"""Cycle live du mode `core_satellite` (voir `trading_bot.portfolio.
core_satellite` pour la logique de décision, partagée avec le backtest).

Appelé par `trading_bot.live.engine.run_once` à la place du pipeline
signaux -> risque -> ordres quand `core_satellite` est la stratégie active.
Sur un broker manuel (PEA Fortuneo), un cycle :
  1. valorise le compte (`account_file`) et détecte un éventuel apport :
     variation de valeur que les ordres du bot n'expliquent pas (Julien
     ajoute le virement au cash de `account_file`). Comptabilité par parts
     (`state.core_satellite`) : la NAV par part mesure la performance hors
     apports, base des alertes de drawdown ;
  2. appelle `plan_rebalance` (dérive, date annuelle, cash à investir) ;
  3. s'il y a des ordres : une ligne de contexte (raison, poids actuels ->
     après ordres, cibles), puis les ordres au format Fortuneo — le tout dans
     UN push (voir `notify_cycle_instructions`). Rien à faire : rien envoyé ;
  4. alertes d'information dédupliquées (drawdown du portefeuille par
     paliers, baisse de la poche à levier), jamais d'ordre, jamais de stop.

Positions du fichier de compte hors des poches du plan : ignorées (ni
valorisées dans les poids, ni vendues) — le plan ne gère que ses poches.
"""

from __future__ import annotations

import copy
import math

import pandas as pd

from trading_bot.config import AppConfig
from trading_bot.execution.broker_base import Broker
from trading_bot.execution.manual_broker import broker_ticker, fmt_eur, fmt_pct
from trading_bot.execution.rebalancer import execute_orders
from trading_bot.live.trade_realization import detect_realized_trades, snapshot_positions
from trading_bot.logger import get_logger
from trading_bot.market_calendar import MarketCalendar
from trading_bot.portfolio.core_satellite import (
    CoreSatelliteParams,
    RebalancePlan,
    calendar_rebalance_due,
    plan_rebalance,
)
from trading_bot.portfolio.fees import ExecutionRules
from trading_bot.state import LiveState

logger = get_logger()

MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]
# Variation de valeur inexpliquée en dessous de laquelle on ne parle pas
# d'apport (corrections de prix d'exécution de quelques euros).
FLOW_NOTICE_EUR = 1.0


def _pts(value: float) -> str:
    return f"{value:+.1f}".replace(".", ",") + " pts"


def plan_summary(params: CoreSatelliteParams) -> str:
    sleeves = " · ".join(f"{broker_ticker(s)} {fmt_pct(w)}" for s, w in params.targets.items())
    when = (
        f" ou en {MONTHS_FR[params.annual_rebalance_month - 1]}"
        if params.annual_rebalance_month is not None
        else ""
    )
    return (
        f"Plan : {sleeves} ; rééquilibrage si une poche dérive de plus de "
        f"{params.drift_threshold_pts:g} pts{when}, apports vers les poches en retard"
    )


def update_units(cs: dict, cash: float, qty: dict[str, float], prices: dict[str, float]) -> float:
    """Met à jour la comptabilité par parts de `cs` et renvoie le flux
    externe détecté depuis le dernier cycle (apport > 0, retrait < 0) : écart
    entre la valeur actuelle et celle qu'aurait le portefeuille laissé par le
    bot au dernier cycle, aux prix du jour."""
    equity = cash + sum(qty[s] * prices[s] for s in qty)
    if not cs.get("units"):
        cs.update(units=equity, nav=1.0, nav_peak=1.0)
        return 0.0
    last_qty = cs.get("last_qty", {})
    expected = float(cs.get("last_cash", 0.0)) + sum(float(q) * prices.get(s, 0.0) for s, q in last_qty.items())
    flow = equity - expected
    units = float(cs["units"])
    if abs(flow) >= FLOW_NOTICE_EUR and expected > 0:
        units += flow / (expected / units)
    elif expected <= 0 < equity:
        units = equity / float(cs.get("nav", 1.0) or 1.0)
    cs["units"] = units
    cs["nav"] = equity / units if units > 0 else 1.0
    cs["nav_peak"] = max(float(cs.get("nav_peak", 1.0)), cs["nav"])
    return flow if abs(flow) >= FLOW_NOTICE_EUR else 0.0


def describe_plan(plan: RebalancePlan, params: CoreSatelliteParams, flow: float, equity: float) -> list[str]:
    """Lignes de contexte du push : raison, puis poids actuels -> après."""
    lines = []
    prefix = f"Apport détecté : {'+' if flow > 0 else ''}{fmt_eur(flow)}. " if flow else ""
    how = (
        "avec vente(s) : passe d'abord la ou les ventes, leur produit finance les achats."
        if plan.sells
        else "par les achats seulement, aucune vente."
    )
    if plan.reason == "contribution":
        deployable = max(0.0, plan.cash_before - params.cash_buffer(plan.cash_before))
        lines.append(
            f"{prefix}Plan cœur-satellite : {fmt_eur(deployable)} de cash à investir (seuil "
            f"{fmt_eur(params.contribution_threshold(equity))}), vers les poches en retard, {how}"
        )
    elif plan.reason == "drift":
        symbol = max(params.targets, key=lambda s: abs(plan.weights_before.get(s, 0.0) - params.targets[s]))
        gap = (plan.weights_before.get(symbol, 0.0) - params.targets[symbol]) * 100
        lines.append(
            f"{prefix}Plan cœur-satellite : {broker_ticker(symbol)} dérive de {_pts(gap)} (seuil "
            f"{params.drift_threshold_pts:g} pts) — rééquilibrage {how}"
        )
    else:
        lines.append(f"{prefix}Plan cœur-satellite : rééquilibrage annuel — {how}")
    weights = " · ".join(
        f"{broker_ticker(s)} {fmt_pct(plan.weights_before.get(s, 0.0))} → {fmt_pct(plan.weights_after.get(s, 0.0))} "
        f"(cible {fmt_pct(t)})"
        for s, t in params.targets.items()
    )
    lines.append(f"Poids : {weights} · cash {fmt_eur(plan.cash_before)} → {fmt_eur(plan.cash_after)}")
    return lines


def _robust_peak(closes: pd.Series) -> float:
    """Plus haut des clôtures, en ignorant les cotations aberrantes (yfinance
    renvoie parfois un cours x300 un jour sans volume, vu sur CL2.PA)."""
    closes = closes.dropna()
    if closes.empty:
        return 0.0
    median = closes.rolling(21, center=True, min_periods=1).median()
    clean = closes[(closes <= 3 * median) & (closes >= median / 3)]
    return float(clean.max()) if not clean.empty else float(closes.max())


def check_core_satellite_alerts(
    config: AppConfig,
    params: CoreSatelliteParams,
    state: LiveState,
    cs: dict,
    data_by_symbol: dict[str, pd.DataFrame],
    broker: Broker,
    note,
) -> None:
    """Alertes d'INFORMATION du plan, chacune notifiée une fois par
    franchissement (réarmée quand la baisse est revenue sous la moitié du
    palier) : drawdown du portefeuille (NAV par part, hors apports) par
    paliers `live.alerts.drawdown_levels`, et baisse d'une poche depuis son
    plus haut (`live.alerts.symbol_drop_levels`)."""
    alerts = config.live.alerts
    if not alerts.enabled:
        return
    flags = state.alert_flags
    plan = plan_summary(params)

    def crossing(key_prefix: str, levels: list[float], drop: float) -> float | None:
        new = [lv for lv in sorted(levels) if drop >= lv and not flags.get(f"{key_prefix}:{lv:g}")]
        for lv in levels:
            key = f"{key_prefix}:{lv:g}"
            if drop >= lv:
                flags[key] = True
            elif flags.get(key) and drop < lv / 2:
                flags[key] = False
        return max(new) if new else None

    nav, peak = float(cs.get("nav", 1.0)), float(cs.get("nav_peak", 1.0))
    drawdown = 1.0 - nav / peak if peak > 0 else 0.0
    level = crossing("plan_drawdown", alerts.drawdown_levels, drawdown)
    if level is not None:
        text = (
            f"Portefeuille à -{fmt_pct(drawdown)} de son plus haut (performance hors apports), palier "
            f"-{fmt_pct(level)} franchi. Ne vends pas, c'est prévu dans le plan : une baisse de cette ampleur fait "
            f"partie du chemin d'un portefeuille 100 % actions avec levier, vendre maintenant figerait la perte. "
            f"{plan}. Aucun ordre à passer, continue les apports."
        )
        if level >= 0.5:
            text += " Le plan est fait pour traverser ces creux, pas pour les éviter (voir README, tests 2000-2012 et 2007-2009)."
        note(broker, text, kind="alert")

    peaks = cs.setdefault("peaks", {})
    for symbol, levels in alerts.symbol_drop_levels.items():
        df = data_by_symbol.get(symbol)
        if df is None or df.empty:
            continue
        close = float(df["close"].dropna().iloc[-1])
        sym_peak = max(float(peaks.get(symbol, 0.0)), _robust_peak(df["close"]))
        peaks[symbol] = sym_peak
        drop = 1.0 - close / sym_peak if sym_peak > 0 else 0.0
        level = crossing(f"drop:{symbol}", levels, drop)
        if level is None:
            continue
        ticker = broker_ticker(symbol)
        note(
            broker,
            f"{ticker} à -{fmt_pct(drop)} de son plus haut ({fmt_eur(close)} contre {fmt_eur(sym_peak)}), palier "
            f"-{fmt_pct(level)} franchi. Normal pour un ETF à levier x2 quotidien : il amplifie les baisses et perd en "
            f"plus à la volatilité (réinitialisation quotidienne). Ne vends pas : si son poids passe sous sa cible de "
            f"plus de {params.drift_threshold_pts:g} pts, le plan le rachète (apports d'abord). Aucun ordre à passer.",
            kind="alert",
        )


def run_core_satellite_cycle(
    config: AppConfig,
    params: CoreSatelliteParams,
    broker: Broker,
    dry_run: bool,
    state: LiveState,
    data_by_symbol: dict[str, pd.DataFrame],
    account,
    positions: dict,
    last_prices: dict[str, float],
    realized_trades: list,
    now_ts: pd.Timestamp,
    note,
    update_track_record,
) -> LiveState:
    if config.risk.stop_mode != "none":
        raise ValueError("core_satellite : aucun stop dans ce mode, `risk.stop_mode` doit valoir 'none'.")
    sleeves = list(params.targets)
    prices = {s: last_prices.get(s) for s in sleeves}
    bad = [s for s, p in prices.items() if p is None or not math.isfinite(p) or p <= 0]
    if bad:
        logger.warning("Plan cœur-satellite : pas de prix valide pour %s, cycle sans ordre.", ", ".join(bad))
        return state
    qty = {s: (positions[s].qty if s in positions else 0.0) for s in sleeves}
    cash = float(account.cash)
    equity = cash + sum(qty[s] * prices[s] for s in sleeves)
    rules = ExecutionRules.from_backtest_config(config.backtest)
    calendar = MarketCalendar(config.market.calendar)
    today = pd.Timestamp.now(tz=calendar.timezone).tz_localize(None).normalize()

    cs = copy.deepcopy(state.core_satellite) if dry_run else state.core_satellite
    flow = update_units(cs, cash, qty, prices)
    cs.setdefault("last_calendar_year", int(today.year))
    calendar_due = calendar_rebalance_due(params, today, cs.get("last_calendar_year"), calendar)
    if calendar_due:
        cs["last_calendar_year"] = int(today.year)

    plan = plan_rebalance(qty, prices, cash, params, rules, calendar_due=calendar_due)
    if plan.orders:
        for line in describe_plan(plan, params, flow, equity):
            note(broker, line)
        execute_orders(plan.orders, broker, dry_run=dry_run)
        if not state.alert_flags.get("core_satellite_no_stop_explained") and any(o.side == "buy" for o in plan.orders):
            note(
                broker,
                f"Pas de stop à poser ({', '.join(broker_ticker(s) for s in sleeves)}) : risk.stop_mode vaut none "
                "dans cette config."
                + (" Des alertes d'information préviennent en cas de forte baisse, sans jamais vendre." if config.live.alerts.enabled else ""),
            )
            state.alert_flags["core_satellite_no_stop_explained"] = True
    else:
        logger.info(
            "Plan cœur-satellite : rien à faire (dérive max %.1f pts, cash %.2f €%s).",
            plan.max_drift_before_pts,
            cash,
            ", rééquilibrage annuel sans ordre utile" if calendar_due else "",
        )

    check_core_satellite_alerts(config, params, state, cs, data_by_symbol, broker, note)

    if not dry_run:
        if plan.orders:
            positions_after = broker.get_positions()
            realized_trades += detect_realized_trades(
                snapshot_positions(positions), positions_after, last_prices, now_ts
            )
            positions = positions_after
        cs["last_cash"] = float(broker.get_account().cash)
        cs["last_qty"] = {s: float(positions[s].qty) for s in sleeves if s in positions}
    state.last_known_positions = snapshot_positions(positions)
    update_track_record(config, realized_trades)
    return state

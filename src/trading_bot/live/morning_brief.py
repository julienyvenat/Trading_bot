"""Aperçu du matin (`live.morning_brief`) : UN push Pushover concis avant
l'ouverture, pour le mode `core_satellite` sur broker manuel (PEA Fortuneo).

Contenu (moins de 1 024 caractères, voir `build_morning_brief`) :
  - valeur du portefeuille à la dernière clôture (cash compris), variation
    sur la veille, depuis le début du mois et de l'année — calculées à
    positions ACTUELLES sur l'historique de cours (pas d'historique de valeur
    à tenir, et les apports ne faussent pas la variation) ;
  - positions hors plan du fichier de compte (ex. 24 AXA gardées à côté) :
    valeur à la dernière clôture, plus-value contre le PRU, variation sur la
    veille, puis valeur totale du PEA (plan + hors plan). Elles n'entrent
    JAMAIS dans les poids, la dérive ni le cash à investir ;
  - poids de chaque poche du plan contre sa cible, dérive max contre le seuil ;
  - rappel des ordres poussés par le cycle de la veille au soir
    (`state.last_cycle_orders`), à passer aujourd'hui ;
  - cash non investi et ce qu'il attend (seuil d'apport) ;
  - alerte de drawdown active, le cas échéant, avec le rappel du plan ;
  - contexte de marché sur une ligne (S&P 500, MSCI World via URTH, EUR/USD,
    futures S&P). Toute donnée indisponible est simplement omise.

LECTURE SEULE : aucun ordre passé ni modifié, fichier de compte ouvert en
lecture, état du plan (`core_satellite`, `alert_flags`) jamais modifié. Seul
`run_forever` note la date d'envoi (`state.last_morning_brief`).

Planification : `seconds_until_morning_brief`, combinée dans
`trading_bot.live.engine.run_forever` avec le cycle du soir (le process se
réveille pour le premier des deux).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from trading_bot.config import AppConfig
from trading_bot.execution.manual_broker import broker_ticker, fmt_eur, fmt_pct
from trading_bot.logger import get_logger
from trading_bot.market_calendar import MarketCalendar
from trading_bot.notify.pushover import PushoverNotifier, truncate_lines
from trading_bot.portfolio.core_satellite import (
    CoreSatelliteParams,
    core_satellite_params,
    max_drift_pts,
    plan_rebalance,
    weights_of,
)
from trading_bot.portfolio.fees import ExecutionRules
from trading_bot.state import LiveState

logger = get_logger()

# (symbole yfinance, libellé) du contexte de marché, dans l'ordre d'affichage.
MARKET_CONTEXT = [
    ("^GSPC", "S&P 500"),
    ("URTH", "MSCI World"),
    ("ES=F", "futures S&P"),
    ("EURUSD=X", "EUR/USD"),
]
MONTHS_SHORT_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
ACTION_KINDS = ("order", "stop", "cancel")


@dataclass
class MorningBrief:
    title: str
    message: str


# --- Planification --------------------------------------------------------------


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _local_time(day: dt.date, hhmm: str, tz: str) -> pd.Timestamp:
    """`hhmm` du jour `day`, heure locale de `tz` (robuste aux changements
    d'heure : on localise l'heure murale, on n'ajoute pas une durée à minuit)."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    naive = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hour, minute=minute)
    return naive.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")


def seconds_until_morning_brief(
    now: pd.Timestamp,
    calendar: MarketCalendar,
    at: str,
    last_brief_date: str | None,
    only_trading_days: bool = True,
    catch_up_until: str = "12:00",
    search_days: int = 21,
) -> tuple[float, str]:
    """(secondes à attendre, date locale ISO de l'aperçu visé). 0 = à envoyer
    maintenant : jour éligible (jour de bourse si `only_trading_days`), pas
    encore envoyé, heure locale >= `at`. Démarré après `at` sans aperçu ce
    jour-là : envoyé tout de suite jusqu'à `catch_up_until`, sauté au-delà
    (le cycle du soir arrive bientôt)."""
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    tz = calendar.timezone
    local_now = now.tz_convert(tz)
    today = local_now.date()
    for offset in range(search_days):
        day = today + dt.timedelta(days=offset)
        day_str = day.isoformat()
        if last_brief_date is not None and day_str <= last_brief_date:
            continue
        if only_trading_days and not calendar.is_trading_day(pd.Timestamp(day)):
            continue
        target = _local_time(day, at, tz)
        if target > local_now:
            return (target - local_now).total_seconds(), day_str
        if offset == 0 and local_now < _local_time(day, catch_up_until, tz):
            return 0.0, day_str
    raise RuntimeError(f"Aucun jour éligible à l'aperçu du matin dans les {search_days} prochains jours.")


# --- Données --------------------------------------------------------------------


def _default_fetch_bars(config: AppConfig, symbols: list[str]) -> dict[str, pd.DataFrame]:
    from trading_bot.live import engine  # import différé : évite l'import circulaire

    return engine._fetch_live_bars(config, symbols)


def fetch_quote_changes(symbols: list[str]) -> dict[str, tuple[float, float]]:
    """{symbole: (dernier cours, variation depuis la clôture précédente)} via
    yfinance. Un symbole en échec est simplement absent (jamais d'exception)."""
    out: dict[str, tuple[float, float]] = {}
    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001 - contexte de marché facultatif
        return out
    for symbol in symbols:
        try:
            closes = yf.Ticker(symbol).history(period="10d")["Close"].dropna()
            if len(closes) < 2:
                continue
            last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
            if math.isfinite(last) and math.isfinite(prev) and prev > 0:
                out[symbol] = (last, last / prev - 1.0)
        except Exception as exc:  # noqa: BLE001 - donnée omise, jamais bloquante
            logger.info("Aperçu du matin : cours de %s indisponible (%s), omis.", symbol, exc)
    return out


def _read_account(path: str) -> dict:
    """Lecture SEULE du fichier de compte manuel."""
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


def _close_frame(
    bars: dict[str, pd.DataFrame], symbols: list[str], calendar: MarketCalendar, now: pd.Timestamp
) -> pd.DataFrame | None:
    """Clôtures alignées des poches, sans la bougie du jour tant que la séance
    n'est pas close (yfinance renvoie alors un cours en cours de séance)."""
    series = {}
    for symbol in symbols:
        df = bars.get(symbol)
        if df is None or df.empty or "close" not in df:
            return None
        # yfinance publie parfois la dernière bougie avec une clôture NaN :
        # on s'en tient aux clôtures réellement publiées, jamais reportées.
        series[symbol] = df["close"].dropna()
    frame = pd.DataFrame(series).sort_index().dropna()
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    session = calendar.session_for(now)
    if session is not None and now < session[1]:
        today = pd.Timestamp(now.tz_convert(calendar.timezone).date())
        frame = frame[frame.index.normalize() < today]
    frame = frame[(frame > 0).all(axis=1)]
    return frame if not frame.empty else None


# --- Mise en forme ----------------------------------------------------------------


def _signed_pct(value: float) -> str:
    return f"{round(value * 100, 1) + 0.0:+.1f}".replace(".", ",") + " %"  # jamais « -0,0 % »


def _signed_eur(value: float) -> str:
    return ("+" if value >= 0 else "") + fmt_eur(value)


def _pts(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") + " pts"


def _date_fr(day) -> str:
    return pd.Timestamp(day).strftime("%d/%m")


def brief_title(config: AppConfig) -> str:
    morning = config.live.morning_brief
    if morning.title:
        return morning.title
    base = config.live.notifications.pushover.title
    return f"{base} — aperçu du matin" if base else "Aperçu du matin"


def _previous_trading_day(calendar: MarketCalendar, day: dt.date) -> dt.date | None:
    end = pd.Timestamp(day) - pd.Timedelta(days=1)
    days = calendar.trading_days(end - pd.Timedelta(days=20), end)
    return days[-1].date() if len(days) else None


def _next_trading_day(calendar: MarketCalendar, day: dt.date) -> dt.date | None:
    start = pd.Timestamp(day) + pd.Timedelta(days=1)
    days = calendar.trading_days(start, start + pd.Timedelta(days=20))
    return days[0].date() if len(days) else None


def _value_lines(frame: pd.DataFrame, qty: dict[str, float], cash: float, label: str = "Valeur") -> list[str]:
    values = cash + (frame * pd.Series(qty)).sum(axis=1)
    last_date = frame.index[-1]
    last = float(values.iloc[-1])
    line = f"{label} à la clôture du {_date_fr(last_date)} : {fmt_eur(last)}"
    invested = any(q > 0 for q in qty.values())
    if not invested:
        return [line + " (cash seul)."]
    if len(values) >= 2 and float(values.iloc[-2]) > 0:
        prev = float(values.iloc[-2])
        line += f" ({_signed_eur(last - prev)}, {_signed_pct(last / prev - 1)} sur la veille)"
    lines = [line + "."]
    perfs = []
    month_ref = values[values.index < last_date.replace(day=1)]
    if not month_ref.empty and float(month_ref.iloc[-1]) > 0:
        perfs.append(f"{MONTHS_SHORT_FR[last_date.month - 1]} {_signed_pct(last / float(month_ref.iloc[-1]) - 1)}")
    year_ref = values[values.index < pd.Timestamp(year=last_date.year, month=1, day=1)]
    if not year_ref.empty and float(year_ref.iloc[-1]) > 0:
        perfs.append(f"{last_date.year} {_signed_pct(last / float(year_ref.iloc[-1]) - 1)}")
    if perfs:
        lines.append("Perf. à positions actuelles : " + " · ".join(perfs) + ".")
    return lines


def _weights_and_cash_lines(
    config: AppConfig, params: CoreSatelliteParams, qty: dict[str, float], prices: dict[str, float], cash: float
) -> list[str]:
    values = {s: qty[s] * prices[s] for s in params.targets}
    equity = cash + sum(values.values())
    weights = weights_of(values, cash)
    lines = []
    if sum(values.values()) > 0:
        sleeves = " · ".join(
            f"{broker_ticker(s)} {fmt_pct(weights[s])} (cible {fmt_pct(t)})" for s, t in params.targets.items()
        )
        drift = max_drift_pts(weights, params.targets)
        verdict = "dans la bande" if drift <= params.drift_threshold_pts + 1e-9 else "rééquilibrage au prochain cycle"
        lines.append(f"Poids du plan : {sleeves}. Dérive max {_pts(drift)} (seuil {params.drift_threshold_pts:g}) : {verdict}.")
    # `plan_rebalance` est pure : on s'en sert pour dire ce que le cash attend.
    rules = ExecutionRules.from_backtest_config(config.backtest)
    plan = plan_rebalance(qty, prices, cash, params, rules)
    deployable = max(0.0, cash - params.cash_buffer(cash))
    threshold = params.contribution_threshold(equity)
    if plan.reason == "waiting":
        status = "aucun achat utile possible, il attend le prochain apport"
    elif deployable >= threshold - 1e-9:
        status = f"au-dessus du seuil d'apport ({fmt_eur(threshold)}) : investi au prochain cycle"
    else:
        status = f"sous le seuil d'apport ({fmt_eur(threshold)}) : il attend le prochain versement"
    lines.append(f"Cash non investi : {fmt_eur(cash)} — {status}.")
    return lines


def _fmt_qty(qty: float) -> str:
    return f"{qty:g}".replace(".", ",")


def _off_plan_lines(
    config: AppConfig,
    positions: dict,
    off_plan: list[str],
    bars: dict[str, pd.DataFrame],
    calendar: MarketCalendar,
    now: pd.Timestamp,
) -> tuple[list[str], float | None]:
    """Une ligne par position hors plan, valorisée à sa dernière clôture
    valide (bougie du jour en cours et clôtures NaN exclues). Renvoie aussi
    leur valeur totale, None si l'une d'elles n'a pas pu être valorisée."""
    labels = config.live.morning_brief.labels or {}
    lines, total = [], 0.0
    for symbol in off_plan:
        pos = positions.get(symbol) or {}
        qty = float(pos.get("qty", 0.0))
        name = f"{_fmt_qty(qty)} {labels.get(symbol) or broker_ticker(symbol)} ({symbol})"
        frame = _close_frame(bars, [symbol], calendar, now)
        if frame is None:
            lines.append(f"Hors plan : {name}, cours indisponible.")
            total = None
            continue
        closes = frame[symbol]
        last = float(closes.iloc[-1])
        value = qty * last
        details = []
        pru = float(pos.get("avg_entry_price", 0.0) or 0.0)
        if math.isfinite(pru) and pru > 0:
            details.append(f"{_signed_pct(last / pru - 1)} vs PRU {fmt_eur(pru)}")
        if len(closes) >= 2:
            details.append(f"{_signed_pct(last / float(closes.iloc[-2]) - 1)} veille")
        lines.append(f"Hors plan : {name} ≈ {fmt_eur(value)}" + (f" ({', '.join(details)})" if details else "") + ".")
        if total is not None:
            total += value
    return lines, total


def _orders_lines(state: LiveState, calendar: MarketCalendar, today: dt.date, trading_today: bool) -> list[str]:
    record = state.last_cycle_orders or {}
    session = record.get("session")
    previous = _previous_trading_day(calendar, today)
    when = "aujourd'hui" if trading_today else "à la prochaine séance"
    lines = []
    if not record and state.last_daily_run:
        lines.append("Ordres de la veille non enregistrés (première utilisation de l'aperçu) : voir le push d'hier soir.")
    elif session and previous and session >= previous.isoformat() and record.get("orders"):
        if trading_today and session == today.isoformat():
            when = "à la prochaine séance"  # aperçu demandé après le cycle du soir
        lines.append(f"Ordres à passer {when} (cycle du {_date_fr(session)}) :")
        lines += [f"- {text.split(' · ')[0]}" for text in record["orders"]]
    else:
        lines.append(f"Aucun ordre à passer {when}.")
    if state.last_daily_run and previous and state.last_daily_run < previous.isoformat():
        lines.append(f"Attention : pas de cycle du soir depuis le {_date_fr(state.last_daily_run)} (bot arrêté ?).")
    return lines


def _alert_line(config: AppConfig, state: LiveState) -> str | None:
    alerts = config.live.alerts
    if not alerts.enabled:
        return None
    active = [lv for lv in alerts.drawdown_levels if state.alert_flags.get(f"plan_drawdown:{lv:g}")]
    if not active:
        return None
    cs = state.core_satellite
    nav, peak = float(cs.get("nav", 1.0)), float(cs.get("nav_peak", 1.0))
    drawdown = 1.0 - nav / peak if peak > 0 else 0.0
    return (
        f"Alerte drawdown active (palier -{fmt_pct(max(active))}) : portefeuille à -{fmt_pct(drawdown)} de son plus "
        "haut hors apports au dernier cycle. Rien à vendre : le plan traverse les baisses, continue les apports."
    )


def _market_line(quotes: dict[str, tuple[float, float]]) -> str | None:
    parts = []
    for symbol, label in MARKET_CONTEXT:
        if symbol not in quotes:
            continue
        last, change = quotes[symbol]
        if symbol == "EURUSD=X":
            parts.append(f"{label} {last:.4f}".replace(".", ",") + f" ({_signed_pct(change)})")
        else:
            parts.append(f"{label} {_signed_pct(change)}")
    return "Marchés : " + " · ".join(parts) + "." if parts else None


# --- Aperçu -----------------------------------------------------------------------


def build_morning_brief(
    config: AppConfig,
    state: LiveState,
    now: pd.Timestamp | None = None,
    fetch_bars=None,
    fetch_quotes=None,
) -> MorningBrief:
    """Construit l'aperçu (lecture seule, voir la docstring du module)."""
    params = core_satellite_params(config)
    if params is None or config.live.broker != "manual":
        raise ValueError(
            "L'aperçu du matin n'est disponible qu'en mode core_satellite avec `live.broker: manual`."
        )
    calendar = MarketCalendar(config.market.calendar)
    now = _now_utc() if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    today = now.tz_convert(calendar.timezone).date()
    trading_today = calendar.is_trading_day(pd.Timestamp(today))
    fetch_bars = fetch_bars or _default_fetch_bars
    fetch_quotes = fetch_quotes or fetch_quote_changes

    account = _read_account(config.live.manual.account_file)
    cash = float(account.get("cash", 0.0))
    positions = account.get("positions", {}) or {}
    sleeves = list(params.targets)
    qty = {s: float((positions.get(s) or {}).get("qty", 0.0)) for s in sleeves}
    # Positions hors plan (jamais achetées, vendues ni rééquilibrées par le bot) :
    # affichées à part, jamais comptées dans les poids ni le cash à investir.
    off_plan = [
        s for s, p in positions.items() if s not in params.targets and float((p or {}).get("qty", 0.0) or 0.0) != 0
    ]

    lines: list[str] = []
    if not trading_today:
        nxt = _next_trading_day(calendar, today)
        lines.append(
            f"Bourse fermée aujourd'hui ({config.market.calendar})"
            + (f", prochaine séance le {_date_fr(nxt)}." if nxt else ".")
        )

    frame = None
    try:
        frame = _close_frame(fetch_bars(config, sleeves) or {}, sleeves, calendar, now)
    except Exception as exc:  # noqa: BLE001 - un aperçu ne doit jamais planter
        logger.warning("Aperçu du matin : cours des poches indisponibles (%s).", exc)
    off_plan_bars: dict[str, pd.DataFrame] = {}
    if off_plan:
        try:
            off_plan_bars = fetch_bars(config, off_plan) or {}
        except Exception as exc:  # noqa: BLE001 - hors plan : jamais bloquant
            logger.info("Aperçu du matin : cours hors plan indisponibles (%s).", exc)
    off_lines, off_total = _off_plan_lines(config, positions, off_plan, off_plan_bars, calendar, now)
    if frame is not None:
        prices = {s: float(frame[s].iloc[-1]) for s in sleeves}
        lines += _value_lines(frame, qty, cash, label="Valeur du plan" if off_plan else "Valeur")
        session = calendar.session_for(now)
        closed_today = session is not None and now >= session[1]
        expected = today if closed_today else _previous_trading_day(calendar, today)
        if expected is not None and frame.index[-1].date() < expected:
            lines.append(f"Clôture du {_date_fr(expected)} pas encore publiée par yfinance.")
        lines += off_lines
        if off_plan and off_total is not None:
            plan_value = cash + sum(qty[s] * prices[s] for s in sleeves)
            lines.append(f"Valeur totale du PEA : {fmt_eur(plan_value + off_total)}.")
        weights_and_cash = _weights_and_cash_lines(config, params, qty, prices, cash)
    else:
        lines.append("Cours des poches indisponibles : valeur et poids non calculés.")
        lines += off_lines
        weights_and_cash = [f"Cash non investi : {fmt_eur(cash)}."]
    lines += weights_and_cash[:-1]
    lines += _orders_lines(state, calendar, today, trading_today)
    lines.append(weights_and_cash[-1])
    alert = _alert_line(config, state)
    if alert:
        lines.append(alert)
    try:
        market = _market_line(fetch_quotes([s for s, _ in MARKET_CONTEXT]) or {})
    except Exception as exc:  # noqa: BLE001 - contexte facultatif
        logger.info("Aperçu du matin : contexte de marché indisponible (%s).", exc)
        market = None
    if market:
        lines.append(market)
    return MorningBrief(title=brief_title(config), message=truncate_lines(lines))


def send_morning_brief(config: AppConfig, brief: MorningBrief, force: bool = False) -> bool:
    """Envoie l'aperçu (priorité `live.morning_brief.priority`). Le titre est
    déjà complet : on n'y préfixe pas une seconde fois le titre des
    notifications. Ne lève jamais (voir `PushoverNotifier.send`)."""
    notifier = PushoverNotifier(replace(config.live.notifications.pushover, title=""))
    return notifier.send(brief.title, brief.message, priority=config.live.morning_brief.priority, force=force)

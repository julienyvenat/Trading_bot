"""Aperçu du matin (`live.morning_brief`) : planification avec le cycle du
soir, contenu, données manquantes, lecture seule, rétrocompatibilité.
Réseau (yfinance, Pushover) et horloge entièrement simulés."""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from trading_bot import cli
from trading_bot.config import ManualBrokerConfig, MorningBriefConfig, load_config
from trading_bot.live import engine as engine_module
from trading_bot.live import morning_brief as mb
from trading_bot.market_calendar import MarketCalendar
from trading_bot.state import LiveState, load_state, save_state

ROOT = Path(__file__).resolve().parents[1]
CONFIG_80_20 = ROOT / "config" / "config_pea_fortuneo_80_20.yaml"
DCAM, PSP5 = "DCAM.PA", "PSP5.PA"
ACCOUNT = {"cash": 56.41, "positions": {DCAM: {"qty": 292, "avg_entry_price": 6.2}, PSP5: {"qty": 7, "avg_entry_price": 59.3}}}
QUOTES = {"^GSPC": (7704.13, -0.0031), "URTH": (207.65, -0.0009), "ES=F": (7761.5, 0.0012), "EURUSD=X": (1.1374, -0.0007)}
ORDER_TEXT = "ACHETER 3 PSP5 — ordre au marché (ou à cours limité 60,30 €) · ≈ 180,00 € au cours de 60,00 €, frais ≈ 1,95 €"
FRIDAY_0830 = pd.Timestamp("2026-09-25 06:30", tz="UTC")  # vendredi 08:30 à Paris


@pytest.fixture(scope="module")
def xpar():
    return MarketCalendar("XPAR")


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setattr("trading_bot.config.load_dotenv", lambda: None)


def _bars(last_day: str = "2026-09-24") -> dict[str, pd.DataFrame]:
    """DCAM 5 € (2025) -> 5,5 € (janv.-août 2026) -> 6 € -> 6,1 € le dernier
    jour ; PSP5 50 -> 55 -> 60 €."""
    index = pd.bdate_range("2025-11-03", last_day)
    last = pd.Timestamp(last_day)

    def closes(p2025, p2026, p_sep, p_last):
        values = [
            p2025 if d.year == 2025 else (p2026 if d.month < 9 else (p_last if d == last else p_sep)) for d in index
        ]
        s = pd.Series(values, index=index, dtype=float)
        return pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1})

    return {DCAM: closes(5.0, 5.5, 6.0, 6.1), PSP5: closes(50.0, 55.0, 60.0, 60.0)}


@pytest.fixture
def setup(tmp_path):
    account = tmp_path / "account.json"
    account.write_text(json.dumps(ACCOUNT))
    config = load_config(CONFIG_80_20)
    config.live = replace(
        config.live,
        state_file=str(tmp_path / "state.json"),
        track_record_file=str(tmp_path / "track.json"),
        manual=ManualBrokerConfig(account_file=str(account)),
    )
    return config, account


def _brief(config, state=None, now=FRIDAY_0830, bars=None, quotes=QUOTES):
    return mb.build_morning_brief(
        config,
        state or LiveState(),
        now=now,
        fetch_bars=lambda config, symbols: _bars() if bars is None else bars,
        fetch_quotes=lambda symbols: dict(quotes),
    )


# --- Planification (fonction pure) ----------------------------------------------------


def test_brief_waits_until_0830_paris(xpar):
    wait, day = mb.seconds_until_morning_brief(pd.Timestamp("2026-09-24 06:00", tz="UTC"), xpar, "08:30", None)
    assert day == "2026-09-24" and wait == pytest.approx(1800)


def test_restart_after_0830_catches_up_once_then_next_day(xpar):
    now = pd.Timestamp("2026-09-24 07:30", tz="UTC")  # 09:30 à Paris, aucun aperçu envoyé
    assert mb.seconds_until_morning_brief(now, xpar, "08:30", "2026-09-23") == (0.0, "2026-09-24")
    # Déjà envoyé aujourd'hui (état persisté) : pas de second envoi, prochain = vendredi 08:30.
    wait, day = mb.seconds_until_morning_brief(now, xpar, "08:30", "2026-09-24")
    assert day == "2026-09-25"
    assert wait == pytest.approx((pd.Timestamp("2026-09-25 06:30", tz="UTC") - now).total_seconds())
    # Redémarré après `catch_up_until` : sauté pour aujourd'hui.
    late = pd.Timestamp("2026-09-24 11:00", tz="UTC")  # 13:00 à Paris
    assert mb.seconds_until_morning_brief(late, xpar, "08:30", None)[1] == "2026-09-25"


def test_brief_skips_weekend_and_holidays(xpar):
    wait, day = mb.seconds_until_morning_brief(pd.Timestamp("2026-09-26 10:00", tz="UTC"), xpar, "08:30", "2026-09-25")
    assert day == "2026-09-28"
    # 25/12 (vendredi) et 01/01 fermés.
    assert mb.seconds_until_morning_brief(pd.Timestamp("2026-12-24 20:00", tz="UTC"), xpar, "08:30", "2026-12-24")[1] == "2026-12-28"
    assert mb.seconds_until_morning_brief(pd.Timestamp("2026-12-31 20:00", tz="UTC"), xpar, "08:30", "2026-12-31")[1] == "2027-01-04"


def test_brief_on_non_trading_days_when_requested(xpar):
    now = pd.Timestamp("2026-09-26 05:00", tz="UTC")  # samedi 07:00 à Paris
    wait, day = mb.seconds_until_morning_brief(now, xpar, "08:30", "2026-09-25", only_trading_days=False)
    assert day == "2026-09-26" and wait == pytest.approx(1.5 * 3600)


@pytest.mark.parametrize(
    "now, last, expected_utc",
    [
        # Passage à l'heure d'hiver le dimanche 25/10/2026 : lundi 08:30 CET = 07:30 UTC.
        ("2026-10-23 20:00", "2026-10-23", "2026-10-26 07:30"),
        # Passage à l'heure d'été le dimanche 29/03/2026 : lundi 08:30 CEST = 06:30 UTC.
        ("2026-03-27 20:00", "2026-03-27", "2026-03-30 06:30"),
    ],
)
def test_brief_time_is_local_across_dst(xpar, now, last, expected_utc):
    now = pd.Timestamp(now, tz="UTC")
    wait, _ = mb.seconds_until_morning_brief(now, xpar, "08:30", last)
    assert now + pd.Timedelta(seconds=wait) == pd.Timestamp(expected_utc, tz="UTC")


def test_brief_on_dst_sunday_when_not_only_trading_days(xpar):
    now = pd.Timestamp("2026-10-24 20:00", tz="UTC")
    wait, day = mb.seconds_until_morning_brief(now, xpar, "08:30", "2026-10-24", only_trading_days=False)
    assert day == "2026-10-25"
    assert now + pd.Timedelta(seconds=wait) == pd.Timestamp("2026-10-25 07:30", tz="UTC")


# --- Planification dans run_forever (horloge simulée) ------------------------------------


class _Stop(BaseException):
    pass


def _simulate(monkeypatch, config, start, end, state=None, orders_on=("2026-09-24",)):
    """Fait tourner `run_forever` sur une horloge simulée de `start` à `end` ;
    renvoie les événements (type, heure locale de Paris, message)."""
    clock = {"now": pd.Timestamp(start, tz="UTC")}
    events = []
    if state is not None:
        save_state(config.live.state_file, state)

    def _sleep(seconds):
        assert seconds > 0
        clock["now"] += pd.Timedelta(seconds=seconds)
        if clock["now"] > pd.Timestamp(end, tz="UTC"):
            raise _Stop

    def _fake_run_once(config, broker, dry_run, state):
        local = clock["now"].tz_convert("Europe/Paris")
        events.append(("soir", local.strftime("%a %d/%m %H:%M"), None))
        if local.date().isoformat() in orders_on:
            broker.add_note(ORDER_TEXT, kind="order")
        return state

    def _fake_send(config, brief, force=False):
        events.append(("matin", clock["now"].tz_convert("Europe/Paris").strftime("%a %d/%m %H:%M"), brief.message))
        return True

    monkeypatch.setattr(engine_module, "_now_utc", lambda: clock["now"])
    monkeypatch.setattr(engine_module.time, "sleep", _sleep)
    monkeypatch.setattr(engine_module, "run_once", _fake_run_once)
    monkeypatch.setattr(engine_module, "send_morning_brief", _fake_send)
    monkeypatch.setattr(engine_module, "_fetch_live_bars", lambda config, symbols: _bars())
    monkeypatch.setattr(mb, "fetch_quote_changes", lambda symbols: dict(QUOTES))
    with pytest.raises(_Stop):
        engine_module.run_forever(config)
    return events


def test_run_forever_interleaves_morning_brief_and_evening_cycle(monkeypatch, setup):
    config, account = setup
    before = account.read_bytes()
    events = _simulate(monkeypatch, config, "2026-09-24 05:00", "2026-09-29 05:00")
    assert [(kind, when) for kind, when, _ in events] == [
        ("matin", "Thu 24/09 08:30"),
        ("soir", "Thu 24/09 18:30"),
        ("matin", "Fri 25/09 08:30"),
        ("soir", "Fri 25/09 18:30"),
        ("matin", "Mon 28/09 08:30"),  # week-end sauté
        ("soir", "Mon 28/09 18:30"),
    ]
    friday, monday = events[2][2], events[4][2]
    assert "Ordres à passer aujourd'hui (cycle du 24/09) :\n- ACHETER 3 PSP5 — ordre au marché (ou à cours limité 60,30 €)\n" in friday
    assert "Aucun ordre à passer aujourd'hui." in monday  # le cycle de vendredi n'a rien poussé
    state = load_state(config.live.state_file)
    assert state.last_morning_brief == "2026-09-28" and state.last_daily_run == "2026-09-28"
    assert state.last_cycle_orders == {"session": "2026-09-28", "orders": []}
    assert account.read_bytes() == before  # l'aperçu ne touche jamais au compte


@pytest.mark.parametrize(
    "start, last_brief, first_events",
    [
        # Redémarré à 09:30 après l'aperçu du jour : pas de renvoi.
        ("2026-09-24 07:30", "2026-09-24", [("soir", "Thu 24/09 18:30"), ("matin", "Fri 25/09 08:30")]),
        # Redémarré à 09:30 sans aperçu ce jour : envoyé tout de suite, une fois.
        ("2026-09-24 07:30", "2026-09-23", [("matin", "Thu 24/09 09:30"), ("soir", "Thu 24/09 18:30")]),
        # Redémarré à 13:00 (après catch_up_until) : sauté jusqu'au lendemain.
        ("2026-09-24 11:00", None, [("soir", "Thu 24/09 18:30"), ("matin", "Fri 25/09 08:30")]),
        # Démarré avant 08:30 : envoyé à 08:30.
        ("2026-09-24 04:10", None, [("matin", "Thu 24/09 08:30"), ("soir", "Thu 24/09 18:30")]),
    ],
)
def test_run_forever_restart(monkeypatch, setup, start, last_brief, first_events):
    config, _ = setup
    state = LiveState(last_morning_brief=last_brief, last_daily_run="2026-09-23")
    events = _simulate(monkeypatch, config, start, "2026-09-25 07:00", state=state)
    assert [(k, w) for k, w, _ in events][:2] == first_events


def test_run_forever_brief_failure_does_not_block_evening_nor_retry(monkeypatch, setup):
    config, _ = setup
    monkeypatch.setattr(engine_module, "build_morning_brief", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    events = _simulate(monkeypatch, config, "2026-09-24 05:00", "2026-09-25 05:00")
    assert [(k, w) for k, w, _ in events] == [("soir", "Thu 24/09 18:30")]
    assert load_state(config.live.state_file).last_morning_brief == "2026-09-24"


def test_run_forever_without_morning_brief_is_unchanged(monkeypatch, setup):
    config, _ = setup
    config.live = replace(config.live, morning_brief=MorningBriefConfig())
    events = _simulate(monkeypatch, config, "2026-09-24 05:00", "2026-09-25 17:00")
    assert [(k, w) for k, w, _ in events] == [("soir", "Thu 24/09 18:30"), ("soir", "Fri 25/09 18:30")]
    assert load_state(config.live.state_file).last_morning_brief is None


def test_morning_brief_ignored_outside_core_satellite_manual(setup):
    config, _ = setup
    assert engine_module._morning_brief_usable(config)
    assert not engine_module._morning_brief_usable(replace(config, live=replace(config.live, daily_run_after=None)))
    assert not engine_module._morning_brief_usable(replace(config, live=replace(config.live, broker="alpaca")))


def test_remember_cycle_orders_appends_on_retry_of_same_session():
    from trading_bot.execution.manual_broker import ManualInstruction

    state = LiveState()
    engine_module.remember_cycle_orders(state, "2026-09-24", [ManualInstruction("info", "x"), ManualInstruction("order", "A")])
    engine_module.remember_cycle_orders(state, "2026-09-24", [ManualInstruction("stop", "B")])
    assert state.last_cycle_orders == {"session": "2026-09-24", "orders": ["A", "B"]}
    engine_module.remember_cycle_orders(state, "2026-09-25", [])
    assert state.last_cycle_orders == {"session": "2026-09-25", "orders": []}


# --- Contenu ------------------------------------------------------------------------------


def test_brief_content_without_pending_orders(setup):
    config, _ = setup
    state = LiveState(last_daily_run="2026-09-24", last_cycle_orders={"session": "2026-09-24", "orders": []})
    brief = _brief(config, state)
    assert brief.title == "PEA Fortuneo 80/20 — aperçu du matin"
    assert brief.message.split("\n") == [
        "Valeur à la clôture du 24/09 : 2 257,61 € (+29,20 €, +1,3 % sur la veille).",
        "Perf. à positions actuelles : sept. +10,3 % · 2026 +21,0 %.",
        "Poids du plan : DCAM 78,9 % (cible 80,0 %) · PSP5 18,6 % (cible 20,0 %). Dérive max 1,4 pts (seuil 5) : dans la bande.",
        "Aucun ordre à passer aujourd'hui.",
        "Cash non investi : 56,41 € — sous le seuil d'apport (200,00 €) : il attend le prochain versement.",
        "Marchés : S&P 500 -0,3 % · MSCI World -0,1 % · futures S&P +0,1 % · EUR/USD 1,1374 (-0,1 %).",
    ]
    assert len(brief.message) < 1024


def test_brief_content_with_pending_orders(setup):
    config, _ = setup
    orders = [ORDER_TEXT, "ACHETER 30 DCAM — ordre au marché (ou à cours limité 6,13 €) · ≈ 183,00 € au cours de 6,10 €, frais ≈ 1,95 €"]
    state = LiveState(last_daily_run="2026-09-24", last_cycle_orders={"session": "2026-09-24", "orders": orders})
    lines = _brief(config, state).message.split("\n")
    i = lines.index("Ordres à passer aujourd'hui (cycle du 24/09) :")
    assert lines[i + 1 : i + 3] == [
        "- ACHETER 3 PSP5 — ordre au marché (ou à cours limité 60,30 €)",
        "- ACHETER 30 DCAM — ordre au marché (ou à cours limité 6,13 €)",
    ]
    assert "Aucun ordre" not in "\n".join(lines)


def test_brief_ignores_stale_orders_and_warns_about_missed_evening_cycle(setup):
    config, _ = setup
    state = LiveState(last_daily_run="2026-09-22", last_cycle_orders={"session": "2026-09-22", "orders": [ORDER_TEXT]})
    message = _brief(config, state).message
    assert "Aucun ordre à passer aujourd'hui." in message
    assert "Attention : pas de cycle du soir depuis le 22/09 (bot arrêté ?)." in message


def test_brief_after_upgrade_without_recorded_orders(setup):
    config, _ = setup
    message = _brief(config, LiveState(last_daily_run="2026-09-24")).message
    assert "Ordres de la veille non enregistrés (première utilisation de l'aperçu) : voir le push d'hier soir." in message


def test_brief_cash_only_account_above_threshold(setup):
    config, account = setup
    account.write_text(json.dumps({"cash": 2306.59, "positions": {}}))
    lines = _brief(config).message.split("\n")
    assert lines[0] == "Valeur à la clôture du 24/09 : 2 306,59 € (cash seul)."
    assert "Cash non investi : 2 306,59 € — au-dessus du seuil d'apport (200,00 €) : investi au prochain cycle." in lines
    assert not any(line.startswith("Poids") for line in lines)
    assert not any("Hors plan" in line or "Valeur totale" in line for line in lines)


def test_brief_drift_above_threshold(setup):
    config, account = setup
    account.write_text(json.dumps({"cash": 0.0, "positions": {DCAM: {"qty": 350, "avg_entry_price": 6.0}}}))
    message = _brief(config).message
    assert "Dérive max 20,0 pts (seuil 5) : rééquilibrage au prochain cycle." in message


def test_brief_drops_in_progress_and_nan_bars(setup):
    config, _ = setup
    bars = _bars("2026-09-25")  # bougie du jour (en cours de séance à 08:30)
    for df in bars.values():
        df.loc[pd.Timestamp("2026-09-24"), "close"] = float("nan")  # clôture pas encore publiée par yfinance
    message = _brief(config, bars=bars).message
    assert message.startswith("Valeur à la clôture du 23/09 : 2 228,41 €")
    assert "Clôture du 24/09 pas encore publiée par yfinance." in message
    assert "pas encore publiée" not in _brief(config).message


def test_brief_data_failures_are_omitted(setup):
    config, _ = setup

    def _boom(config, symbols):
        raise RuntimeError("yfinance down")

    brief = mb.build_morning_brief(config, LiveState(), now=FRIDAY_0830, fetch_bars=_boom, fetch_quotes=lambda s: {"URTH": QUOTES["URTH"]})
    lines = brief.message.split("\n")
    assert lines[0] == "Cours des poches indisponibles : valeur et poids non calculés."
    assert "Cash non investi : 56,41 €." in lines
    assert lines[-1] == "Marchés : MSCI World -0,1 %."

    def _quotes_boom(symbols):
        raise RuntimeError("réseau")

    message = mb.build_morning_brief(config, LiveState(), now=FRIDAY_0830, fetch_bars=lambda c, s: _bars(), fetch_quotes=_quotes_boom).message
    assert "Marchés" not in message and message.startswith("Valeur à la clôture du 24/09")
    # Une poche sans données : pas de valeur partielle trompeuse.
    assert _brief(config, bars={DCAM: _bars()[DCAM]}).message.startswith("Cours des poches indisponibles")


def test_fetch_quote_changes_omits_failing_symbols(monkeypatch):
    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period="10d"):
            if self.symbol == "ES=F":
                raise RuntimeError("timeout")
            if self.symbol == "URTH":
                return pd.DataFrame({"Close": [207.0]})  # une seule clôture : pas de variation
            return pd.DataFrame({"Close": [100.0, 101.0, float("nan")]})

    monkeypatch.setattr("yfinance.Ticker", _Ticker)
    quotes = mb.fetch_quote_changes(["^GSPC", "URTH", "ES=F", "EURUSD=X"])
    assert set(quotes) == {"^GSPC", "EURUSD=X"}
    assert quotes["^GSPC"] == (101.0, pytest.approx(0.01))


# --- Positions hors plan (24 AXA gardées à côté du plan) ------------------------------------

AXA = "CS.PA"
AXA_POSITION = {"qty": 24, "avg_entry_price": 20.242}


def _axa_bars(last_day: str = "2026-09-24", prev: float = 43.90, last: float = 43.77) -> pd.DataFrame:
    index = pd.bdate_range("2026-09-01", last_day)
    s = pd.Series([prev] * (len(index) - 1) + [last], index=index, dtype=float)
    return pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1})


def _with_axa(account, cash=None):
    data = json.loads(json.dumps(ACCOUNT)) if cash is None else {"cash": cash, "positions": {}}
    data["positions"][AXA] = dict(AXA_POSITION)
    account.write_text(json.dumps(data))


def _fetch_with_axa(axa=None, fail_axa=False):
    calls = []

    def fetch(config, symbols):
        calls.append(list(symbols))
        if AXA in symbols:
            if fail_axa:
                raise RuntimeError("yfinance down pour CS.PA")
            return {AXA: _axa_bars() if axa is None else axa}
        return _bars()

    return fetch, calls


def _brief_axa(config, state=None, quotes=None, **kwargs):
    fetch, calls = _fetch_with_axa(**kwargs)
    brief = mb.build_morning_brief(
        config, state or LiveState(), now=FRIDAY_0830, fetch_bars=fetch, fetch_quotes=lambda s: dict(quotes or {})
    )
    return brief, calls


def test_brief_shows_off_plan_axa_and_total_pea_value(setup):
    config, account = setup
    _with_axa(account)
    state = LiveState(last_daily_run="2026-09-24", last_cycle_orders={"session": "2026-09-24", "orders": []})
    brief, calls = _brief_axa(config, state, quotes=QUOTES)
    assert calls == [[DCAM, PSP5], [AXA]]  # poches seules d'abord : AXA n'entre pas dans les calculs du plan
    assert brief.message.split("\n") == [
        "Valeur du plan à la clôture du 24/09 : 2 257,61 € (+29,20 €, +1,3 % sur la veille).",
        "Perf. à positions actuelles : sept. +10,3 % · 2026 +21,0 %.",
        "Hors plan : 24 AXA (CS.PA) ≈ 1 050,48 € (+116,2 % vs PRU 20,24 €, -0,3 % veille).",
        "Valeur totale du PEA : 3 308,09 €.",
        # Poids, dérive et cash : strictement identiques à l'aperçu sans AXA.
        "Poids du plan : DCAM 78,9 % (cible 80,0 %) · PSP5 18,6 % (cible 20,0 %). Dérive max 1,4 pts (seuil 5) : dans la bande.",
        "Aucun ordre à passer aujourd'hui.",
        "Cash non investi : 56,41 € — sous le seuil d'apport (200,00 €) : il attend le prochain versement.",
        "Marchés : S&P 500 -0,3 % · MSCI World -0,1 % · futures S&P +0,1 % · EUR/USD 1,1374 (-0,1 %).",
    ]
    assert len(brief.message) <= 1024


def test_brief_off_plan_label_falls_back_to_ticker(setup):
    config, account = setup
    config.live = replace(config.live, morning_brief=replace(config.live.morning_brief, labels={}))
    _with_axa(account)
    assert "Hors plan : 24 CS (CS.PA) ≈ 1 050,48 €" in _brief_axa(config)[0].message


def test_brief_off_plan_price_failure_is_graceful(setup):
    config, account = setup
    plan_lines = [line for line in _brief(config).message.split("\n") if line.startswith(("Poids", "Cash"))]
    _with_axa(account)
    lines = _brief_axa(config, fail_axa=True)[0].message.split("\n")
    assert "Hors plan : 24 AXA (CS.PA), cours indisponible." in lines
    assert not any(line.startswith("Valeur totale") for line in lines)  # jamais de total partiel
    assert lines[0].startswith("Valeur du plan à la clôture du 24/09 : 2 257,61 €")
    assert [line for line in lines if line.startswith(("Poids", "Cash"))] == plan_lines
    # Clôture NaN / bougie du jour en cours : dernière clôture valide, jamais NaN.
    axa = _axa_bars("2026-09-25")
    axa.loc[pd.Timestamp("2026-09-24"), "close"] = float("nan")
    message = _brief_axa(config, axa=axa)[0].message
    assert "Hors plan : 24 AXA (CS.PA) ≈ 1 053,60 € (+116,9 % vs PRU 20,24 €, +0,0 % veille)." in message
    assert "nan" not in message.lower()
    # Aucune clôture exploitable : même repli.
    axa = _axa_bars()
    axa["close"] = float("nan")
    assert "Hors plan : 24 AXA (CS.PA), cours indisponible." in _brief_axa(config, axa=axa)[0].message


def test_brief_off_plan_with_cash_only_plan(setup):
    """Le compte réel aujourd'hui : 2 306,59 € de cash + 24 AXA."""
    config, account = setup
    _with_axa(account, cash=2306.59)
    lines = _brief_axa(config)[0].message.split("\n")
    assert lines[:3] == [
        "Valeur du plan à la clôture du 24/09 : 2 306,59 € (cash seul).",
        "Hors plan : 24 AXA (CS.PA) ≈ 1 050,48 € (+116,2 % vs PRU 20,24 €, -0,3 % veille).",
        "Valeur totale du PEA : 3 357,07 €.",
    ]
    # Le seuil d'apport reste calculé sur le plan seul.
    assert "Cash non investi : 2 306,59 € — au-dessus du seuil d'apport (200,00 €) : investi au prochain cycle." in lines


def test_brief_with_off_plan_is_read_only_and_under_limit(setup):
    config, account = setup
    _with_axa(account)
    state = LiveState(last_daily_run="2026-09-24", last_cycle_orders={"session": "2026-09-24", "orders": [ORDER_TEXT] * 30})
    dict_before, account_before = json.dumps(state.to_dict(), sort_keys=True), account.read_bytes()
    message = _brief_axa(config, state, quotes=QUOTES)[0].message
    assert len(message) <= 1024 and "Valeur totale du PEA" in message
    assert account.read_bytes() == account_before
    assert json.dumps(state.to_dict(), sort_keys=True) == dict_before


def test_brief_active_drawdown_alert_is_calm_and_read_only(setup):
    config, _ = setup
    state = LiveState(
        alert_flags={"plan_drawdown:0.2": True, "plan_drawdown:0.35": False},
        core_satellite={"units": 100.0, "nav": 0.78, "nav_peak": 1.0},
    )
    before = json.dumps(state.to_dict(), sort_keys=True)
    message = _brief(config, state).message
    assert (
        "Alerte drawdown active (palier -20,0 %) : portefeuille à -22,0 % de son plus haut hors apports au dernier "
        "cycle. Rien à vendre : le plan traverse les baisses, continue les apports."
    ) in message
    assert json.dumps(state.to_dict(), sort_keys=True) == before  # flags ni consommés ni modifiés


def test_brief_on_closed_market_day(setup):
    config, _ = setup
    config.live = replace(config.live, morning_brief=replace(config.live.morning_brief, only_trading_days=False))
    state = LiveState(last_daily_run="2026-09-25", last_cycle_orders={"session": "2026-09-25", "orders": [ORDER_TEXT]})
    lines = _brief(config, state, now=pd.Timestamp("2026-09-26 06:30", tz="UTC")).message.split("\n")
    assert lines[0] == "Bourse fermée aujourd'hui (XPAR), prochaine séance le 28/09."
    assert "Ordres à passer à la prochaine séance (cycle du 25/09) :" in lines


def test_brief_is_read_only(setup):
    config, account = setup
    state = LiveState(
        last_daily_run="2026-09-24",
        last_cycle_orders={"session": "2026-09-24", "orders": [ORDER_TEXT]},
        core_satellite={"units": 2000.0, "nav": 1.1, "nav_peak": 1.2, "last_cash": 56.41, "last_qty": {DCAM: 292.0}},
        alert_flags={"core_satellite_no_stop_explained": True},
    )
    save_state(config.live.state_file, state)
    account_before, state_before = account.read_bytes(), Path(config.live.state_file).read_bytes()
    dict_before = json.dumps(state.to_dict(), sort_keys=True)
    _brief(config, state)
    assert account.read_bytes() == account_before
    assert Path(config.live.state_file).read_bytes() == state_before
    assert json.dumps(state.to_dict(), sort_keys=True) == dict_before


def test_brief_stays_under_pushover_limit(setup):
    config, _ = setup
    orders = [ORDER_TEXT] * 30
    state = LiveState(last_daily_run="2026-09-24", last_cycle_orders={"session": "2026-09-24", "orders": orders})
    assert len(_brief(config, state).message) <= 1024


def test_brief_title_override_and_refused_outside_core_satellite(setup):
    config, _ = setup
    config.live = replace(config.live, morning_brief=replace(config.live.morning_brief, title="PEA — matin"))
    assert _brief(config).title == "PEA — matin"
    with pytest.raises(ValueError, match="core_satellite"):
        _brief(replace(config, live=replace(config.live, broker="alpaca")))


# --- Configuration et état : rétrocompatibilité ---------------------------------------------


def test_morning_brief_config_defaults_and_shipped_configs():
    assert MorningBriefConfig() == MorningBriefConfig(
        enabled=False, at="08:30", only_trading_days=True, priority=-1, title=None, catch_up_until="12:00", labels={}
    )
    for path in sorted((ROOT / "config").glob("*.yaml")):
        config = load_config(path)
        assert config.live.morning_brief.enabled == (path.name == "config_pea_fortuneo_80_20.yaml"), path.name
    brief = load_config(CONFIG_80_20).live.morning_brief
    assert (brief.at, brief.priority, brief.only_trading_days) == ("08:30", -1, True)
    assert brief.labels == {"CS.PA": "AXA"}


def test_old_state_file_without_morning_brief_fields_loads(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"alert_flags": {"x": True}, "last_daily_run": "2026-09-24", "core_satellite": {"nav": 1.0}}))
    state = load_state(path)
    assert state.last_morning_brief is None and state.last_cycle_orders == {}
    state.last_morning_brief = "2026-09-25"
    state.last_cycle_orders = {"session": "2026-09-24", "orders": ["A"]}
    save_state(path, state)
    again = load_state(path)
    assert again.last_morning_brief == "2026-09-25" and again.last_cycle_orders == {"session": "2026-09-24", "orders": ["A"]}


# --- CLI ------------------------------------------------------------------------------------


def _write_cli_config(tmp_path, account) -> Path:
    text = CONFIG_80_20.read_text(encoding="utf-8")
    text = text.replace("state/manual_account_fortuneo.json", str(account))
    text = text.replace("state/live_state_pea_80_20.json", str(tmp_path / "state.json"))
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    account = tmp_path / "account.json"
    account.write_text(json.dumps(ACCOUNT))
    monkeypatch.setattr(engine_module, "_fetch_live_bars", lambda config, symbols: _bars())
    monkeypatch.setattr(mb, "fetch_quote_changes", lambda symbols: dict(QUOTES))
    monkeypatch.setattr(mb, "_now_utc", lambda: FRIDAY_0830)
    return _write_cli_config(tmp_path, account), account


def test_cli_morning_brief_print_sends_nothing(cli_env, capsys, tmp_path):
    config_path, account = cli_env
    before = account.read_bytes()
    cli.main(["morning-brief", "--config", str(config_path), "--print"])
    out = capsys.readouterr().out
    assert out.startswith("PEA Fortuneo 80/20 — aperçu du matin\nValeur à la clôture du 24/09 : 2 257,61 €")
    assert account.read_bytes() == before
    assert not (tmp_path / "state.json").exists()


def test_cli_morning_brief_sends_quiet_push(cli_env, monkeypatch):
    config_path, _ = cli_env
    calls = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"status": 1}'

    def _urlopen(request, timeout=None):
        calls.append(dict(urllib.parse.parse_qsl(request.data.decode("utf-8"))))
        return _Response()

    monkeypatch.setattr("trading_bot.notify.pushover.urllib.request.urlopen", _urlopen)
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "app-token")
    monkeypatch.setenv("PUSHOVER_USER_KEY", "user-key")
    cli.main(["morning-brief", "--config", str(config_path)])
    assert len(calls) == 1
    assert calls[0]["title"] == "PEA Fortuneo 80/20 — aperçu du matin"
    assert calls[0]["priority"] == "-1"
    assert calls[0]["message"].startswith("Valeur à la clôture du 24/09")

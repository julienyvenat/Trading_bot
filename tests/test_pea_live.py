from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_bot.config import AlertsConfig, ManualBrokerConfig
from trading_bot.execution.manual_broker import ManualBroker, broker_ticker, fmt_eur, fmt_pct
from trading_bot.live import engine as engine_module
from trading_bot.market_calendar import MarketCalendar
from trading_bot.portfolio.fees import CommissionModel
from trading_bot.state import LiveState, load_state, save_state

from conftest import make_ohlcv
from test_pea_backtest import FORTUNEO, make_buyhold_config


class _Prices:
    """Faux yfinance.Ticker : dernier cours piloté par le test."""

    current: dict[str, float] = {}

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def history(self, period: str = "5d"):
        return pd.DataFrame({"Close": [self.current[self.symbol]]})


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr("yfinance.Ticker", _Prices)
    account = tmp_path / "account.json"
    account.write_text(json.dumps({"cash": 2306.59, "positions": {}}))
    config = make_buyhold_config(stop_mode="trailing_pct", trailing_stop_pct=0.20, stop_reentry="sma", stop_reentry_sma_window=50)
    config.live = replace(
        config.live,
        broker="manual",
        state_file=str(tmp_path / "state.json"),
        track_record_file=str(tmp_path / "track.json"),
        manual=ManualBrokerConfig(account_file=str(account), native_trailing_stop=True),
    )
    holder = {"df": make_ohlcv(np.linspace(50.0, 59.42, 260))}
    monkeypatch.setattr(engine_module, "_fetch_live_bars", lambda config, symbols: {"ETF": holder["df"]})

    def set_bars(df):
        holder["df"] = df
        _Prices.current = {"ETF": float(df["close"].iloc[-1])}

    set_bars(holder["df"])
    return config, account, holder, set_bars


def _broker(config) -> ManualBroker:
    return engine_module.build_broker(config)


def _append_bar(df: pd.DataFrame, open_, high, low, close) -> pd.DataFrame:
    dt = df.index[-1] + pd.offsets.BDay(1)
    row = pd.DataFrame({"open": [open_], "high": [high], "low": [low], "close": [close], "volume": [1]}, index=[dt])
    return pd.concat([df, row])


def test_native_trailing_stop_notified_once_then_silent_then_detected(env):
    config, account, holder, set_bars = env
    broker = _broker(config)

    state = engine_module.run_once(config, broker, dry_run=False, state=LiveState())
    texts = [(i.kind, i.text) for i in broker.drain_instructions()]
    assert texts[0][0] == "order"
    assert texts[0][1].startswith("ACHETER 38 ETF — ordre au marché (ou à cours limité 59,72 €)")
    assert "frais ≈ 4,52 €" in texts[0][1]
    assert texts[1] == (
        "stop",
        "Une fois l'achat exécuté, poser un STOP SUIVEUR : vendre 38 ETF, écart 20,0 % (≈ 11,88 €), "
        "seuil de départ 47,54 €",
    )
    assert len(texts) == 2
    data = json.loads(account.read_text())
    assert data["positions"]["ETF"]["qty"] == 38
    assert data["cash"] == pytest.approx(2306.59 - 38 * 59.42 - 38 * 59.42 * 0.002)
    assert state.native_stops["ETF"]["trail_pct"] == 0.20

    # Séance suivante, nouveau plus haut sans toucher le seuil : RIEN à notifier
    # (le courtier remonte lui-même le stop), mais le suivi interne ratchete.
    set_bars(_append_bar(holder["df"], 59.5, 62.0, 59.0, 61.5))
    state = engine_module.run_once(config, broker, dry_run=False, state=state)
    assert broker.drain_instructions() == []
    assert state.native_stops["ETF"]["high_water"] == pytest.approx(62.0)

    # Krach : le plus bas passe sous 62 x 0,8 = 49,60 -> sortie comptabilisée,
    # alerte "à vérifier", et pas de rachat (clôture sous la SMA50).
    set_bars(_append_bar(holder["df"], 55.0, 55.5, 48.0, 48.5))
    state = engine_module.run_once(config, broker, dry_run=False, state=state)
    instructions = broker.drain_instructions()
    assert [i.kind for i in instructions] == ["alert"]
    assert "STOP SUIVEUR ETF probablement déclenché" in instructions[0].text
    data = json.loads(account.read_text())
    assert data["positions"] == {}
    assert "ETF" in state.stopped_out and "ETF" not in state.native_stops

    # Tant que la clôture reste sous la SMA50 : pas de ré-entrée.
    set_bars(_append_bar(holder["df"], 48.5, 49.0, 48.0, 48.8))
    state = engine_module.run_once(config, broker, dry_run=False, state=state)
    assert broker.drain_instructions() == []


def test_strategy_exit_cancels_native_stop_before_selling(env, monkeypatch):
    config, account, holder, set_bars = env
    broker = _broker(config)
    state = engine_module.run_once(config, broker, dry_run=False, state=LiveState())
    broker.drain_instructions()

    # La stratégie ne veut plus rien détenir : annulation du stop PUIS vente.
    monkeypatch.setattr(
        engine_module.SignalAllocator, "latest_target_exposures", lambda self, data: {"ETF": 0.0}
    )
    state = engine_module.run_once(config, broker, dry_run=False, state=state)
    kinds = [(i.kind, i.text.split(" ")[0]) for i in broker.drain_instructions()]
    assert kinds == [("cancel", "Annuler"), ("order", "VENDRE")]
    assert state.native_stops == {}


def test_stop_mode_none_says_no_stop(env):
    config, account, holder, set_bars = env
    config.risk = replace(config.risk, stop_mode="none")
    config.live = replace(config.live, alerts=AlertsConfig(enabled=True, symbol="ETF", sma_window=50))
    broker = _broker(config)
    state = engine_module.run_once(config, broker, dry_run=False, state=LiveState())
    instructions = broker.drain_instructions()
    assert [i.kind for i in instructions] == ["order", "info"]
    assert instructions[1].text.startswith("Pas de stop à poser sur ETF : risk.stop_mode vaut none dans cette config")
    assert "PSP5" not in instructions[1].text  # neutre vis-à-vis de la config (pas de backtest cité)
    assert state.native_stops == {}

    # Cycle suivant sans changement : aucune notification.
    state = engine_module.run_once(config, broker, dry_run=False, state=state)
    assert broker.drain_instructions() == []


def test_alerts_only_on_crossing(env):
    config, account, holder, set_bars = env
    config.risk = replace(config.risk, stop_mode="none")
    config.live = replace(config.live, alerts=AlertsConfig(enabled=True, symbol="ETF", sma_window=50, drawdown_pct=0.2))
    broker = _broker(config)
    state = engine_module.run_once(config, broker, dry_run=False, state=LiveState())
    broker.drain_instructions()

    set_bars(_append_bar(holder["df"], 57.0, 57.0, 44.0, 45.0))  # sous la SMA50 et -24 % sur la position
    state = engine_module.run_once(config, broker, dry_run=False, state=state)
    alerts = [i for i in broker.drain_instructions() if i.kind == "alert"]
    assert any("SOUS sa SMA50" in a.text for a in alerts)
    assert any("de son plus haut" in a.text for a in alerts)

    set_bars(_append_bar(holder["df"], 45.0, 45.5, 44.5, 45.2))
    state = engine_module.run_once(config, broker, dry_run=False, state=state)
    assert broker.drain_instructions() == []  # pas de répétition

    set_bars(_append_bar(holder["df"], 58.0, 62.0, 58.0, 61.0))
    state = engine_module.run_once(config, broker, dry_run=False, state=state)
    infos = broker.drain_instructions()
    assert any("repassé au-dessus" in i.text for i in infos)


def test_stop_for_existing_position_without_buy_this_cycle(env, tmp_path):
    """Position déjà détenue au démarrage (aucun achat ce cycle) : simple
    "Poser un STOP SUIVEUR", sans référence à un achat."""
    config, account, holder, set_bars = env
    account.write_text(json.dumps({"cash": 50.0, "positions": {"ETF": {"qty": 38, "avg_entry_price": 55.0}}}))
    broker = _broker(config)
    engine_module.run_once(config, broker, dry_run=False, state=LiveState())
    texts = [i.text for i in broker.drain_instructions()]
    assert texts == ["Poser un STOP SUIVEUR : vendre 38 ETF, écart 20,0 % (≈ 11,88 €), seuil de départ 47,54 €"]


def test_trailing_pct_without_native_manual_broker_is_rejected(env):
    config, *_ = env
    config.live = replace(config.live, manual=replace(config.live.manual, native_trailing_stop=False))
    with pytest.raises(ValueError, match="trailing_pct"):
        engine_module._effective_stop_mode(config)


def test_native_mode_converts_atr_stop_to_fixed_pct(env):
    config, *_ = env
    config.risk = replace(config.risk, stop_mode="atr")
    assert engine_module._effective_stop_mode(config) == ("trailing_pct", True)


def test_state_roundtrip_with_pea_fields(tmp_path):
    state = LiveState(
        native_stops={"ETF": {"trail_pct": 0.2, "high_water": 60.0, "qty": 38, "last_date": "2026-09-24"}},
        stopped_out={"X": "2026-01-02"},
        alert_flags={"drawdown": True},
        last_daily_run="2026-09-24",
    )
    path = tmp_path / "s.json"
    save_state(path, state)
    loaded = load_state(path)
    assert loaded.native_stops == state.native_stops
    assert loaded.stopped_out == state.stopped_out
    assert loaded.alert_flags == state.alert_flags
    assert loaded.last_daily_run == "2026-09-24"
    # Rétrocompatibilité : un ancien fichier d'état sans ces champs se charge.
    path.write_text(json.dumps({"trailing_stops": {}}))
    assert load_state(path).native_stops == {}


# --- ManualBroker ------------------------------------------------------------


def test_manual_broker_fees_and_format(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _Prices)
    _Prices.current = {"PSP5.PA": 59.42}
    account = tmp_path / "a.json"
    account.write_text(json.dumps({"cash": 1000.0, "positions": {}}))
    broker = ManualBroker(str(account), commission=CommissionModel.from_schedule(FORTUNEO))

    broker.submit_market_order("PSP5.PA", 5, "buy")  # 297,10 € -> 0,5 % = 1,49 € -> min 1,95 €
    text = broker.drain_instructions()[0].text
    assert text == (
        "ACHETER 5 PSP5 — ordre au marché (ou à cours limité 59,72 €) · ≈ 297,10 € au cours de 59,42 €, frais ≈ 1,95 €"
    )
    assert json.loads(account.read_text())["cash"] == pytest.approx(1000 - 297.10 - 1.95)

    broker.submit_market_order("PSP5.PA", 5, "sell")
    assert "VENDRE 5 PSP5 — ordre au marché (ou à cours limité 59,12 €)" in broker.drain_instructions()[0].text
    assert json.loads(account.read_text())["cash"] == pytest.approx(1000 - 2 * 1.95)


def test_manual_broker_price_cache_is_cleared(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _Prices)
    _Prices.current = {"X.PA": 10.0}
    account = tmp_path / "a.json"
    account.write_text(json.dumps({"cash": 0.0, "positions": {}}))
    broker = ManualBroker(str(account))
    assert broker.get_last_price("X.PA") == 10.0
    _Prices.current = {"X.PA": 11.0}
    assert broker.get_last_price("X.PA") == 10.0
    broker.clear_price_cache()
    assert broker.get_last_price("X.PA") == 11.0


def test_formatters():
    assert fmt_eur(2259.861) == "2 259,86 €"
    assert fmt_pct(0.125) == "12,5 %"
    assert broker_ticker("PSP5.PA") == "PSP5"


def test_notification_title_for_alert_only(monkeypatch, tmp_path):
    account = tmp_path / "a.json"
    account.write_text(json.dumps({"cash": 0.0, "positions": {}}))
    broker = ManualBroker(str(account))
    broker.add_note("PSP5 sous sa SMA200", kind="alert")
    sent = []

    class _N:
        def send(self, title, message, priority=None, force=False):
            sent.append((title, message))
            return True

    engine_module.notify_cycle_instructions(_N(), broker)
    assert sent == [("Alerte (à vérifier)", "ALERTE : PSP5 sous sa SMA200")]


# --- Cycle quotidien ------------------------------------------------------------


@pytest.fixture(scope="module")
def xpar():
    return MarketCalendar("XPAR")


def test_daily_run_waits_until_after_close(xpar):
    # Jeudi 24/09/2026 16:00 UTC = 18:00 à Paris : cycle à 18:30 -> 30 min.
    wait, day = engine_module.seconds_until_daily_run(pd.Timestamp("2026-09-24 16:00", tz="UTC"), xpar, "18:30", None)
    assert day == "2026-09-24" and wait == pytest.approx(1800)


def test_daily_run_now_then_next_trading_day(xpar):
    now = pd.Timestamp("2026-09-25 17:00", tz="UTC")  # vendredi 19:00 à Paris
    wait, day = engine_module.seconds_until_daily_run(now, xpar, "18:30", "2026-09-24")
    assert wait == 0 and day == "2026-09-25"
    # Déjà exécuté ce vendredi : prochain = lundi 28/09 18:30 Paris (16:30 UTC).
    wait, day = engine_module.seconds_until_daily_run(now, xpar, "18:30", "2026-09-25")
    assert day == "2026-09-28"
    assert wait == pytest.approx((pd.Timestamp("2026-09-28 16:30", tz="UTC") - now).total_seconds())


def test_daily_run_skips_holidays(xpar):
    # 25/12/2026 (vendredi) : Euronext fermé -> lundi 28/12.
    wait, day = engine_module.seconds_until_daily_run(pd.Timestamp("2026-12-24 20:00", tz="UTC"), xpar, "18:30", "2026-12-24")
    assert day == "2026-12-28"


class _Stop(BaseException):
    pass


def test_run_forever_daily_mode_runs_once_per_session(monkeypatch, env):
    config, *_ = env
    config.live = replace(config.live, daily_run_after="18:30")
    calls = []
    waits = iter([(0.0, "2026-09-24"), (3600.0, "2026-09-25")])

    def _fake_wait(now, calendar, run_after, last_run):
        try:
            return next(waits)
        except StopIteration:
            raise _Stop

    def _fake_run_once(config, broker, dry_run, state):
        calls.append(state.last_daily_run)
        return state

    monkeypatch.setattr(engine_module, "seconds_until_daily_run", _fake_wait)
    monkeypatch.setattr(engine_module, "run_once", _fake_run_once)
    monkeypatch.setattr(engine_module.time, "sleep", lambda s: None)
    with pytest.raises(_Stop):
        engine_module.run_forever(config)
    assert calls == [None]
    assert load_state(config.live.state_file).last_daily_run == "2026-09-24"

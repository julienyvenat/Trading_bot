from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.parse

import pandas as pd
import pytest

from trading_bot import cli
from trading_bot.config import PushoverConfig, load_config
from trading_bot.execution.manual_broker import ManualBroker
from trading_bot.live import engine as engine_module
from trading_bot.notify import pushover as pushover_module
from trading_bot.notify.pushover import MAX_MESSAGE_CHARS, PushoverNotifier, truncate_lines
from trading_bot.portfolio.circuit_breaker import RiskState

from test_config import _MINIMAL_YAML


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class _FakeUrlopen:
    """Faux `urllib.request.urlopen` : enregistre chaque requête, renvoie
    `{"status": 1}` (ou lève `error` si fourni)."""

    def __init__(self, body: dict | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._body = body if body is not None else {"status": 1, "request": "abc"}
        self._error = error

    def __call__(self, request, timeout=None):
        payload = dict(urllib.parse.parse_qsl(request.data.decode("utf-8")))
        self.calls.append({"url": request.full_url, "method": request.get_method(), "payload": payload})
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._body)


@pytest.fixture
def fake_urlopen(monkeypatch):
    fake = _FakeUrlopen()
    monkeypatch.setattr(pushover_module.urllib.request, "urlopen", fake)
    return fake


def _notifier(**overrides) -> PushoverNotifier:
    config = PushoverConfig(enabled=True, title="Trading Bot PEA", **overrides)
    return PushoverNotifier(config, app_token="app-token", user_key="user-key")


class _FakeTicker:
    _prices = {"TTE.PA": 80.28, "MC.PA": 780.0}

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol

    def history(self, period: str = "5d"):
        return pd.DataFrame({"Close": [self._prices[self._symbol]]})


@pytest.fixture
def manual_broker(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    account_file.write_text(json.dumps({"cash": 10_000.0, "positions": {}}))
    return ManualBroker(str(account_file))


# --- Configuration ---------------------------------------------------------


def test_pushover_disabled_by_default_when_section_absent(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_MINIMAL_YAML)

    config = load_config(config_file)

    assert config.live.notifications.pushover.enabled is False


def test_pushover_section_is_parsed(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        _MINIMAL_YAML
        + "\n  notifications:\n    pushover:\n      enabled: true\n      priority: -1\n      title: \"PEA\"\n"
    )

    pushover = load_config(config_file).live.notifications.pushover

    assert pushover.enabled is True
    assert pushover.priority == -1
    assert pushover.title == "PEA"
    assert pushover.alert_priority == 1


def test_pea_example_config_enables_pushover():
    from test_config import REPO_ROOT

    config = load_config(REPO_ROOT / "config" / "config_pea_fortuneo.example.yaml")

    assert config.live.notifications.pushover.enabled is True


def test_credentials_read_from_environment(monkeypatch):
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "tok")
    monkeypatch.setenv("PUSHOVER_USER_KEY", "usr")

    notifier = PushoverNotifier(PushoverConfig(enabled=True))

    assert notifier.has_credentials
    assert notifier.enabled


# --- Troncature -------------------------------------------------------------


def test_truncate_lines_leaves_short_message_untouched():
    assert truncate_lines(["a", "b"]) == "a\nb"


def test_truncate_lines_keeps_whole_lines_under_limit():
    lines = [f"ORDRE : ACHETER {i} TTE.PA (≈ 642.24 € au dernier cours de 80.28 €)" for i in range(60)]

    text = truncate_lines(lines)

    assert len(text) <= MAX_MESSAGE_CHARS
    kept = text.split("\n")[:-1]
    assert kept == lines[: len(kept)]  # aucune ligne coupée au milieu
    assert text.endswith(f"… (+{len(lines) - len(kept)} ligne(s), voir les logs)")


def test_truncate_lines_cuts_single_oversized_line():
    text = truncate_lines(["x" * 5000])

    assert len(text) == MAX_MESSAGE_CHARS
    assert text.endswith("…")


# --- Envoi -----------------------------------------------------------------


def test_send_posts_expected_payload(fake_urlopen):
    assert _notifier(priority=-1).send("2 ordre(s) à passer", "ACHETER 8 TTE.PA") is True

    assert len(fake_urlopen.calls) == 1
    call = fake_urlopen.calls[0]
    assert call["url"] == "https://api.pushover.net/1/messages.json"
    assert call["method"] == "POST"
    assert call["payload"] == {
        "token": "app-token",
        "user": "user-key",
        "title": "Trading Bot PEA — 2 ordre(s) à passer",
        "message": "ACHETER 8 TTE.PA",
        "priority": "-1",
    }


def test_send_emergency_priority_adds_retry_and_expire(fake_urlopen):
    _notifier().send("x", "y", priority=2)

    payload = fake_urlopen.calls[0]["payload"]
    assert payload["priority"] == "2"
    assert "retry" in payload and "expire" in payload


def test_send_does_nothing_when_disabled(fake_urlopen):
    notifier = PushoverNotifier(PushoverConfig(), app_token="a", user_key="u")

    assert notifier.send("x", "y") is False
    assert fake_urlopen.calls == []


def test_send_with_missing_credentials_warns_and_skips(fake_urlopen, caplog):
    notifier = PushoverNotifier(PushoverConfig(enabled=True))

    with caplog.at_level(logging.WARNING, logger="trading_bot"):
        notifier.warn_if_misconfigured()
        assert notifier.send("x", "y") is False

    assert fake_urlopen.calls == []
    assert "PUSHOVER_APP_TOKEN" in caplog.text


def test_no_startup_warning_when_disabled(caplog):
    with caplog.at_level(logging.WARNING, logger="trading_bot"):
        PushoverNotifier(PushoverConfig()).warn_if_misconfigured()

    assert caplog.text == ""


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError(
            "https://api.pushover.net/1/messages.json", 400, "Bad Request", {}, io.BytesIO(b'{"status":0}')
        ),
        urllib.error.URLError("réseau indisponible"),
        TimeoutError("timeout"),
    ],
)
def test_send_failure_never_raises(monkeypatch, caplog, error):
    fake = _FakeUrlopen(error=error)
    monkeypatch.setattr(pushover_module.urllib.request, "urlopen", fake)

    with caplog.at_level(logging.WARNING, logger="trading_bot"):
        assert _notifier().send("x", "y") is False

    assert len(fake.calls) == 1
    assert "Pushover" in caplog.text
    assert "app-token" not in caplog.text  # le token ne doit jamais fuiter dans les logs


def test_send_returns_false_when_api_rejects(monkeypatch):
    monkeypatch.setattr(
        pushover_module.urllib.request, "urlopen", _FakeUrlopen(body={"status": 0, "errors": ["user invalid"]})
    )

    assert _notifier().send("x", "y") is False


# --- Récapitulatif par cycle -------------------------------------------------


def test_one_push_per_cycle_batches_orders_and_stops(fake_urlopen, manual_broker):
    manual_broker.submit_market_order("TTE.PA", 8.7, "buy")
    manual_broker.submit_market_order("MC.PA", 1, "buy")
    manual_broker.cancel_order("manual-TTE.PA-1700000000")
    manual_broker.submit_stop_order("TTE.PA", 8, "sell", 76.42)

    engine_module.notify_cycle_instructions(_notifier(), manual_broker)

    assert len(fake_urlopen.calls) == 1
    payload = fake_urlopen.calls[0]["payload"]
    assert payload["title"] == "Trading Bot PEA — 3 ordre(s) à passer"
    assert payload["message"].split("\n") == [
        "ORDRE : ACHETER 8 TTE — ordre au marché (ou à cours limité 80,68 €) · ≈ 642,24 € au cours de 80,28 €, "
        "frais ≈ 0,00 €",
        "ORDRE : ACHETER 1 MC — ordre au marché (ou à cours limité 783,90 €) · ≈ 780,00 € au cours de 780,00 €, "
        "frais ≈ 0,00 €",
        "Annuler le stop précédent sur TTE.PA",
        "STOP : Poser un ordre STOP : vendre 8 TTE, seuil de déclenchement 76,42 €",
    ]

    # Instructions vidées après envoi : le cycle suivant sans ordre n'envoie rien.
    engine_module.notify_cycle_instructions(_notifier(), manual_broker)
    assert len(fake_urlopen.calls) == 1


def test_no_push_when_nothing_to_do(fake_urlopen, manual_broker):
    manual_broker.submit_market_order("TTE.PA", 0.4, "buy")  # arrondi à 0 action : ignoré

    engine_module.notify_cycle_instructions(_notifier(), manual_broker)

    assert fake_urlopen.calls == []


def test_no_push_for_broker_without_manual_instructions(fake_urlopen):
    engine_module.notify_cycle_instructions(_notifier(), object())

    assert fake_urlopen.calls == []


def test_large_batch_is_truncated_under_pushover_limit(fake_urlopen, manual_broker):
    for i in range(80):
        manual_broker.submit_stop_order("TTE.PA", 8 + i, "sell", 76.42)

    engine_module.notify_cycle_instructions(_notifier(), manual_broker)

    payload = fake_urlopen.calls[0]["payload"]
    assert payload["title"] == "Trading Bot PEA — 80 ordre(s) à passer"
    assert len(payload["message"]) <= MAX_MESSAGE_CHARS
    assert "ligne(s), voir les logs" in payload["message"]


def test_batch_push_failure_does_not_raise(monkeypatch, manual_broker):
    monkeypatch.setattr(
        pushover_module.urllib.request, "urlopen", _FakeUrlopen(error=urllib.error.URLError("down"))
    )
    manual_broker.submit_stop_order("TTE.PA", 8, "sell", 76.42)

    engine_module.notify_cycle_instructions(_notifier(), manual_broker)  # ne doit pas lever


# --- Alertes -----------------------------------------------------------------


def _risk(daily: bool = False, drawdown: bool = False) -> RiskState:
    return RiskState(equity_peak=100.0, session_start_equity=100.0, daily_halted=daily, drawdown_halted=drawdown)


def test_alert_sent_when_circuit_breaker_triggers(fake_urlopen):
    engine_module.notify_risk_transitions(_notifier(), _risk(), _risk(drawdown=True))

    assert len(fake_urlopen.calls) == 1
    payload = fake_urlopen.calls[0]["payload"]
    assert "DRAWDOWN" in payload["title"]
    assert payload["priority"] == "1"


def test_no_alert_while_circuit_breaker_stays_active(fake_urlopen):
    engine_module.notify_risk_transitions(_notifier(), _risk(daily=True), _risk(daily=True))
    engine_module.notify_risk_transitions(_notifier(), _risk(), _risk())

    assert fake_urlopen.calls == []


def test_daily_alert_on_first_cycle_without_previous_state(fake_urlopen):
    engine_module.notify_risk_transitions(_notifier(), None, _risk(daily=True))

    assert len(fake_urlopen.calls) == 1


class _StopLoop(BaseException):
    pass


def test_run_forever_alerts_once_on_repeated_crash_and_still_pushes_orders(
    monkeypatch, tmp_path, fake_urlopen, manual_broker
):
    from test_live_engine import make_config

    config = make_config()
    config.live.trade_only_when_market_open = False
    config.live.state_file = str(tmp_path / "state.json")
    config.live.broker = "manual"
    config.live.notifications.pushover = PushoverConfig(enabled=True, title="Trading Bot PEA")
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "app-token")
    monkeypatch.setenv("PUSHOVER_USER_KEY", "user-key")

    def _crashing_run_once(config, broker, dry_run, state):
        broker.submit_stop_order("TTE.PA", 8, "sell", 76.42)  # affiché avant le crash
        raise RuntimeError("yfinance down")

    sleeps = []

    def _sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise _StopLoop

    monkeypatch.setattr(engine_module, "build_broker", lambda config: manual_broker)
    monkeypatch.setattr(engine_module, "run_once", _crashing_run_once)
    monkeypatch.setattr(engine_module.time, "sleep", _sleep)

    with pytest.raises(_StopLoop):
        engine_module.run_forever(config)

    titles = [call["payload"]["title"] for call in fake_urlopen.calls]
    assert titles.count("Trading Bot PEA — Erreur de cycle") == 1  # pas de spam sur erreur persistante
    assert titles.count("Trading Bot PEA — 1 ordre(s) à passer") == 2  # ordres affichés malgré le crash


# --- CLI notify-test --------------------------------------------------------


def _write_pea_like_config(tmp_path, enabled: bool = True):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        _MINIMAL_YAML + f"\n  notifications:\n    pushover:\n      enabled: {str(enabled).lower()}\n"
    )
    return config_file


def test_notify_test_sends_push(monkeypatch, tmp_path, fake_urlopen):
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "app-token")
    monkeypatch.setenv("PUSHOVER_USER_KEY", "user-key")

    cli.main(["notify-test", "--config", str(_write_pea_like_config(tmp_path, enabled=False))])

    assert len(fake_urlopen.calls) == 1
    assert fake_urlopen.calls[0]["payload"]["title"].endswith("Test")


def test_notify_test_exits_when_credentials_missing(monkeypatch, tmp_path, fake_urlopen):
    monkeypatch.setattr("trading_bot.config.load_dotenv", lambda: None)  # ignore un éventuel .env local

    with pytest.raises(SystemExit):
        cli.main(["notify-test", "--config", str(_write_pea_like_config(tmp_path))])

    assert fake_urlopen.calls == []


def test_notify_test_exits_when_push_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "app-token")
    monkeypatch.setenv("PUSHOVER_USER_KEY", "user-key")
    monkeypatch.setattr(
        pushover_module.urllib.request, "urlopen", _FakeUrlopen(body={"status": 0, "errors": ["token invalid"]})
    )

    with pytest.raises(SystemExit):
        cli.main(["notify-test", "--config", str(_write_pea_like_config(tmp_path))])


def test_manual_broker_drain_instructions_is_empty_initially(manual_broker):
    assert manual_broker.drain_instructions() == []

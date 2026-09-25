from __future__ import annotations

import json

import pandas as pd
import pytest

from trading_bot.execution.manual_broker import ManualBroker


class _FakeTicker:
    _prices = {"MC.PA": 780.0, "OR.PA": 400.0}

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol

    def history(self, period: str = "5d"):
        return pd.DataFrame({"Close": [self._prices[self._symbol]]})


def _write_account(path, cash: float, positions: dict | None = None) -> None:
    path.write_text(json.dumps({"cash": cash, "positions": positions or {}}))


def test_missing_account_file_raises_with_bootstrap_instructions(tmp_path):
    with pytest.raises(RuntimeError, match="introuvable"):
        ManualBroker(str(tmp_path / "does_not_exist.json"))


def test_get_account_sums_cash_and_positions_market_value(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    _write_account(account_file, cash=1000.0, positions={"MC.PA": {"qty": 2, "avg_entry_price": 750.0}})

    broker = ManualBroker(str(account_file))
    account = broker.get_account()

    assert account.cash == 1000.0
    assert account.equity == pytest.approx(1000.0 + 2 * 780.0)
    assert account.buying_power == account.cash


def test_get_positions_reflects_account_file(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    _write_account(account_file, cash=0.0, positions={"MC.PA": {"qty": 2, "avg_entry_price": 750.0}})

    broker = ManualBroker(str(account_file))
    positions = broker.get_positions()

    assert positions["MC.PA"].qty == 2
    assert positions["MC.PA"].avg_entry_price == 750.0
    assert positions["MC.PA"].market_value == pytest.approx(2 * 780.0)


def test_submit_market_order_buy_updates_cash_and_position(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    _write_account(account_file, cash=10_000.0)

    broker = ManualBroker(str(account_file))
    broker.submit_market_order("MC.PA", qty=2, side="buy")

    data = json.loads(account_file.read_text())
    assert data["cash"] == pytest.approx(10_000.0 - 2 * 780.0)
    assert data["positions"]["MC.PA"]["qty"] == 2
    assert data["positions"]["MC.PA"]["avg_entry_price"] == pytest.approx(780.0)


def test_submit_market_order_sell_closes_position_and_credits_cash(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    _write_account(account_file, cash=0.0, positions={"MC.PA": {"qty": 2, "avg_entry_price": 750.0}})

    broker = ManualBroker(str(account_file))
    broker.submit_market_order("MC.PA", qty=2, side="sell")

    data = json.loads(account_file.read_text())
    assert data["cash"] == pytest.approx(2 * 780.0)
    assert "MC.PA" not in data["positions"]


def test_submit_market_order_averages_entry_price_on_partial_buy(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    _write_account(account_file, cash=10_000.0, positions={"MC.PA": {"qty": 2, "avg_entry_price": 700.0}})

    broker = ManualBroker(str(account_file))
    broker.submit_market_order("MC.PA", qty=1, side="buy")

    data = json.loads(account_file.read_text())
    assert data["positions"]["MC.PA"]["qty"] == 3
    # (2 * 700 + 1 * 780) / 3
    assert data["positions"]["MC.PA"]["avg_entry_price"] == pytest.approx((2 * 700.0 + 780.0) / 3)


def test_submit_market_order_rounds_fractional_qty_down_to_whole_shares(tmp_path, monkeypatch):
    """Contrairement à Alpaca, un PEA ne se négocie qu'en actions entières :
    une quantité fractionnaire (ex: dimensionnement ATR) doit être arrondie
    vers le bas plutôt que d'afficher un ordre impossible à exécuter chez le
    courtier."""
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    _write_account(account_file, cash=10_000.0)

    broker = ManualBroker(str(account_file))
    broker.submit_market_order("MC.PA", qty=2.9, side="buy")

    data = json.loads(account_file.read_text())
    assert data["positions"]["MC.PA"]["qty"] == 2
    assert data["cash"] == pytest.approx(10_000.0 - 2 * 780.0)


def test_submit_market_order_skips_order_rounding_to_zero_shares(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    _write_account(account_file, cash=10_000.0)

    broker = ManualBroker(str(account_file))
    broker.submit_market_order("MC.PA", qty=0.5, side="buy")

    data = json.loads(account_file.read_text())
    assert data["cash"] == 10_000.0
    assert data["positions"] == {}


def test_submit_stop_order_never_calls_a_real_broker_and_returns_a_local_id(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    _write_account(account_file, cash=0.0, positions={"MC.PA": {"qty": 2, "avg_entry_price": 750.0}})

    broker = ManualBroker(str(account_file))
    order_id = broker.submit_stop_order("MC.PA", qty=2, side="sell", stop_price=720.0)

    assert order_id.startswith("manual-MC.PA-")
    # Aucun ordre réel n'est passé : le fichier de compte n'est pas modifié par un stop.
    data = json.loads(account_file.read_text())
    assert data["positions"]["MC.PA"]["qty"] == 2


def test_cancel_order_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    _write_account(account_file, cash=5000.0)

    broker = ManualBroker(str(account_file))
    broker.cancel_order("manual-MC.PA-123")  # ne doit pas lever

    assert json.loads(account_file.read_text())["cash"] == 5000.0


def test_is_market_open_uses_euronext_paris_calendar_by_default(tmp_path, monkeypatch):
    account_file = tmp_path / "account.json"
    _write_account(account_file, cash=5000.0)
    seen_calendar_names = []

    class _FakeCalendar:
        def __init__(self, name: str) -> None:
            seen_calendar_names.append(name)

        def is_open_now(self) -> bool:
            return True

    monkeypatch.setattr("trading_bot.market_calendar.MarketCalendar", _FakeCalendar)

    broker = ManualBroker(str(account_file))
    assert broker.is_market_open() is True
    assert seen_calendar_names == ["XPAR"]


def test_unmanaged_position_without_price_is_valued_at_entry_price(tmp_path, monkeypatch):
    """Mode core_satellite : une position hors plan (ex. AXA) sans cours ne
    fait pas échouer le cycle, elle est valorisée à son PRU ; une poche du
    plan sans cours échoue toujours (jamais de prix inventé pour un ordre)."""
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    account_file = tmp_path / "account.json"
    positions = {"MC.PA": {"qty": 2, "avg_entry_price": 750.0}, "CS.PA": {"qty": 24, "avg_entry_price": 20.242}}
    _write_account(account_file, cash=100.0, positions=positions)

    broker = ManualBroker(str(account_file), managed_symbols=["MC.PA"])
    assert broker.get_positions()["CS.PA"].market_value == pytest.approx(24 * 20.242)
    assert broker.get_account().equity == pytest.approx(100.0 + 2 * 780.0 + 24 * 20.242)

    # Sans `managed_symbols` (autres modes) ou pour une poche gérée : inchangé, l'erreur remonte.
    with pytest.raises(KeyError):
        ManualBroker(str(account_file)).get_account()
    with pytest.raises(KeyError):
        ManualBroker(str(account_file), managed_symbols=["MC.PA", "CS.PA"]).get_positions()

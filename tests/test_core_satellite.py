from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.metrics import max_drawdown_recovery, money_weighted_return, worst_rolling_return
from trading_bot.config import AlertsConfig, ManualBrokerConfig, StrategyConfig, load_config
from trading_bot.live import engine as engine_module
from trading_bot.market_calendar import MarketCalendar
from trading_bot.portfolio.core_satellite import (
    CoreSatelliteParams,
    allocate_cash,
    calendar_rebalance_due,
    core_satellite_params,
    plan_rebalance,
)
from trading_bot.portfolio.fees import CommissionModel, ExecutionRules
from trading_bot.state import LiveState

import itertools

from conftest import make_ohlcv
from test_pea_backtest import FORTUNEO, make_buyhold_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "config" / "config_pea_core_satellite.example.yaml"
W, SP, LEV = "DCAM.PA", "PSP5.PA", "CL2.PA"
TARGETS = {W: 0.55, SP: 0.20, LEV: 0.25}
PRICES = {W: 6.23, SP: 59.31, LEV: 32.44}
RULES = ExecutionRules(
    commission=CommissionModel.from_schedule(FORTUNEO), whole_shares=True, min_order_value=150, max_fee_pct=0.02
)
PARAMS = CoreSatelliteParams(targets=TARGETS)


def _qty_at_target(total: float) -> dict[str, float]:
    return {s: float(int(total * w / PRICES[s])) for s, w in TARGETS.items()}


def _cost(orders) -> float:
    fee = RULES.commission.fee
    return sum((1 if o.side == "buy" else -1) * o.notional_value + fee(o.notional_value) for o in orders)


# --- Paramètres ---------------------------------------------------------------


def test_params_validation():
    with pytest.raises(ValueError, match="somme"):
        CoreSatelliteParams.from_params({"targets": {W: 0.5, SP: 0.2}})
    with pytest.raises(ValueError, match="inconnu"):
        CoreSatelliteParams.from_params({"targets": TARGETS, "drift": 5})
    with pytest.raises(ValueError, match="obligatoire"):
        CoreSatelliteParams.from_params({})
    params = CoreSatelliteParams.from_params({"targets": TARGETS, "annual_rebalance_month": None})
    assert params.annual_rebalance_month is None and params.drift_threshold_pts == 5.0


def test_core_satellite_must_be_the_only_strategy():
    config = load_config(EXAMPLE)
    config.strategies.append(StrategyConfig(name="buy_and_hold", enabled=True, weight=1.0))
    with pytest.raises(ValueError, match="seule active"):
        core_satellite_params(config)


# --- Décision (fonction pure) --------------------------------------------------


def test_no_action_within_band():
    qty = _qty_at_target(20000)
    cash = 60.0  # < seuil d'apport max(200, 1 %)
    qty[LEV] += 20  # dérive de ~3 pts < 5
    plan = plan_rebalance(qty, PRICES, cash, PARAMS, RULES)
    assert plan.orders == [] and plan.reason is None
    assert 0 < plan.max_drift_before_pts < 5


def test_initial_cash_is_deployed_without_sells_in_whole_shares():
    plan = plan_rebalance({}, PRICES, 2306.59, PARAMS, RULES)
    assert plan.reason == "contribution" and not plan.sells
    assert {o.symbol: o.qty for o in plan.orders} == {W: 200.0, SP: 8.0, LEV: 17.0}
    assert plan.max_drift_after_pts < 1.5
    assert all(o.side == "buy" and o.qty == int(o.qty) for o in plan.orders)
    assert _cost(plan.orders) <= 2306.59 - PARAMS.cash_buffer(2306.59) + 1e-9
    assert all(abs(plan.weights_after[s] - t) < 0.02 for s, t in TARGETS.items())


def test_contribution_goes_to_most_underweight_sleeve():
    qty = _qty_at_target(20000)
    qty[LEV] -= 30  # CL2 en retard de ~5 pts... mais sous le seuil de dérive
    qty[W] += 50
    plan = plan_rebalance(qty, PRICES, 250.0, PARAMS, RULES)
    assert plan.reason == "contribution" and not plan.sells
    # 250 € ne permettent qu'un ordre >= 150 € : tout va à la poche la plus en retard.
    assert [(o.symbol, o.side) for o in plan.orders] == [(LEV, "buy")]
    assert plan.orders[0].notional_value <= 250 - PARAMS.cash_buffer(250)


def test_contribution_below_threshold_waits():
    qty = _qty_at_target(20000)
    plan = plan_rebalance(qty, PRICES, 150.0, PARAMS, RULES)  # seuil = max(200, 1 % x ~20 k€)
    assert plan.orders == []


def test_allocate_cash_small_amount_never_exceeds_a_sleeve_gap():
    values = {W: 0.0, SP: 0.0, LEV: 0.0}
    buys = allocate_cash(values, PRICES, TARGETS, 300.0, RULES)
    # Au prorata, aucune poche n'atteindrait 150 € ; seule DCAM (écart 165 €)
    # peut recevoir un ordre valable, et jamais plus que son écart + 1 action.
    assert set(buys) == {W}
    assert 150 <= buys[W] * PRICES[W] <= 0.55 * 300 + PRICES[W]


def test_drift_fixed_by_cash_first_then_by_sales():
    qty = _qty_at_target(20000)
    qty[LEV] = float(int(qty[LEV] * 1.45))  # CL2 bondit : poids ~33 %, dérive > 5 pts
    # a) assez de cash pour corriger par les seuls achats : aucune vente.
    plan = plan_rebalance(qty, PRICES, 6000.0, PARAMS, RULES)
    assert plan.reason in ("drift", "contribution") and not plan.sells
    assert all(o.side == "buy" for o in plan.orders)
    # b) sans cash : vente de CL2 ramené à sa cible, produit réinvesti dans le reste.
    plan = plan_rebalance(qty, PRICES, 20.0, PARAMS, RULES)
    assert plan.reason == "drift" and plan.sells
    assert plan.orders[0].side == "sell" and plan.orders[0].symbol == LEV
    assert {o.symbol for o in plan.orders if o.side == "buy"} <= {W, SP}
    assert all(o.qty == int(o.qty) for o in plan.orders)
    assert abs(plan.weights_after[LEV] - 0.25) < 0.01
    assert plan.cash_after >= 0


def test_annual_rebalance_trades_small_drift_only_above_min():
    qty = _qty_at_target(20000)
    qty[LEV] = float(int(qty[LEV] * 1.15))  # dérive ~3 pts : rien hors date annuelle
    assert plan_rebalance(qty, PRICES, 20.0, PARAMS, RULES).orders == []
    plan = plan_rebalance(qty, PRICES, 20.0, PARAMS, RULES, calendar_due=True)
    assert plan.reason == "calendar" and plan.sells and plan.orders
    # Dérive sous calendar_min_drift_pts : même à la date annuelle, rien à faire.
    assert plan_rebalance(_qty_at_target(20000), PRICES, 20.0, PARAMS, RULES, calendar_due=True).orders == []


def test_calendar_rebalance_due_first_trading_day_and_missed_days():
    xpar = MarketCalendar("XPAR")
    params = PARAMS
    assert not calendar_rebalance_due(params, pd.Timestamp("2027-01-01"), 2026, xpar)  # férié
    assert calendar_rebalance_due(params, pd.Timestamp("2027-01-04"), 2026, xpar)  # 1er jour de bourse
    assert calendar_rebalance_due(params, pd.Timestamp("2027-03-10"), 2026, xpar)  # cycle manqué : rattrapé
    assert not calendar_rebalance_due(params, pd.Timestamp("2027-03-10"), 2027, xpar)  # déjà fait
    assert not calendar_rebalance_due(replace(params, annual_rebalance_month=None), pd.Timestamp("2027-01-04"), 2026, xpar)


# --- Métriques -----------------------------------------------------------------


def test_money_weighted_return_and_recovery_helpers():
    flows = [(pd.Timestamp("2020-01-01"), 100.0)]
    irr = money_weighted_return(flows, pd.Timestamp("2020-01-01") + pd.Timedelta(days=round(2 * 365.25)), 121.0)
    assert irr == pytest.approx(0.10, abs=1e-4)
    curve = pd.Series([1.0, 1.2, 0.6, 0.9, 1.25, 1.3], index=pd.date_range("2020-01-01", periods=6, freq="MS"))
    peak, trough, recovered = max_drawdown_recovery(curve)
    assert (peak, trough, recovered) == (curve.index[1], curve.index[2], curve.index[4])
    assert worst_rolling_return(curve, window=2) == pytest.approx(0.6 / 1.0 - 1)


# --- Backtest ------------------------------------------------------------------


def _synthetic(seed: int = 0, start: str = "2019-01-01", n: int = 900) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0004, 0.011, n)
    return {
        W: make_ohlcv(6.0 * np.exp(np.cumsum(base * 0.9 + rng.normal(0, 0.003, n))), start=start),
        SP: make_ohlcv(60.0 * np.exp(np.cumsum(base)), start=start),
        LEV: make_ohlcv(30.0 * np.exp(np.cumsum(2 * base - 0.0002)), start=start),
    }


def _example_config(**backtest):
    config = load_config(EXAMPLE)
    config.market = replace(config.market, calendar="NYSE")
    config.backtest = replace(config.backtest, start_date="2019-01-01", **backtest)
    return config


def test_example_config_backtest_whole_shares_few_orders():
    result = run_backtest(_example_config(), _synthetic())
    assert all(q == int(q) for q in result.final_positions.values())
    years = len(result.equity_curve) / 252
    assert 3 <= result.num_orders <= 3 + 4 * years  # entrée + quelques rééquilibrages
    assert result.total_contributed == pytest.approx(2306.59)
    assert result.money_weighted_return_pct == pytest.approx(result.metrics.annualized_return_pct, abs=1.0)
    assert (result.nav_curve * 2306.59 - result.equity_curve).abs().max() < 1e-6


def test_monthly_contribution_accounting():
    data = _synthetic()
    no_contrib = run_backtest(_example_config(), data)
    with_contrib = run_backtest(_example_config(monthly_contribution=100.0), data)
    months = len({(d.year, d.month) for d in with_contrib.equity_curve.index}) - 1
    assert with_contrib.total_contributed == pytest.approx(2306.59 + 100 * months)
    # La NAV par part ne « gagne » rien grâce aux apports (écart = frais/arrondis).
    assert with_contrib.metrics.annualized_return_pct == pytest.approx(no_contrib.metrics.annualized_return_pct, abs=1.5)
    assert with_contrib.equity_curve.iloc[-1] > no_contrib.equity_curve.iloc[-1] + 50 * months
    assert with_contrib.num_orders > no_contrib.num_orders


def test_single_sleeve_buy_and_hold_nav_tracks_price():
    config = _example_config(monthly_contribution=100.0)
    config.symbols = [SP]
    config.strategies[0].params = {"targets": {SP: 1.0}}
    data = _synthetic()
    result = run_backtest(config, {SP: data[SP]})
    nav = result.nav_curve
    price = data[SP]["close"].loc[nav.index]
    assert (nav.iloc[-1] / nav.iloc[1]) == pytest.approx(price.iloc[-1] / price.iloc[1], rel=0.03)


def test_monthly_contribution_rejected_outside_core_satellite():
    config = make_buyhold_config()
    config.backtest = replace(config.backtest, monthly_contribution=100.0)
    with pytest.raises(ValueError, match="core_satellite"):
        run_backtest(config, {"ETF": make_ohlcv(np.linspace(59.0, 80.0, 100))})


# --- Live (broker manuel, réseau simulé) ---------------------------------------


class _Prices:
    current: dict[str, float] = {}

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def history(self, period: str = "5d"):
        return pd.DataFrame({"Close": [self.current[self.symbol]]})


@pytest.fixture
def live(monkeypatch, tmp_path):
    monkeypatch.setattr("yfinance.Ticker", _Prices)
    account = tmp_path / "account.json"
    account.write_text(json.dumps({"cash": 2306.59, "positions": {}}))
    config = load_config(EXAMPLE)
    config.live = replace(
        config.live,
        state_file=str(tmp_path / "state.json"),
        track_record_file=str(tmp_path / "track.json"),
        manual=ManualBrokerConfig(account_file=str(account)),
    )
    bars = {s: make_ohlcv(np.full(260, p)) for s, p in PRICES.items()}
    holder = {"bars": bars}
    monkeypatch.setattr(engine_module, "_fetch_live_bars", lambda config, symbols: holder["bars"])

    def set_prices(prices: dict[str, float]):
        new = {}
        for s, df in holder["bars"].items():
            dt = df.index[-1] + pd.offsets.BDay(1)
            p = prices[s]
            row = pd.DataFrame({"open": [p], "high": [p], "low": [p], "close": [p], "volume": [1]}, index=[dt])
            new[s] = pd.concat([df, row])
        holder["bars"] = new
        _Prices.current = dict(prices)

    _Prices.current = dict(PRICES)
    return config, account, set_prices


class _Notifier:
    def __init__(self):
        self.sent = []

    def send(self, title, message, priority=None, force=False):
        self.sent.append((title, message))
        return True


def _cycle(config, broker, state, notifier):
    state = engine_module.run_once(config, broker, dry_run=False, state=state)
    engine_module.notify_cycle_instructions(notifier, broker)
    return state


def test_live_first_cycle_one_push_then_silent(live):
    config, account, set_prices = live
    broker = engine_module.build_broker(config)
    notifier = _Notifier()
    state = _cycle(config, broker, LiveState(), notifier)
    assert len(notifier.sent) == 1
    title, message = notifier.sent[0]
    assert title == "3 ordre(s) à passer"
    lines = message.split("\n")
    assert lines[0].startswith("INFO : Plan cœur-satellite : apport — 2 283,52 € de cash à investir")
    assert lines[1].startswith("INFO : Poids : DCAM 0,0 % → 54,") and "(cible 55,0 %)" in lines[1]
    assert [line.split(" — ")[0] for line in lines[2:5]] == [
        "ORDRE : ACHETER 200 DCAM", "ORDRE : ACHETER 8 PSP5", "ORDRE : ACHETER 17 CL2"
    ]
    assert lines[5].startswith("INFO : Pas de stop à poser (DCAM, PSP5, CL2) : risk.stop_mode vaut none")
    data = json.loads(account.read_text())
    assert {s: p["qty"] for s, p in data["positions"].items()} == {W: 200, SP: 8, LEV: 17}
    assert data["cash"] > 0
    assert state.core_satellite["units"] == pytest.approx(2306.59)

    # Cycle suivant, rien n'a bougé : aucun push.
    state = _cycle(config, broker, state, notifier)
    assert len(notifier.sent) == 1


def test_live_detects_contribution_and_buys_underweight(live):
    config, account, set_prices = live
    broker = engine_module.build_broker(config)
    notifier = _Notifier()
    state = _cycle(config, broker, LiveState(), notifier)
    units = state.core_satellite["units"]

    data = json.loads(account.read_text())
    data["cash"] += 1000.0  # virement déclaré à la main
    account.write_text(json.dumps(data))
    state = _cycle(config, broker, state, notifier)
    assert len(notifier.sent) == 2
    message = notifier.sent[1][1]
    assert message.startswith("INFO : Apport détecté : +1 000,00 €. Plan cœur-satellite : apport")
    assert "ORDRE : VENDRE" not in message and "ORDRE : ACHETER" in message
    assert "Pas de stop" not in message  # expliqué une seule fois
    # L'apport achète des parts à la NAV courante : pas de faux rendement.
    assert state.core_satellite["units"] > units
    assert state.core_satellite["nav"] == pytest.approx(1.0, abs=0.01)


def test_live_drawdown_and_leverage_alerts_are_deduplicated(live):
    config, account, set_prices = live
    broker = engine_module.build_broker(config)
    notifier = _Notifier()
    state = _cycle(config, broker, LiveState(), notifier)

    # Krach homogène -30 % : pas de dérive, mais palier -20 % franchi.
    set_prices({s: p * 0.7 for s, p in PRICES.items()})
    state = _cycle(config, broker, state, notifier)
    title, message = notifier.sent[-1]
    assert title == "Alerte (à vérifier)"
    assert message.count("ALERTE") == 2  # portefeuille -30 % et CL2 -30 %
    assert "Ne vends pas, c'est prévu dans le plan" in message
    assert "Plan : DCAM 55,0 % · PSP5 20,0 % · CL2 25,0 %" in message
    assert "CL2 à -30,0 % de son plus haut" in message

    # Même niveau le lendemain : rien de nouveau.
    set_prices({s: p * 0.7 for s, p in PRICES.items()})
    before = len(notifier.sent)
    state = _cycle(config, broker, state, notifier)
    assert len(notifier.sent) == before

    # Aggravation jusqu'à -40 % : un seul nouveau palier (-35 %).
    set_prices({s: p * 0.6 for s, p in PRICES.items()})
    state = _cycle(config, broker, state, notifier)
    assert len(notifier.sent) == before + 1
    assert "palier -35,0 % franchi" in notifier.sent[-1][1]
    assert state.alert_flags["plan_drawdown:0.2"] and state.alert_flags["plan_drawdown:0.35"]


def test_live_small_account_does_not_sell_below_min_order_value(live):
    """2 306 € : après -45 % sur CL2, l'excès des autres poches (< 150 €) ne
    vaut pas ses frais — aucune vente, on attend l'apport suivant."""
    config, account, set_prices = live
    broker = engine_module.build_broker(config)
    notifier = _Notifier()
    state = _cycle(config, broker, LiveState(), notifier)
    set_prices({W: PRICES[W], SP: PRICES[SP], LEV: PRICES[LEV] * 0.55})
    state = _cycle(config, broker, state, notifier)
    assert "ORDRE" not in notifier.sent[-1][1] and "CL2 à -45,0 %" in notifier.sent[-1][1]


def test_live_leverage_crash_triggers_rebalance_with_context(live):
    config, account, set_prices = live
    account.write_text(json.dumps({"cash": 20000.0, "positions": {}}))
    broker = engine_module.build_broker(config)
    notifier = _Notifier()
    state = _cycle(config, broker, LiveState(), notifier)
    set_prices({W: PRICES[W], SP: PRICES[SP], LEV: PRICES[LEV] * 0.55})
    state = _cycle(config, broker, state, notifier)
    title, message = notifier.sent[-1]
    assert "CL2 dérive de -" in message and "avec vente(s)" in message
    lines = message.split("\n")
    orders = [line for line in lines if line.startswith("ORDRE")]
    assert orders[0].startswith("ORDRE : VENDRE") and orders[-1].startswith("ORDRE : ACHETER")
    assert any("ACHETER" in o and "CL2" in o for o in orders)
    assert title == f"{len(orders)} ordre(s) à passer"


def test_live_orphan_positions_are_ignored(live):
    config, account, set_prices = live
    _Prices.current["MC.PA"] = 700.0
    account.write_text(json.dumps({"cash": 2306.59, "positions": {"MC.PA": {"qty": 2, "avg_entry_price": 650.0}}}))
    broker = engine_module.build_broker(config)
    engine_module.run_once(config, broker, dry_run=False, state=LiveState())
    texts = [i.text for i in broker.drain_instructions()]
    assert not any("MC" in t and "VENDRE" in t for t in texts)
    assert json.loads(account.read_text())["positions"]["MC.PA"]["qty"] == 2


AXA = "CS.PA"


def _run_scenario(live, axa_prices):
    """Même suite de cycles du soir, avec ou sans 24 AXA hors plan dans le
    fichier de compte. `axa_prices` : cours d'AXA à chaque cycle (None = pas
    d'AXA dans le compte ; valeur None = cours indisponible ce cycle)."""
    config, account, set_prices = live
    positions = {} if axa_prices is None else {AXA: {"qty": 24, "avg_entry_price": 20.242}}
    account.write_text(json.dumps({"cash": 2306.59, "positions": positions}))
    broker = engine_module.build_broker(config)
    notifier = _Notifier()
    state = LiveState()
    steps = [
        (dict(PRICES), None),  # 1er cycle : apport initial investi
        (dict(PRICES), 1000.0),  # virement de 1 000 €
        ({s: p * 0.7 for s, p in PRICES.items()}, None),  # krach -30 % : palier de drawdown
        ({s: p * 0.7 for s, p in PRICES.items()}, None),  # même niveau : rien de nouveau
        ({s: p * 0.6 for s, p in PRICES.items()}, None),  # -40 % : palier -35 %
    ]
    for i, (prices, contribution) in enumerate(steps):
        set_prices(prices)
        if axa_prices is not None and axa_prices[i] is not None:
            _Prices.current[AXA] = axa_prices[i]
        if contribution:
            data = json.loads(account.read_text())
            data["cash"] += contribution
            account.write_text(json.dumps(data))
        state = _cycle(config, broker, state, notifier)
    data = json.loads(account.read_text())
    return {
        "pushes": notifier.sent,
        "cash": data["cash"],
        "sleeves": {s: p for s, p in data["positions"].items() if s != AXA},
        "axa": data["positions"].get(AXA),
        "core_satellite": state.core_satellite,
        "alert_flags": state.alert_flags,
    }


def test_live_off_plan_axa_does_not_change_plan_decisions(live, caplog):
    """24 AXA hors plan, dont le cours fait n'importe quoi (x2, -60 %, cours
    indisponible) : ordres, pushes, détection d'apport, NAV par part et
    alertes de drawdown strictement identiques au même compte sans AXA ;
    AXA jamais vendue ni rachetée ; pas d'avertissement quotidien."""
    reference = _run_scenario(live, None)
    assert reference["alert_flags"]["plan_drawdown:0.2"] and reference["alert_flags"]["plan_drawdown:0.35"]
    for axa_prices in ([43.77, 87.54, 17.5, 43.77, 43.77], [43.77, None, None, 60.0, None], [None] * 5):
        caplog.clear()
        with caplog.at_level("INFO", logger="trading_bot"):
            result = _run_scenario(live, axa_prices)
        assert result["axa"] == {"qty": 24, "avg_entry_price": 20.242}
        assert {k: v for k, v in result.items() if k != "axa"} == {k: v for k, v in reference.items() if k != "axa"}
        assert not any("CS" in text and ("VENDRE" in text or "ACHETER" in text) for _, m in result["pushes"] for text in [m])
        assert not [r for r in caplog.records if r.levelname == "WARNING" and "CS.PA" in r.getMessage()]
        assert sum("hors des poches du plan" in r.getMessage() for r in caplog.records) == 5  # une ligne d'info par cycle


def test_live_off_plan_axa_crash_alone_triggers_no_plan_alert(live):
    config, account, set_prices = live
    account.write_text(json.dumps({"cash": 2306.59, "positions": {AXA: {"qty": 24, "avg_entry_price": 20.242}}}))
    _Prices.current[AXA] = 43.77
    broker = engine_module.build_broker(config)
    notifier = _Notifier()
    state = _cycle(config, broker, LiveState(), notifier)
    pushes = len(notifier.sent)
    set_prices(dict(PRICES))
    _Prices.current[AXA] = 10.0  # AXA -77 %, plan inchangé
    state = _cycle(config, broker, state, notifier)
    assert len(notifier.sent) == pushes  # ni alerte, ni faux retrait, ni ordre
    assert not any(state.alert_flags.get(f"plan_drawdown:{lv:g}") for lv in config.live.alerts.drawdown_levels)
    assert state.core_satellite["nav"] == pytest.approx(1.0, abs=0.01)


def test_live_rejects_stops_in_core_satellite(live):
    config, account, set_prices = live
    config.risk = replace(config.risk, stop_mode="atr")
    with pytest.raises(ValueError, match="stop_mode"):
        engine_module.run_once(config, engine_module.build_broker(config), dry_run=False, state=LiveState())


def test_live_dry_run_does_not_touch_account_or_units(live):
    config, account, set_prices = live
    before = account.read_text()
    broker = engine_module.build_broker(config)
    state = engine_module.run_once(config, broker, dry_run=True, state=LiveState())
    assert account.read_text() == before
    assert state.core_satellite == {}


# --- Rétrocompatibilité --------------------------------------------------------


def test_all_shipped_configs_still_load():
    for path in sorted((REPO_ROOT / "config").glob("*.yaml")):
        config = load_config(path)
        assert config.backtest.monthly_contribution >= 0
        assert isinstance(config.live.alerts, AlertsConfig)
    old = load_config(REPO_ROOT / "config" / "config_pea_buyhold.example.yaml")
    assert old.live.alerts.drawdown_levels == [] and old.live.alerts.symbol_drop_levels == {}
    assert core_satellite_params(old) is None


def test_old_state_file_without_core_satellite_field_loads():
    state = LiveState.from_dict({"trailing_stops": {}, "alert_flags": {"drawdown": True}})
    assert state.core_satellite == {}
    state.core_satellite = {"units": 10.0, "last_qty": {W: 3.0}}
    assert LiveState.from_dict(state.to_dict()).core_satellite == state.core_satellite


# --- Garanties de l'allocation (revue) ------------------------------------------


def _max_drift(qty, cash):
    values = {s: qty.get(s, 0.0) * PRICES[s] for s in TARGETS}
    total = cash + sum(values.values())
    return max(abs(values[s] / total - t) * 100 for s, t in TARGETS.items())


def _after(qty, cash, plan):
    qty = dict(qty)
    for o in plan.orders:
        qty[o.symbol] = qty.get(o.symbol, 0.0) + (o.qty if o.side == "buy" else -o.qty)
    return qty, plan.cash_after


def test_small_account_plus_300_eur_buys_only_underweight_within_gap():
    """Revue : 2 306,59 € investis, puis +300 € d'apport. Avant correctif :
    ACHETER 50 DCAM (48 -> 60 %, cible 55 %), dérive bloquée au-dessus du seuil."""
    first = plan_rebalance({}, PRICES, 2306.59, PARAMS, RULES)
    qty, cash = _after({}, 2306.59, first)
    cash += 300.0
    plan = plan_rebalance(qty, PRICES, cash, PARAMS, RULES)
    assert plan.reason == "contribution" and not plan.sells
    new_qty, new_cash = _after(qty, cash, plan)
    assert _max_drift(new_qty, new_cash) <= min(_max_drift(qty, cash), PARAMS.drift_threshold_pts)
    for o in plan.orders:
        assert plan.weights_after[o.symbol] <= TARGETS[o.symbol] + PRICES[o.symbol] / (cash + 2306.59) + 1e-9
    assert plan.weights_after[W] < 0.56
    # Cycle suivant : dans la bande, rien à faire.
    assert plan_rebalance(new_qty, PRICES, new_cash, PARAMS, RULES).orders == []


def test_overweight_sleeve_is_never_bought():
    qty = _qty_at_target(20000)
    qty[W] = float(int(qty[W] * 1.15))  # DCAM au-dessus de sa cible, même cash compris
    equity = 1000.0 + sum(qty[s] * PRICES[s] for s in TARGETS)
    assert qty[W] * PRICES[W] > 0.55 * equity
    plan = plan_rebalance(qty, PRICES, 1000.0, PARAMS, RULES)
    assert plan.orders and all(o.symbol != W for o in plan.orders if o.side == "buy")


def test_cash_waits_when_no_useful_buy_and_says_why():
    qty = {W: 200.0, SP: 8.0, LEV: 17.0}
    # Toutes les poches sont à moins de 150 € de leur cible : rien de valable.
    plan = plan_rebalance(qty, PRICES, 240.0, PARAMS, RULES)
    assert plan.orders == [] and plan.reason == "waiting" and plan.waiting_cash > 200


def test_calendar_cycle_with_small_buys_is_labelled_calendar():
    qty = _qty_at_target(20000)
    qty[LEV] = float(int(qty[LEV] * 0.9))  # ~2,5 pts de retard, sous le seuil
    plan = plan_rebalance(qty, PRICES, 120.0, PARAMS, RULES, calendar_due=True)
    assert plan.orders and plan.reason == "calendar"


def _brute_force_best_buy_only(qty, cash, params):
    """Toutes les combinaisons d'achats en actions entières respectant les
    mêmes règles (écart + 1 action, garde-fous, cash) : dérive max minimale."""
    values = {s: qty.get(s, 0.0) * PRICES[s] for s in TARGETS}
    deployable = max(0.0, cash - params.cash_buffer(cash))
    investable = sum(values.values()) + deployable
    ranges = []
    for s, t in TARGETS.items():
        gap = t * investable - values[s]
        options = [0]
        if gap > 0:
            for n in range(1, int(gap // PRICES[s]) + 2):
                notional = n * PRICES[s]
                fee = RULES.commission.fee(notional)
                if notional >= RULES.min_order_value and fee <= RULES.max_fee_pct * notional:
                    options.append(n)
        ranges.append(options)
    best = None
    for combo in itertools.product(*ranges):
        cost = sum(n * PRICES[s] + RULES.commission.fee(n * PRICES[s]) for s, n in zip(TARGETS, combo) if n)
        if cost > deployable + 1e-9:
            continue
        new = {s: qty.get(s, 0.0) + n for s, n in zip(TARGETS, combo)}
        drift = _max_drift(new, cash - cost)
        best = drift if best is None else min(best, drift)
    return best


def test_fuzz_plans_never_worsen_drift_and_respect_gaps():
    rng = np.random.default_rng(12345)
    worsened, overbought = [], []
    for _ in range(3000):
        scale = float(rng.choice([2500.0, 8000.0, 20000.0, 60000.0]))
        weights = rng.dirichlet([1.0, 1.0, 1.0])
        qty = {s: float(int(scale * w / PRICES[s])) for s, w in zip(TARGETS, weights)}
        cash = float(rng.choice([0.0, 20.0, 150.0, 300.0, 1000.0, 0.3 * scale])) * float(rng.uniform(0.5, 1.5))
        calendar = bool(rng.random() < 0.2)
        plan = plan_rebalance(qty, PRICES, cash, PARAMS, RULES, calendar_due=calendar)
        before = _max_drift(qty, cash)
        new_qty, new_cash = _after(qty, cash, plan)
        if plan.orders and _max_drift(new_qty, new_cash) > before + 1e-9:
            worsened.append((qty, cash))
        assert new_cash >= -1e-6
        equity = cash + sum(qty[s] * PRICES[s] for s in TARGETS)
        for o in plan.orders:
            if o.side == "buy" and not plan.sells:
                deployable = cash - PARAMS.cash_buffer(cash)
                gap = TARGETS[o.symbol] * (equity - cash + deployable) - qty[o.symbol] * PRICES[o.symbol]
                if gap <= 0 or o.notional_value > gap + PRICES[o.symbol] + 1e-6:
                    overbought.append((o.symbol, qty, cash))
    assert worsened == [] and overbought == []


def test_fuzz_buy_only_matches_brute_force_within_band():
    """Petits apports (recherche exhaustive des deux côtés) : si une
    combinaison d'achats seuls ramène dans la bande, le plan y arrive aussi."""
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(300):
        scale = float(rng.choice([2300.0, 4000.0]))
        weights = np.clip(np.array(list(TARGETS.values())) + rng.normal(0, 0.04, 3), 0.01, None)
        weights /= weights.sum()
        qty = {s: float(int(scale * w / PRICES[s])) for s, w in zip(TARGETS, weights)}
        cash = float(rng.uniform(200.0, 700.0))
        plan = plan_rebalance(qty, PRICES, cash, PARAMS, RULES)
        if plan.sells:
            continue
        best = _brute_force_best_buy_only(qty, cash, PARAMS)
        new_qty, new_cash = _after(qty, cash, plan)
        after = _max_drift(new_qty, new_cash)
        if best is not None and best <= PARAMS.drift_threshold_pts:
            assert after <= PARAMS.drift_threshold_pts + 1e-9, (qty, cash, best, after)
        checked += 1
    assert checked > 150


def test_dry_run_does_not_consume_alerts_or_no_stop_note(live):
    config, account, set_prices = live
    broker = engine_module.build_broker(config)
    state = engine_module.run_once(config, broker, dry_run=True, state=LiveState())
    assert any("Pas de stop" in i.text for i in broker.drain_instructions())
    assert state.alert_flags == {}
    # Le vrai cycle suivant explique encore l'absence de stop.
    state = engine_module.run_once(config, broker, dry_run=False, state=state)
    assert any("Pas de stop" in i.text for i in broker.drain_instructions())

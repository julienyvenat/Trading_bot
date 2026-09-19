from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trading_bot.config import (
    AppConfig,
    BacktestConfig,
    LiveConfig,
    MarketConfig,
    RiskConfig,
    StrategyConfig,
)
from trading_bot.execution.broker_base import AccountInfo, Broker, Position
from trading_bot.live import engine as engine_module
from trading_bot.state import LiveState


class FakeBroker(Broker):
    """Faux broker en mémoire : simule les fills des ordres marché et le
    carnet des ordres stop natifs, pour pouvoir tester leur cycle de vie
    (soumission / annulation / remplacement) sans dépendre d'Alpaca."""

    def __init__(self, equity: float, positions: dict[str, Position], prices: dict[str, float]) -> None:
        self.equity = equity
        self.positions = dict(positions)
        self.prices = prices
        self.price_lookups: list[str] = []
        self.submitted_orders: list[tuple[str, float, str]] = []
        self.stop_orders: dict[str, dict] = {}  # order_id -> {symbol, qty, side, stop_price}
        self.cancelled_order_ids: list[str] = []
        self._next_order_id = 1

    def get_account(self) -> AccountInfo:
        return AccountInfo(equity=self.equity, cash=self.equity, buying_power=self.equity)

    def get_positions(self) -> dict[str, Position]:
        return dict(self.positions)

    def get_last_price(self, symbol: str) -> float:
        self.price_lookups.append(symbol)
        return self.prices.get(symbol, 0.0)

    def submit_market_order(self, symbol: str, qty: float, side: str) -> None:
        self.submitted_orders.append((symbol, qty, side))
        delta = qty if side == "buy" else -qty
        current = self.positions.get(symbol)
        new_qty = (current.qty if current else 0.0) + delta
        price = self.prices.get(symbol, 0.0)
        if new_qty == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = Position(symbol=symbol, qty=new_qty, market_value=new_qty * price, avg_entry_price=price)

    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float) -> str:
        order_id = f"stop-{self._next_order_id}"
        self._next_order_id += 1
        self.stop_orders[order_id] = {"symbol": symbol, "qty": qty, "side": side, "stop_price": stop_price}
        return order_id

    def cancel_order(self, order_id: str) -> None:
        self.cancelled_order_ids.append(order_id)
        self.stop_orders.pop(order_id, None)

    def is_market_open(self) -> bool:
        return True


def make_config() -> AppConfig:
    risk = RiskConfig(
        allow_short=False,
        max_gross_exposure_pct=0.9,
        max_position_weight_pct=0.5,
        risk_per_trade_pct=0.02,
        atr_stop_multiple=2.5,
        atr_window=14,
        max_open_positions=5,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.20,
    )
    return AppConfig(
        symbols=["UP"],
        timeframe="1Day",
        strategies=[
            StrategyConfig(name="sma_crossover", enabled=True, weight=1.0, params={"fast_window": 10, "slow_window": 30}),
        ],
        risk=risk,
        market=MarketConfig(calendar="NYSE", close_buffer_minutes=15),
        backtest=BacktestConfig(start_date="2020-01-01", end_date=None, initial_cash=100_000, commission_pct=0.0005),
        live=LiveConfig(loop_interval_seconds=3600, trade_only_when_market_open=True, state_file="state/live_state.json"),
    )


@pytest.fixture
def uptrend_bars() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(loc=0.5, scale=0.5, size=250))
    index = pd.date_range(start="2020-01-01", periods=len(prices), freq="B")
    close = pd.Series(prices, index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_run_once_prices_and_flattens_positions_outside_universe(monkeypatch, uptrend_bars):
    """Une position ouverte sur un symbole retiré de config.yaml (ici ORPHAN,
    absent de config.symbols) doit quand même être valorisée et liquidée,
    plutôt que d'être ignorée faute de prix (régression du bug rencontré en
    dry-run : positions MDB/RIVN/PLTR/SHOP restées orphelines)."""
    config = make_config()

    monkeypatch.setattr(engine_module, "load_alpaca_credentials", lambda: object())
    monkeypatch.setattr(engine_module, "fetch_latest_bars", lambda symbols, timeframe, credentials: {"UP": uptrend_bars})

    broker = FakeBroker(
        equity=100_000.0,
        positions={
            "ORPHAN": Position(symbol="ORPHAN", qty=10.0, market_value=500.0, avg_entry_price=45.0),
        },
        prices={"UP": float(uptrend_bars["close"].iloc[-1]), "ORPHAN": 50.0},
    )

    engine_module.run_once(config, broker, dry_run=False, state=LiveState())

    # Le prix d'ORPHAN doit avoir été demandé même s'il n'est plus dans l'univers configuré...
    assert "ORPHAN" in broker.price_lookups
    # ...et la position doit être liquidée (vendue en totalité) plutôt que laissée orpheline.
    assert ("ORPHAN", 10.0, "sell") in broker.submitted_orders


def test_run_once_submits_native_stop_order_for_new_position(monkeypatch, uptrend_bars):
    """Ouvrir une position doit poser un vrai ordre stop natif chez le broker
    (pas seulement suivre le stop en Python), avec son id conservé dans l'état."""
    config = make_config()
    monkeypatch.setattr(engine_module, "load_alpaca_credentials", lambda: object())
    monkeypatch.setattr(engine_module, "fetch_latest_bars", lambda symbols, timeframe, credentials: {"UP": uptrend_bars})

    last_price = float(uptrend_bars["close"].iloc[-1])
    broker = FakeBroker(equity=100_000.0, positions={}, prices={"UP": last_price})

    state = engine_module.run_once(config, broker, dry_run=False, state=LiveState())

    assert "UP" in state.stop_order_ids
    order_id = state.stop_order_ids["UP"]
    assert order_id in broker.stop_orders
    assert broker.stop_orders[order_id]["side"] == "sell"  # position longue -> stop de vente
    assert broker.stop_orders[order_id]["stop_price"] < last_price


def test_run_once_cancels_native_stop_before_flattening_on_drawdown(monkeypatch, uptrend_bars):
    """Le coupe-circuit de drawdown doit annuler le stop natif existant avant
    de flatten la position (sinon le stop resterait posé sur une position
    qui n'existe plus côté broker)."""
    config = make_config()
    monkeypatch.setattr(engine_module, "load_alpaca_credentials", lambda: object())
    monkeypatch.setattr(engine_module, "fetch_latest_bars", lambda symbols, timeframe, credentials: {"UP": uptrend_bars})

    last_price = float(uptrend_bars["close"].iloc[-1])
    broker = FakeBroker(
        equity=70_000.0,  # drawdown > 20% par rapport au pic mémorisé -> coupe-circuit
        positions={"UP": Position(symbol="UP", qty=100.0, market_value=100.0 * last_price, avg_entry_price=last_price)},
        prices={"UP": last_price},
    )
    broker.stop_orders["stop-existing"] = {"symbol": "UP", "qty": 100.0, "side": "sell", "stop_price": last_price * 0.9}
    state = LiveState(stop_order_ids={"UP": "stop-existing"})
    state.risk_state = engine_module.RiskState.initial(100_000.0, today=pd.Timestamp.now(tz="UTC").date())

    engine_module.run_once(config, broker, dry_run=False, state=state)

    assert "stop-existing" in broker.cancelled_order_ids
    assert ("UP", 100.0, "sell") in broker.submitted_orders


def test_run_once_records_pre_existing_realized_trade_but_keeps_stale_snapshot_on_drawdown_halt(
    monkeypatch, tmp_path, uptrend_bars
):
    """Le coupe-circuit de drawdown doit quand même enregistrer un trade
    réalisé AVANT ce cycle (ex: stop déclenché juste avant le
    déclenchement du coupe-circuit), mais ne doit PAS mettre à jour
    `last_known_positions` : le flatten qu'il vient de déclencher sera
    détecté comme réalisé au PROCHAIN cycle, par comparaison avec cet
    instantané resté volontairement inchangé (voir le commentaire dans
    trading_bot.live.engine.run_once)."""
    config = make_config()
    config.live.track_record_file = str(tmp_path / "track_record.json")
    monkeypatch.setattr(engine_module, "load_alpaca_credentials", lambda: object())
    monkeypatch.setattr(engine_module, "fetch_latest_bars", lambda symbols, timeframe, credentials: {"UP": uptrend_bars})

    last_price = float(uptrend_bars["close"].iloc[-1])
    broker = FakeBroker(equity=70_000.0, positions={}, prices={"UP": last_price})  # UP déjà fermée avant ce cycle

    from trading_bot.execution.broker_base import PositionSnapshot

    stale_snapshot = PositionSnapshot(qty=50.0, avg_entry_price=80.0)
    state = LiveState(last_known_positions={"UP": stale_snapshot})
    state.risk_state = engine_module.RiskState.initial(100_000.0, today=pd.Timestamp.now(tz="UTC").date())

    new_state = engine_module.run_once(config, broker, dry_run=False, state=state)

    track_record = json.loads((tmp_path / "track_record.json").read_text())
    assert track_record["UP"]["trade_count"] == 1
    assert new_state.last_known_positions == {"UP": stale_snapshot}


def make_two_symbol_config() -> AppConfig:
    config = make_config()
    config.symbols = ["UP", "UP2"]
    return config


def test_run_once_survives_stop_order_rejection_on_one_symbol(monkeypatch, uptrend_bars):
    """Régression : Alpaca a déjà rejeté un ordre stop juste après le fill du
    rebalancement ("potential wash trade detected"), ce qui faisait planter
    tout le cycle avant de poser les stops des AUTRES positions et avant la
    moindre sauvegarde d'état. Le rejet d'un symbole ne doit affecter que ce
    symbole."""
    config = make_two_symbol_config()
    monkeypatch.setattr(engine_module, "load_alpaca_credentials", lambda: object())
    monkeypatch.setattr(
        engine_module,
        "fetch_latest_bars",
        lambda symbols, timeframe, credentials: {"UP": uptrend_bars, "UP2": uptrend_bars},
    )
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)  # pas d'attente réelle en test

    last_price = float(uptrend_bars["close"].iloc[-1])
    broker = FakeBroker(equity=100_000.0, positions={}, prices={"UP": last_price, "UP2": last_price})

    real_submit = broker.submit_stop_order

    def failing_submit(symbol, qty, side, stop_price):
        if symbol == "UP":
            raise RuntimeError("potential wash trade detected")
        return real_submit(symbol, qty, side, stop_price)

    broker.submit_stop_order = failing_submit

    state = engine_module.run_once(config, broker, dry_run=False, state=LiveState())

    # UP2 a bien reçu son stop natif malgré l'échec sur UP.
    assert "UP2" in state.stop_order_ids
    assert state.stop_order_ids["UP2"] in broker.stop_orders
    # UP a échoué après plusieurs tentatives mais n'a pas fait planter le cycle.
    assert "UP" not in state.stop_order_ids


def test_run_once_survives_rebalance_order_rejection_on_one_symbol(monkeypatch, uptrend_bars):
    """Régression : un ordre de rebalancement rejeté par Alpaca ("potential
    wash trade detected", voir `execution.rebalancer._submit_market_order_with_retry`)
    faisait planter tout `run_once` avant même d'atteindre la pose des stops
    ou la sauvegarde d'état — aucun AUTRE symbole n'était alors rebalancé ni
    protégé ce cycle-là. Le rejet d'un symbole ne doit affecter que lui."""
    config = make_two_symbol_config()
    monkeypatch.setattr(engine_module, "load_alpaca_credentials", lambda: object())
    monkeypatch.setattr(
        engine_module,
        "fetch_latest_bars",
        lambda symbols, timeframe, credentials: {"UP": uptrend_bars, "UP2": uptrend_bars},
    )
    monkeypatch.setattr(engine_module.time, "sleep", lambda seconds: None)  # pas d'attente réelle en test

    last_price = float(uptrend_bars["close"].iloc[-1])
    broker = FakeBroker(equity=100_000.0, positions={}, prices={"UP": last_price, "UP2": last_price})

    real_submit = broker.submit_market_order

    def failing_submit(symbol, qty, side):
        if symbol == "UP":
            raise RuntimeError("potential wash trade detected")
        return real_submit(symbol, qty, side)

    broker.submit_market_order = failing_submit

    state = engine_module.run_once(config, broker, dry_run=False, state=LiveState())

    # UP2 a bien été rebalancé et protégé malgré l'échec sur UP.
    assert "UP2" in state.stop_order_ids
    assert "UP2" in broker.positions
    # UP a échoué après plusieurs tentatives mais n'a pas fait planter le cycle
    # (donc pas de position ouverte ni de stop pour lui).
    assert "UP" not in broker.positions
    assert "UP" not in state.stop_order_ids


def test_run_once_renews_stop_order_on_new_trading_day_even_if_price_unchanged(monkeypatch, uptrend_bars):
    """Les stops natifs sont posés en TimeInForce.DAY (obligatoire côté
    Alpaca pour une quantité fractionnaire) : ils expirent donc à la clôture
    et doivent être reposés à la séance suivante même si le prix du stop n'a
    pas bougé, sinon la position se retrouve silencieusement sans protection."""
    config = make_config()
    monkeypatch.setattr(engine_module, "load_alpaca_credentials", lambda: object())
    monkeypatch.setattr(engine_module, "fetch_latest_bars", lambda symbols, timeframe, credentials: {"UP": uptrend_bars})

    last_price = float(uptrend_bars["close"].iloc[-1])
    qty = 100.0
    broker = FakeBroker(
        equity=100_000.0,
        positions={"UP": Position(symbol="UP", qty=qty, market_value=qty * last_price, avg_entry_price=last_price)},
        prices={"UP": last_price},
    )
    # Stop déjà posé hier, au même prix que celui recalculé aujourd'hui (le
    # stop suiveur ne "ratchet" pas puisque le prix n'a pas bougé) : sans la
    # logique d'expiration, `stop_moved` serait False et rien ne serait reposé.
    from trading_bot.portfolio.stops import StopLevel

    existing_stop = StopLevel(direction=1, stop_price=last_price * 0.85)
    broker.stop_orders["stop-yesterday"] = {"symbol": "UP", "qty": qty, "side": "sell", "stop_price": existing_stop.stop_price}
    state = LiveState(
        trailing_stops={"UP": existing_stop},
        stop_order_ids={"UP": "stop-yesterday"},
        stop_order_dates={"UP": "2020-01-01"},  # une séance antérieure
    )

    new_state = engine_module.run_once(config, broker, dry_run=False, state=state)

    # L'ancien ordre (expiré côté broker) a été annulé (best-effort) et un
    # nouveau a été reposé pour la séance en cours, même sans mouvement de prix.
    assert "stop-yesterday" in broker.cancelled_order_ids
    assert new_state.stop_order_ids["UP"] != "stop-yesterday"
    assert new_state.stop_order_dates["UP"] == pd.Timestamp.now(tz="UTC").date().isoformat()


def test_run_once_records_stop_fill_between_cycles_into_track_record(monkeypatch, tmp_path, uptrend_bars):
    """Une position qui a disparu côté broker depuis le dernier cycle connu
    (stop natif déclenché entre deux cycles, ou toute autre sortie externe)
    doit être détectée comme un trade RÉALISÉ et alimenter la base
    persistante de suivi par symbole (voir
    trading_bot.portfolio.symbol_track_record), pas seulement déclencher le
    nettoyage de trailing_stops/stop_order_ids déjà en place."""
    config = make_config()
    config.live.track_record_file = str(tmp_path / "track_record.json")
    monkeypatch.setattr(engine_module, "load_alpaca_credentials", lambda: object())
    monkeypatch.setattr(engine_module, "fetch_latest_bars", lambda symbols, timeframe, credentials: {"UP": uptrend_bars})

    last_price = float(uptrend_bars["close"].iloc[-1])
    broker = FakeBroker(equity=100_000.0, positions={}, prices={"UP": last_price})  # déjà fermée côté broker

    from trading_bot.execution.broker_base import PositionSnapshot

    state = LiveState(last_known_positions={"UP": PositionSnapshot(qty=100.0, avg_entry_price=90.0)})

    engine_module.run_once(config, broker, dry_run=False, state=state)

    track_record = json.loads((tmp_path / "track_record.json").read_text())
    assert track_record["UP"]["trade_count"] == 1
    assert track_record["UP"]["total_pnl_pct"] == pytest.approx((last_price - 90.0) / 90.0)


def test_run_once_records_same_cycle_liquidation_into_track_record(monkeypatch, tmp_path, uptrend_bars):
    """Une position liquidée PAR CE CYCLE (ex: symbole orphelin hors de
    l'univers configuré, voir test_run_once_prices_and_flattens_positions_outside_universe)
    doit elle aussi être enregistrée comme trade réalisé, pas seulement les
    fermetures détectées entre deux cycles."""
    config = make_config()
    config.live.track_record_file = str(tmp_path / "track_record.json")
    monkeypatch.setattr(engine_module, "load_alpaca_credentials", lambda: object())
    monkeypatch.setattr(engine_module, "fetch_latest_bars", lambda symbols, timeframe, credentials: {"UP": uptrend_bars})

    broker = FakeBroker(
        equity=100_000.0,
        positions={"ORPHAN": Position(symbol="ORPHAN", qty=10.0, market_value=500.0, avg_entry_price=45.0)},
        prices={"UP": float(uptrend_bars["close"].iloc[-1]), "ORPHAN": 50.0},
    )

    engine_module.run_once(config, broker, dry_run=False, state=LiveState())

    track_record = json.loads((tmp_path / "track_record.json").read_text())
    assert track_record["ORPHAN"]["trade_count"] == 1
    assert track_record["ORPHAN"]["total_pnl_pct"] == pytest.approx((50.0 - 45.0) / 45.0)


def test_run_once_does_not_touch_track_record_when_nothing_realized(monkeypatch, tmp_path, uptrend_bars):
    """Pas de trade réalisé ce cycle -> pas d'I/O sur la base de suivi (pas
    de fichier créé)."""
    config = make_config()
    config.live.track_record_file = str(tmp_path / "track_record.json")
    monkeypatch.setattr(engine_module, "load_alpaca_credentials", lambda: object())
    monkeypatch.setattr(engine_module, "fetch_latest_bars", lambda symbols, timeframe, credentials: {"UP": uptrend_bars})

    last_price = float(uptrend_bars["close"].iloc[-1])
    broker = FakeBroker(equity=100_000.0, positions={}, prices={"UP": last_price})

    engine_module.run_once(config, broker, dry_run=False, state=LiveState())

    assert not (tmp_path / "track_record.json").exists()


def test_run_once_uses_yfinance_instead_of_alpaca_when_broker_is_manual(monkeypatch, uptrend_bars):
    """`live.broker: "manual"` (PEA sans API, voir `ManualBroker`) doit
    récupérer les bougies via yfinance, jamais via Alpaca (qui ne couvre pas
    les actions européennes) : `load_alpaca_credentials`/`fetch_latest_bars`
    ne doivent pas être appelés dans ce mode."""
    config = make_config()
    config.live.broker = "manual"

    def _fail(*args, **kwargs):
        raise AssertionError("Alpaca ne doit pas être sollicité en mode `live.broker: manual`.")

    monkeypatch.setattr(engine_module, "load_alpaca_credentials", _fail)
    monkeypatch.setattr(engine_module, "fetch_latest_bars", _fail)
    monkeypatch.setattr(
        "trading_bot.data.historical.fetch_latest_data", lambda symbols, interval: {"UP": uptrend_bars}
    )

    broker = FakeBroker(equity=100_000.0, positions={}, prices={"UP": float(uptrend_bars["close"].iloc[-1])})

    engine_module.run_once(config, broker, dry_run=False, state=LiveState())  # ne doit pas lever


def test_build_broker_returns_manual_broker_when_configured(tmp_path):
    from trading_bot.execution.manual_broker import ManualBroker

    account_file = tmp_path / "account.json"
    account_file.write_text('{"cash": 1000.0, "positions": {}}')

    config = make_config()
    config.live.broker = "manual"
    config.live.manual.account_file = str(account_file)

    broker = engine_module.build_broker(config)

    assert isinstance(broker, ManualBroker)


def test_build_broker_rejects_unknown_broker_name():
    config = make_config()
    config.live.broker = "does-not-exist"

    with pytest.raises(ValueError, match="does-not-exist"):
        engine_module.build_broker(config)

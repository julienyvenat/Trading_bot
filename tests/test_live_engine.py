from __future__ import annotations

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

"""Implémentation du broker via l'API Alpaca (paper trading par défaut)."""

from __future__ import annotations

from trading_bot.config import AlpacaCredentials
from trading_bot.execution.broker_base import AccountInfo, Broker, Position


class AlpacaBroker(Broker):
    def __init__(self, credentials: AlpacaCredentials) -> None:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest
        from alpaca.trading.client import TradingClient

        self._credentials = credentials
        self._trading_client = TradingClient(credentials.api_key, credentials.secret_key, paper=credentials.paper)
        self._data_client = StockHistoricalDataClient(credentials.api_key, credentials.secret_key)
        self._StockLatestTradeRequest = StockLatestTradeRequest

    def get_account(self) -> AccountInfo:
        account = self._trading_client.get_account()
        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
        )

    def get_positions(self) -> dict[str, Position]:
        positions = self._trading_client.get_all_positions()
        return {
            p.symbol: Position(
                symbol=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
            )
            for p in positions
        }

    def get_last_price(self, symbol: str) -> float:
        request = self._StockLatestTradeRequest(symbol_or_symbols=symbol)
        trades = self._data_client.get_stock_latest_trade(request)
        return float(trades[symbol].price)

    def submit_market_order(self, symbol: str, qty: float, side: str) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 4),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        self._trading_client.submit_order(request)

    def is_market_open(self) -> bool:
        clock = self._trading_client.get_clock()
        return bool(clock.is_open)

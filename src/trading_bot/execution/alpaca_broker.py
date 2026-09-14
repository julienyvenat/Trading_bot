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

    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        request = StopOrderRequest(
            symbol=symbol,
            qty=round(qty, 4),
            side=order_side,
            # DAY et non GTC : Alpaca rejette catégoriquement tout ordre
            # stop/stop_limit GTC dès que la quantité est fractionnaire
            # ("stop/stop_limit fractional GTC orders are not enabled"),
            # et le dimensionnement basé sur l'ATR (`RiskManager`) produit
            # quasi systématiquement des quantités fractionnaires. Un ordre
            # DAY expire donc à la clôture : c'est `run_once`
            # (`trading_bot.live.engine`) qui se charge de le reposer à
            # chaque nouvelle séance via `state.stop_order_dates`, pas
            # seulement quand le prix du stop a bougé.
            time_in_force=TimeInForce.DAY,
            stop_price=round(stop_price, 2),
        )
        order = self._trading_client.submit_order(request)
        return str(order.id)

    def cancel_order(self, order_id: str) -> None:
        from alpaca.common.exceptions import APIError

        try:
            self._trading_client.cancel_order_by_id(order_id)
        except APIError:
            # Cas normal : l'ordre a déjà été exécuté (le stop a fillé) ou
            # déjà annulé entre deux cycles. Rien à faire de plus ici, c'est
            # à l'appelant de nettoyer son propre état si besoin.
            pass

    def is_market_open(self) -> bool:
        clock = self._trading_client.get_clock()
        return bool(clock.is_open)

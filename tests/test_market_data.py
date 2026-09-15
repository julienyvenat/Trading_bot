from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.config import AlpacaCredentials
from trading_bot.data.market_data import fetch_historical_bars, fetch_latest_bars


class _FakeBarSet:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


def _fake_bar_set_df(symbols_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reproduit la forme MultiIndex (symbol, timestamp) renvoyée par
    `StockHistoricalDataClient.get_stock_bars(...).df`."""
    return pd.concat(symbols_data, names=["symbol", "timestamp"])


def _ohlcv(values: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(values), freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": values, "high": values, "low": values, "close": values, "volume": [v * 10 for v in values]},
        index=index,
    )


_CREDENTIALS = AlpacaCredentials(api_key="k", secret_key="s", paper=True)


def test_fetch_historical_bars_splits_by_symbol_with_naive_timestamps(monkeypatch):
    fake_df = _fake_bar_set_df({"TSLA": _ohlcv([1.0, 2.0, 3.0]), "AMD": _ohlcv([4.0, 5.0, 6.0])})
    captured_requests = []

    class FakeClient:
        def __init__(self, api_key, secret_key):
            pass

        def get_stock_bars(self, request):
            captured_requests.append(request)
            return _FakeBarSet(fake_df)

    monkeypatch.setattr("alpaca.data.historical.StockHistoricalDataClient", FakeClient)

    result = fetch_historical_bars(
        ["TSLA", "AMD"], start_date="2024-01-01", end_date="2024-01-02", timeframe="5Min", credentials=_CREDENTIALS
    )

    assert set(result) == {"TSLA", "AMD"}
    assert list(result["TSLA"]["close"]) == [1.0, 2.0, 3.0]
    assert list(result["AMD"]["close"]) == [4.0, 5.0, 6.0]
    assert result["TSLA"].index.tz is None  # cohérent avec la source yfinance (timestamps naïfs)
    assert list(result["TSLA"].columns) == ["open", "high", "low", "close", "volume"]
    assert len(captured_requests) == 1


def test_fetch_historical_bars_skips_symbol_missing_from_response(monkeypatch):
    """Un symbole sans données renvoyées (ex: pas coté sur la période) est
    simplement absent du résultat, pas une erreur."""
    fake_df = _fake_bar_set_df({"TSLA": _ohlcv([1.0, 2.0])})

    class FakeClient:
        def __init__(self, api_key, secret_key):
            pass

        def get_stock_bars(self, request):
            return _FakeBarSet(fake_df)

    monkeypatch.setattr("alpaca.data.historical.StockHistoricalDataClient", FakeClient)

    result = fetch_historical_bars(
        ["TSLA", "GHOST"], start_date="2024-01-01", end_date="2024-01-02", timeframe="5Min", credentials=_CREDENTIALS
    )

    assert set(result) == {"TSLA"}


def test_fetch_historical_bars_defaults_end_to_now(monkeypatch):
    """Sans `end_date`, la requête doit porter jusqu'à maintenant plutôt que
    de planter ou de laisser une borne de fin vide."""
    fake_df = _fake_bar_set_df({"TSLA": _ohlcv([1.0])})
    captured_requests = []

    class FakeClient:
        def __init__(self, api_key, secret_key):
            pass

        def get_stock_bars(self, request):
            captured_requests.append(request)
            return _FakeBarSet(fake_df)

    monkeypatch.setattr("alpaca.data.historical.StockHistoricalDataClient", FakeClient)

    fetch_historical_bars(["TSLA"], start_date="2024-01-01", end_date=None, timeframe="5Min", credentials=_CREDENTIALS)

    request = captured_requests[0]
    assert request.end is not None
    end_ts = pd.Timestamp(request.end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    now_ts = pd.Timestamp.now(tz="UTC")
    assert (now_ts - end_ts).total_seconds() < 60


def test_fetch_latest_bars_still_works_after_refactor(monkeypatch):
    """Garde-fou : le partage du code d'éclatement par symbole
    (`_bars_by_symbol`) avec `fetch_historical_bars` ne doit rien changer au
    comportement existant de `fetch_latest_bars` (utilisé en live)."""
    fake_df = _fake_bar_set_df({"TSLA": _ohlcv([1.0, 2.0, 3.0])})

    class FakeClient:
        def __init__(self, api_key, secret_key):
            pass

        def get_stock_bars(self, request):
            return _FakeBarSet(fake_df)

    monkeypatch.setattr("alpaca.data.historical.StockHistoricalDataClient", FakeClient)

    result = fetch_latest_bars(["TSLA"], "5Min", _CREDENTIALS, lookback_days=10)

    assert list(result["TSLA"]["close"]) == [1.0, 2.0, 3.0]

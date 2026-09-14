"""Récupération de données de marché récentes via Alpaca (pour le trading live/paper)."""

from __future__ import annotations

import re

import pandas as pd

from trading_bot.config import AlpacaCredentials

# Ex: "1Day", "1Hour", "5Min", "15Min", "30Min", "1Min" — même convention que
# `yfinance_interval_for_timeframe` côté backtest (trading_bot.data.historical),
# pour que `universe.timeframe` dans config.yaml désigne la même granularité
# en live comme en backtest.
_TIMEFRAME_PATTERN = re.compile(r"(\d+)(Min|Hour|Day)")


def _parse_timeframe(timeframe: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    match = _TIMEFRAME_PATTERN.fullmatch(timeframe)
    if not match:
        raise ValueError(
            f"Timeframe '{timeframe}' non reconnu. Format attendu : '<nombre><Min|Hour|Day>' "
            "(ex: '1Day', '1Hour', '5Min')."
        )
    amount, unit_name = match.groups()
    unit = {"Min": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour, "Day": TimeFrameUnit.Day}[unit_name]
    return TimeFrame(int(amount), unit)


def fetch_latest_bars(
    symbols: list[str],
    timeframe: str,
    credentials: AlpacaCredentials,
    lookback_days: int = 400,
) -> dict[str, pd.DataFrame]:
    """Récupère les dernières bougies OHLCV via l'API de données Alpaca.

    `lookback_days` doit être suffisamment grand pour couvrir la plus longue
    fenêtre utilisée par les stratégies (ex: SMA 50 -> largement > 50 jours
    en quotidien, ou > 50 bougies en intraday : `lookback_days` reste exprimé
    en jours calendaires quelle que soit la granularité, à ajuster en
    conséquence pour une stratégie intraday comme `bollinger_scalping`).
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    tf = _parse_timeframe(timeframe)

    client = StockHistoricalDataClient(credentials.api_key, credentials.secret_key)
    start = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)

    request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start.to_pydatetime())
    bar_set = client.get_stock_bars(request)
    df_all = bar_set.df  # MultiIndex (symbol, timestamp)

    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol not in df_all.index.get_level_values(0):
            continue
        df = df_all.xs(symbol, level=0).copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.rename(columns={"trade_count": "trade_count"})
        data[symbol] = df[["open", "high", "low", "close", "volume"]].sort_index()

    return data

"""Récupération de données de marché récentes via Alpaca (pour le trading live/paper)."""

from __future__ import annotations

import pandas as pd

from trading_bot.config import AlpacaCredentials


def fetch_latest_bars(
    symbols: list[str],
    timeframe: str,
    credentials: AlpacaCredentials,
    lookback_days: int = 400,
) -> dict[str, pd.DataFrame]:
    """Récupère les dernières bougies OHLCV via l'API de données Alpaca.

    `lookback_days` doit être suffisamment grand pour couvrir la plus longue
    fenêtre utilisée par les stratégies (ex: SMA 50 -> largement > 50 jours).
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    tf_map = {
        "1Day": TimeFrame.Day,
        "1Hour": TimeFrame.Hour,
        "1Min": TimeFrame.Minute,
    }
    tf = tf_map.get(timeframe, TimeFrame.Day)

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

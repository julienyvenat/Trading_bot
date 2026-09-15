"""Récupération de données de marché via l'API de données Alpaca : bougies
récentes pour le live/paper (`fetch_latest_bars`) et historique sur plage de
dates explicite pour le backtest (`fetch_historical_bars`, alternative à
yfinance quand son historique intraday limité ne suffit pas — voir
`trading_bot.data.historical`)."""

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


def _bars_by_symbol(df_all: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Éclate le DataFrame MultiIndex (symbol, timestamp) renvoyé par l'API
    de données Alpaca en un DataFrame OHLCV par symbole, timestamps naïfs
    (comme `trading_bot.data.historical.fetch_historical_data`, pour rester
    interchangeable avec la source yfinance côté backtest)."""
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol not in df_all.index.get_level_values(0):
            continue
        df = df_all.xs(symbol, level=0).copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data[symbol] = df[["open", "high", "low", "close", "volume"]].sort_index()
    return data


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
    conséquence pour une stratégie intraday comme `momentum_breakout`
    reparamétrée en bougies 1h, voir `config/config_intraday.example.yaml`).
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    tf = _parse_timeframe(timeframe)

    client = StockHistoricalDataClient(credentials.api_key, credentials.secret_key)
    start = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)

    request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start.to_pydatetime())
    bar_set = client.get_stock_bars(request)
    return _bars_by_symbol(bar_set.df, symbols)


def fetch_historical_bars(
    symbols: list[str],
    start_date: str,
    end_date: str | None,
    timeframe: str,
    credentials: AlpacaCredentials,
) -> dict[str, pd.DataFrame]:
    """Récupère un historique de bougies OHLCV via l'API de données Alpaca,
    sur une plage de dates EXPLICITE (contrairement à `fetch_latest_bars`,
    pensé pour le live avec une fenêtre glissante de `lookback_days` jusqu'à
    maintenant).

    Alternative à `trading_bot.data.historical.fetch_historical_data`
    (yfinance) pour `backtest`/`optimize`/`walk-forward`, sélectionnable via
    `backtest.data_source: "alpaca"` dans config.yaml : utile quand
    l'historique intraday nécessaire dépasse la limite de yfinance (~60
    jours en 5Min/15Min/30Min, 7 jours en 1Min — voir
    `trading_bot.data.historical`). Le plan de données Alpaca gratuit (IEX)
    conserve plusieurs années d'historique intraday, largement de quoi
    faire un walk-forward avec assez de fenêtres hors échantillon.
    """
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    tf = _parse_timeframe(timeframe)
    client = StockHistoricalDataClient(credentials.api_key, credentials.secret_key)

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.utcnow()

    # `feed=IEX` explicite : un compte gratuit/paper n'est pas habilité pour
    # le flux SIP (défaut implicite de l'API pour une plage de dates qui
    # s'approche de l'instant présent), sans quoi Alpaca renvoie une erreur
    # d'abonnement plutôt que les données IEX auxquelles le compte a droit.
    request = StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=tf, start=start.to_pydatetime(), end=end.to_pydatetime(), feed=DataFeed.IEX
    )
    bar_set = client.get_stock_bars(request)
    return _bars_by_symbol(bar_set.df, symbols)

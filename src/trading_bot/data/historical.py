"""Récupération de données historiques (OHLCV) pour le backtesting via yfinance."""

from __future__ import annotations

import pandas as pd

from trading_bot.logger import get_logger

logger = get_logger()

# Correspondance entre `universe.timeframe` (config.yaml, même convention que
# côté live/Alpaca, voir `trading_bot.data.market_data`) et le code
# d'intervalle attendu par yfinance.
_YFINANCE_INTERVAL_MAP = {
    "1Day": "1d",
    "1Hour": "1h",
    "30Min": "30m",
    "15Min": "15m",
    "5Min": "5m",
    "1Min": "1m",
}

# Profondeur d'historique max connue de yfinance par granularité intraday
# (constatée empiriquement, sujette à changer côté Yahoo Finance sans
# préavis) : au-delà, yfinance renvoie silencieusement moins de données que
# demandé plutôt que d'échouer explicitement — d'où l'avertissement dans
# `fetch_historical_data` plutôt qu'une erreur bloquante.
_YFINANCE_MAX_LOOKBACK_DAYS = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
}


def yfinance_interval_for_timeframe(timeframe: str) -> str:
    """Convertit un `universe.timeframe` de config.yaml (ex: '5Min') en code
    d'intervalle yfinance (ex: '5m')."""
    if timeframe not in _YFINANCE_INTERVAL_MAP:
        raise ValueError(
            f"Timeframe '{timeframe}' non reconnu pour le backtest. Valeurs supportées : "
            f"{', '.join(_YFINANCE_INTERVAL_MAP)}."
        )
    return _YFINANCE_INTERVAL_MAP[timeframe]


def fetch_historical_data(
    symbols: list[str],
    start_date: str,
    end_date: str | None = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Télécharge l'historique OHLCV pour chaque symbole.

    Retourne un dict {symbole: DataFrame} avec les colonnes
    ["open", "high", "low", "close", "volume"], indexées par date (tz-naive).
    """
    import yfinance as yf  # import différé : dépendance réseau, inutile pour les tests unitaires

    max_lookback = _YFINANCE_MAX_LOOKBACK_DAYS.get(interval)
    if max_lookback is not None and start_date:
        requested_days = (pd.Timestamp.now() - pd.Timestamp(start_date)).days
        if requested_days > max_lookback:
            logger.warning(
                "Intervalle '%s' demandé depuis %d jours, mais yfinance ne garantit généralement "
                "que %d jours d'historique à cette granularité : les données reçues risquent d'être "
                "tronquées silencieusement (pas d'erreur côté yfinance).",
                interval,
                requested_days,
                max_lookback,
            )

    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        raw = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        data[symbol] = raw[["open", "high", "low", "close", "volume"]].sort_index()

    return data


def fetch_latest_data(symbols: list[str], interval: str = "1d", lookback_days: int = 400) -> dict[str, pd.DataFrame]:
    """Équivalent yfinance de `trading_bot.data.market_data.fetch_latest_bars`
    (Alpaca) : une fenêtre glissante de `lookback_days` jusqu'à aujourd'hui,
    plutôt qu'une plage de dates explicite comme `fetch_historical_data`.
    Utilisé par le moteur live (`trading_bot.live.engine`) quand
    `live.broker: "manual"` (ex: PEA sans API de courtage, voir
    `trading_bot.execution.manual_broker`) — Alpaca ne couvrant pas les
    actions européennes (Euronext...), le live doit alors s'appuyer sur
    yfinance plutôt que sur `market_data.fetch_latest_bars`."""
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    return fetch_historical_data(symbols, start_date=start_date, end_date=None, interval=interval)

"""Récupération de données historiques (OHLCV) pour le backtesting via yfinance."""

from __future__ import annotations

import pandas as pd


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

"""Indicateurs techniques réutilisables par les stratégies (vectorisés pandas)."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (méthode de Wilder)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range à partir d'un DataFrame avec colonnes high/low/close."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).max()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).min()


def bollinger_bands(series: pd.Series, window: int, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bandes de Bollinger : (bande médiane, bande supérieure, bande inférieure).

    `window` est un nombre de BOUGIES, pas de jours : sur des bougies
    intraday (voir `trading_bot.strategies.bollinger_scalping`), une fenêtre
    de 20 correspond à 20 bougies (ex: 100 minutes en 5 min), pas à 20 jours.
    """
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower

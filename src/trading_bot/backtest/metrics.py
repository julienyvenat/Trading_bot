"""Calcul des métriques de performance d'une courbe d'equity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestMetrics:
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trading_days: int

    def summary(self) -> str:
        return (
            f"Rendement total       : {self.total_return_pct:+.2f}%\n"
            f"Rendement annualisé   : {self.annualized_return_pct:+.2f}%\n"
            f"Volatilité annualisée : {self.annualized_volatility_pct:.2f}%\n"
            f"Ratio de Sharpe       : {self.sharpe_ratio:.2f}\n"
            f"Max drawdown          : {self.max_drawdown_pct:.2f}%\n"
            f"Taux de jours positifs: {self.win_rate_pct:.2f}%\n"
            f"Jours de trading      : {self.num_trading_days}"
        )


def compute_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> BacktestMetrics:
    """Calcule les métriques standards à partir d'une courbe d'equity (valeur
    totale du portefeuille au fil du temps)."""
    equity_curve = equity_curve.dropna()
    if len(equity_curve) < 2:
        return BacktestMetrics(0, 0, 0, 0, 0, 0, len(equity_curve))

    daily_returns = equity_curve.pct_change().dropna()
    num_days = len(daily_returns)

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = num_days / TRADING_DAYS_PER_YEAR
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    annualized_vol = daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    excess_return = daily_returns.mean() * TRADING_DAYS_PER_YEAR - risk_free_rate
    sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    win_rate = (daily_returns > 0).mean()

    return BacktestMetrics(
        total_return_pct=total_return * 100,
        annualized_return_pct=annualized_return * 100,
        annualized_volatility_pct=annualized_vol * 100,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_drawdown * 100,
        win_rate_pct=win_rate * 100,
        num_trading_days=num_days,
    )

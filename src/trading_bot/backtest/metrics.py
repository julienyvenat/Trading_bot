"""Calcul des métriques de performance d'une courbe d'equity (et, si
disponibles, des trades individuels qui l'ont produite — voir
`trading_bot.backtest.trades`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.backtest.trades import Trade

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestMetrics:
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    # Comme le Sharpe, mais ne pénalise que la volatilité "défavorable" (les
    # jours en perte) : deux stratégies de même Sharpe mais dont l'une a sa
    # volatilité surtout à la hausse (asymétrie souhaitable) ressortent
    # différemment ici alors qu'elles seraient confondues au Sharpe seul.
    sortino_ratio: float
    # Rendement annualisé / |max drawdown| : rapporte le rendement obtenu au
    # pire creux traversé pour l'obtenir, plutôt qu'à la volatilité globale
    # (plus parlant pour juger si un drawdown donné "en valait la peine").
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float  # % de JOURS positifs (existant, inchangé)
    # Gains bruts / pertes brutes. Calculé à partir des trades réalisés si
    # disponibles (définition standard), sinon approximé à partir des
    # variations quotidiennes de l'equity (voir `compute_metrics`).
    profit_factor: float
    num_trading_days: int
    num_trades: int
    trade_win_rate_pct: float  # % de TRADES gagnants (distinct de win_rate_pct, par jour)
    avg_win_pct: float  # rendement moyen des trades gagnants (sur la valeur engagée à l'entrée)
    avg_loss_pct: float  # rendement moyen des trades perdants (négatif)
    win_loss_ratio: float  # avg_win_pct / |avg_loss_pct|

    def summary(self) -> str:
        return (
            f"Rendement total       : {self.total_return_pct:+.2f}%\n"
            f"Rendement annualisé   : {self.annualized_return_pct:+.2f}%\n"
            f"Volatilité annualisée : {self.annualized_volatility_pct:.2f}%\n"
            f"Ratio de Sharpe       : {self.sharpe_ratio:.2f}\n"
            f"Ratio de Sortino      : {self.sortino_ratio:.2f}\n"
            f"Ratio de Calmar       : {self.calmar_ratio:.2f}\n"
            f"Max drawdown          : {self.max_drawdown_pct:.2f}%\n"
            f"Taux de jours positifs: {self.win_rate_pct:.2f}%\n"
            f"Profit factor         : {self.profit_factor:.2f}\n"
            f"Jours de trading      : {self.num_trading_days}\n"
            f"--- Statistiques par trade ---\n"
            f"Nombre de trades      : {self.num_trades}\n"
            f"Taux de trades gagnants: {self.trade_win_rate_pct:.2f}%\n"
            f"Gain moyen (trades +) : {self.avg_win_pct:+.2f}%\n"
            f"Perte moyenne (trades -): {self.avg_loss_pct:+.2f}%\n"
            f"Ratio gain/perte      : {self.win_loss_ratio:.2f}"
        )


def _trade_stats(trades: list[Trade]) -> dict:
    num_trades = len(trades)
    if num_trades == 0:
        return dict(num_trades=0, trade_win_rate_pct=0.0, avg_win_pct=0.0, avg_loss_pct=0.0, win_loss_ratio=0.0)

    wins = [t.pnl_pct for t in trades if t.pnl > 0]
    losses = [t.pnl_pct for t in trades if t.pnl < 0]

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    if avg_loss != 0:
        win_loss_ratio = avg_win / abs(avg_loss)
    else:
        win_loss_ratio = float("inf") if avg_win > 0 else 0.0

    return dict(
        num_trades=num_trades,
        trade_win_rate_pct=len(wins) / num_trades * 100,
        avg_win_pct=avg_win * 100,
        avg_loss_pct=avg_loss * 100,
        win_loss_ratio=win_loss_ratio,
    )


def _profit_factor_from_trades(trades: list[Trade]) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl < 0)
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _profit_factor_from_equity(equity_curve: pd.Series) -> float:
    """Repli utilisé quand aucun trade n'est fourni (ex: appelant qui ne
    dispose que d'une courbe d'equity) : approxime le profit factor à partir
    des variations quotidiennes plutôt que des trades réalisés. Moins
    standard que la version par trade, mais évite de bloquer les appelants
    existants qui n'ont pas encore de tracking de trades."""
    deltas = equity_curve.diff().dropna()
    gross_profit = deltas[deltas > 0].sum()
    gross_loss = -deltas[deltas < 0].sum()
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[Trade] | None = None,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """Calcule les métriques standards à partir d'une courbe d'equity (valeur
    totale du portefeuille au fil du temps) et, si fournis, des trades
    réalisés (voir `trading_bot.backtest.trades.TradeTracker`)."""
    equity_curve = equity_curve.dropna()
    trades = trades or []

    if len(equity_curve) < 2:
        return BacktestMetrics(
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            annualized_volatility_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            num_trading_days=len(equity_curve),
            **_trade_stats(trades),
        )

    daily_returns = equity_curve.pct_change().dropna()
    num_days = len(daily_returns)

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = num_days / TRADING_DAYS_PER_YEAR
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    annualized_vol = daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    excess_return = daily_returns.mean() * TRADING_DAYS_PER_YEAR - risk_free_rate
    sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0.0

    downside_returns = daily_returns[daily_returns < 0]
    downside_dev = downside_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 0.0
    sortino = excess_return / downside_dev if downside_dev > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    win_rate = (daily_returns > 0).mean()
    profit_factor = _profit_factor_from_trades(trades) if trades else _profit_factor_from_equity(equity_curve)

    return BacktestMetrics(
        total_return_pct=total_return * 100,
        annualized_return_pct=annualized_return * 100,
        annualized_volatility_pct=annualized_vol * 100,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown_pct=max_drawdown * 100,
        win_rate_pct=win_rate * 100,
        profit_factor=profit_factor,
        num_trading_days=num_days,
        **_trade_stats(trades),
    )


def money_weighted_return(
    flows: list[tuple[pd.Timestamp, float]], end: pd.Timestamp, final_value: float
) -> float | None:
    """TRI annualisé (rendement pondéré par l'argent) : taux r tel que la
    somme des versements `flows` (date, montant > 0 versé) capitalisés à r
    jusqu'à `end` égale `final_value`. Résolu par dichotomie (la fonction
    est monotone en r pour des versements tous positifs). None si indéfini."""
    flows = [(pd.Timestamp(d), float(a)) for d, a in flows if a]
    if not flows or final_value <= 0:
        return None
    end = pd.Timestamp(end)

    def future_value(rate: float) -> float:
        return sum(a * (1 + rate) ** ((end - d).days / 365.25) for d, a in flows)

    low, high = -0.99, 10.0
    if not future_value(low) <= final_value <= future_value(high):
        return None
    for _ in range(200):
        mid = (low + high) / 2
        if future_value(mid) < final_value:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def worst_rolling_return(curve: pd.Series, window: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """Pire rendement sur `window` séances glissantes (ex: pire 12 mois)."""
    curve = curve.dropna()
    if len(curve) <= window:
        return None
    return float((curve / curve.shift(window) - 1).min())


def max_drawdown_recovery(curve: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    """(date du plus haut précédant le pire creux, date du creux, date où la
    courbe retrouve ce plus haut — None si jamais retrouvé)."""
    curve = curve.dropna()
    if curve.empty:
        return None, None, None
    drawdown = curve / curve.cummax() - 1
    trough = drawdown.idxmin()
    if drawdown.loc[trough] >= 0:
        return None, None, None
    peak_value = curve.loc[:trough].max()
    peak = curve.loc[:trough].idxmax()
    after = curve.loc[trough:]
    recovered = after[after >= peak_value]
    return peak, trough, (recovered.index[0] if not recovered.empty else None)

"""Moteur de backtest event-driven (jour par jour).

Réutilise exactement les mêmes briques que le trading live (stratégies,
allocateur multi-stratégies, gestion du risque) pour garantir une parité
maximale entre backtest et live : ce qui est testé est ce qui est tradé.

Simplification assumée : le signal calculé à partir de la clôture du jour J
est exécuté à la clôture de J (pas de décalage J+1 open). C'est une
approximation optimiste courante pour un premier prototype ; à garder en
tête en comparant les résultats du backtest à ceux du paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_bot.backtest.metrics import BacktestMetrics, compute_metrics
from trading_bot.config import AppConfig
from trading_bot.indicators import atr
from trading_bot.portfolio.allocator import SignalAllocator
from trading_bot.portfolio.risk import RiskManager
from trading_bot.strategies.registry import build_enabled_strategies

MIN_TRADE_VALUE = 1.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    metrics: BacktestMetrics
    final_positions: dict[str, float] = field(default_factory=dict)


def _portfolio_value(positions: dict[str, float], prices: pd.Series) -> float:
    value = 0.0
    for symbol, qty in positions.items():
        if qty == 0:
            continue
        price = prices.get(symbol)
        if price is None or pd.isna(price):
            continue
        value += qty * price
    return value


def run_backtest(config: AppConfig, data_by_symbol: dict[str, pd.DataFrame]) -> BacktestResult:
    if not data_by_symbol:
        raise ValueError("Aucune donnée historique fournie pour le backtest.")

    strategies_with_weights = build_enabled_strategies(config.strategies)
    if not strategies_with_weights:
        raise ValueError("Aucune stratégie active dans la configuration.")

    allocator = SignalAllocator(strategies_with_weights, allow_short=config.risk.allow_short)
    exposure_series = allocator.target_exposure_series(data_by_symbol)
    atr_series = {sym: atr(df, config.risk.atr_window) for sym, df in data_by_symbol.items()}

    close_df = pd.DataFrame({sym: df["close"] for sym, df in data_by_symbol.items()}).sort_index().ffill()

    start = pd.Timestamp(config.backtest.start_date)
    end = pd.Timestamp(config.backtest.end_date) if config.backtest.end_date else close_df.index.max()
    dates = close_df.index[(close_df.index >= start) & (close_df.index <= end)]

    risk_manager = RiskManager(config.risk)
    commission_pct = config.backtest.commission_pct

    cash = float(config.backtest.initial_cash)
    positions: dict[str, float] = dict.fromkeys(data_by_symbol, 0.0)
    equity_records: dict[pd.Timestamp, float] = {}

    for date in dates:
        prices = close_df.loc[date]
        equity = cash + _portfolio_value(positions, prices)
        if equity <= 0:
            # Compte "ruiné" (ne devrait pas arriver avec une gestion du risque saine) : on arrête.
            break

        # 1) Construit les tailles candidates par symbole à partir des signaux du jour
        raw_sizings = {}
        for symbol in data_by_symbol:
            series = exposure_series[symbol]
            exposure = series.loc[date] if date in series.index else None
            if exposure is None or pd.isna(exposure) or abs(exposure) <= 1e-9:
                continue

            price = prices.get(symbol)
            if price is None or pd.isna(price) or price <= 0:
                continue

            atr_value = atr_series[symbol].loc[date] if date in atr_series[symbol].index else None
            atr_value = None if atr_value is None or pd.isna(atr_value) else float(atr_value)

            raw_sizings[symbol] = risk_manager.size_single(symbol, float(exposure), float(price), atr_value)

        final_sizings = risk_manager.apply_portfolio_caps(raw_sizings)

        # 2) Rebalance vers les poids cibles (0 pour les symboles non sélectionnés)
        for symbol in data_by_symbol:
            price = prices.get(symbol)
            if price is None or pd.isna(price) or price <= 0:
                continue

            target_weight = final_sizings[symbol].target_weight if symbol in final_sizings else 0.0
            target_qty = (target_weight * equity) / price
            current_qty = positions[symbol]
            delta_qty = target_qty - current_qty

            if abs(delta_qty) * price < MIN_TRADE_VALUE:
                continue

            trade_value = delta_qty * price
            commission = abs(trade_value) * commission_pct
            cash -= trade_value + commission
            positions[symbol] = target_qty

        equity_records[date] = cash + _portfolio_value(positions, prices)

    equity_curve = pd.Series(equity_records).sort_index()
    metrics = compute_metrics(equity_curve)
    return BacktestResult(equity_curve=equity_curve, metrics=metrics, final_positions=positions)

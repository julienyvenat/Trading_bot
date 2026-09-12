"""Moteur de backtest event-driven (jour par jour).

Réutilise exactement les mêmes briques que le trading live (stratégies,
allocateur multi-stratégies, gestion du risque, stop suiveur, coupe-circuits,
calendrier de marché) pour garantir une parité maximale entre backtest et
live : ce qui est testé est ce qui est tradé.

Simplification assumée : le signal calculé à partir de la clôture du jour J
est exécuté à la clôture de J (pas de décalage J+1 open). C'est une
approximation optimiste courante pour un premier prototype ; à garder en
tête en comparant les résultats du backtest à ceux du paper trading. Le
stop suiveur, lui, est vérifié contre le plus bas/plus haut intrajournalier
(low/high), ce qui est plus réaliste qu'une simple comparaison de clôture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_bot.backtest.metrics import BacktestMetrics, compute_metrics
from trading_bot.config import AppConfig
from trading_bot.indicators import atr
from trading_bot.logger import get_logger
from trading_bot.market_calendar import MarketCalendar
from trading_bot.portfolio.allocator import SignalAllocator
from trading_bot.portfolio.circuit_breaker import CircuitBreaker, RiskState, apply_halt
from trading_bot.portfolio.risk import RiskManager
from trading_bot.portfolio.stops import StopLevel, is_triggered, update_stop
from trading_bot.strategies.registry import build_enabled_strategies

logger = get_logger()

MIN_TRADE_VALUE = 1.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    metrics: BacktestMetrics
    final_positions: dict[str, float] = field(default_factory=dict)
    num_stop_exits: int = 0
    final_risk_state: RiskState | None = None


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

    calendar = MarketCalendar(config.market.calendar)
    trading_days = calendar.trading_days(start, end)
    in_range = (close_df.index >= start) & (close_df.index <= end)
    dates = close_df.index[in_range & close_df.index.isin(trading_days)]

    missing = in_range.sum() - len(dates)
    if missing > 0:
        logger.warning(
            "%d jour(s) présent(s) dans les données mais absents du calendrier %s ont été ignorés.",
            missing,
            config.market.calendar,
        )

    risk_manager = RiskManager(config.risk)
    circuit_breaker = CircuitBreaker(config.risk)
    commission_pct = config.backtest.commission_pct

    cash = float(config.backtest.initial_cash)
    positions: dict[str, float] = dict.fromkeys(data_by_symbol, 0.0)
    trailing_stops: dict[str, StopLevel] = {}
    equity_records: dict[pd.Timestamp, float] = {}
    risk_state: RiskState | None = None
    num_stop_exits = 0

    for dt in dates:
        prices = close_df.loc[dt]
        equity = cash + _portfolio_value(positions, prices)
        if equity <= 0:
            break  # compte "ruiné" : ne devrait pas arriver avec une gestion du risque saine

        if risk_state is None:
            risk_state = RiskState.initial(equity, today=dt.date())

        # 1) Vérifie les stops suiveurs établis la veille contre le range du jour.
        for symbol, qty in list(positions.items()):
            if qty == 0:
                continue
            stop = trailing_stops.get(symbol)
            df = data_by_symbol.get(symbol)
            if stop is None or df is None or dt not in df.index:
                continue

            low = float(df.loc[dt, "low"])
            high = float(df.loc[dt, "high"])
            if is_triggered(stop, low, high):
                exit_price = stop.stop_price
                commission = abs(qty * exit_price) * commission_pct
                cash += qty * exit_price - commission
                positions[symbol] = 0.0
                trailing_stops.pop(symbol, None)
                num_stop_exits += 1

        equity = cash + _portfolio_value(positions, prices)
        if equity <= 0:
            break

        risk_state = circuit_breaker.update(risk_state, equity, dt.date())

        # 2) Construit les tailles candidates par symbole à partir des signaux du jour.
        raw_sizings = {}
        for symbol in data_by_symbol:
            series = exposure_series[symbol]
            exposure = series.loc[dt] if dt in series.index else None
            if exposure is None or pd.isna(exposure) or abs(exposure) <= 1e-9:
                continue

            price = prices.get(symbol)
            if price is None or pd.isna(price) or price <= 0:
                continue

            atr_value = atr_series[symbol].loc[dt] if dt in atr_series[symbol].index else None
            atr_value = None if atr_value is None or pd.isna(atr_value) else float(atr_value)

            raw_sizings[symbol] = risk_manager.size_single(symbol, float(exposure), float(price), atr_value)

        final_sizings = risk_manager.apply_portfolio_caps(raw_sizings)

        # 3) Applique les coupe-circuits (halt journalier / drawdown) sur les tailles finales.
        current_weights = {
            sym: (positions[sym] * prices[sym] / equity if pd.notna(prices.get(sym)) and equity > 0 else 0.0)
            for sym in data_by_symbol
        }
        final_sizings = apply_halt(final_sizings, current_weights, risk_state)

        # 4) Rebalance vers les poids cibles (0 pour les symboles non sélectionnés).
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

        # 5) Met à jour le stop suiveur de chaque position encore ouverte, pour demain.
        for symbol, qty in positions.items():
            if qty == 0:
                trailing_stops.pop(symbol, None)
                continue

            price = prices.get(symbol)
            if price is None or pd.isna(price):
                continue

            atr_value = atr_series[symbol].loc[dt] if dt in atr_series[symbol].index else None
            atr_value = None if atr_value is None or pd.isna(atr_value) else float(atr_value)
            direction = 1 if qty > 0 else -1
            trailing_stops[symbol] = update_stop(
                trailing_stops.get(symbol), direction, float(price), atr_value, config.risk.atr_stop_multiple
            )

        equity_records[dt] = cash + _portfolio_value(positions, prices)

    equity_curve = pd.Series(equity_records).sort_index()
    metrics = compute_metrics(equity_curve)
    return BacktestResult(
        equity_curve=equity_curve,
        metrics=metrics,
        final_positions=positions,
        num_stop_exits=num_stop_exits,
        final_risk_state=risk_state,
    )

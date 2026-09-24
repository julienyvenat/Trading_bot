"""Backtest du mode `core_satellite` (voir `trading_bot.portfolio.
core_satellite`) : même `plan_rebalance` que le live, décidé sur la clôture
du jour J, exécuté à l'ouverture de J+1 avec les quantités affichées la
veille (comme un ordre au marché passé le matin d'après la notification) —
ventes d'abord, achats plafonnés au cash disponible à l'ouverture.

Apports (`backtest.monthly_contribution`) : versés en cash à l'ouverture du
premier jour de bourse de chaque mois (pas le mois de départ). Comptabilité
par parts pour ne pas mélanger performance et apports :
  - NAV par part (rendement pondéré par le temps) : base de toutes les
    métriques de `BacktestMetrics` (CAGR, Sharpe, max drawdown...) et de la
    courbe `nav_curve` ; chaque apport achète des parts à la NAV de la veille ;
  - TRI (rendement pondéré par l'argent, `money_weighted_return_pct`) : ce
    que l'investisseur a réellement obtenu sur ses versements datés ;
  - `equity_curve` : valeur réelle du compte, apports compris.
Sans apport, NAV x capital initial == equity : métriques identiques à un
backtest classique.
"""

from __future__ import annotations

import pandas as pd

from trading_bot.backtest.metrics import compute_metrics, money_weighted_return
from trading_bot.config import AppConfig
from trading_bot.logger import get_logger
from trading_bot.market_calendar import MarketCalendar
from trading_bot.portfolio.core_satellite import CoreSatelliteParams, calendar_rebalance_due, plan_rebalance
from trading_bot.portfolio.fees import ExecutionRules, max_affordable_qty

logger = get_logger()


class _BacktestCalendar:
    """Adaptateur : « jours de bourse » = dates effectivement simulées."""

    def __init__(self, dates: pd.DatetimeIndex) -> None:
        self._dates = dates

    def trading_days(self, start, end) -> pd.DatetimeIndex:
        return self._dates[(self._dates >= pd.Timestamp(start)) & (self._dates <= pd.Timestamp(end))]


def run_core_satellite_backtest(config: AppConfig, data_by_symbol: dict[str, pd.DataFrame], params: CoreSatelliteParams):
    from trading_bot.backtest.engine import BacktestResult

    symbols = list(params.targets)
    missing = [s for s in symbols if s not in data_by_symbol]
    if missing:
        raise ValueError(f"core_satellite : pas de données pour {', '.join(missing)}.")

    close_df = pd.DataFrame({s: data_by_symbol[s]["close"] for s in symbols}).sort_index().ffill()
    open_df = pd.DataFrame({s: data_by_symbol[s]["open"] for s in symbols}).sort_index()
    open_df = open_df.where(open_df > 0).fillna(close_df.shift(1)).ffill()

    start = pd.Timestamp(config.backtest.start_date)
    end = pd.Timestamp(config.backtest.end_date) if config.backtest.end_date else close_df.index.max()
    trading_days = MarketCalendar(config.market.calendar).trading_days(start, end)
    in_range = (close_df.index >= start) & (close_df.index <= end)
    dates = close_df.index[in_range & close_df.index.normalize().isin(trading_days)]
    # On ne simule qu'à partir du premier jour où TOUTES les poches ont un prix.
    complete = close_df.loc[dates].notna().all(axis=1)
    dates = dates[complete.values]
    if len(dates) == 0:
        raise ValueError("core_satellite : aucune date commune à toutes les poches dans la période demandée.")

    rules = ExecutionRules.from_backtest_config(config.backtest)
    calendar = _BacktestCalendar(pd.DatetimeIndex(dates))
    contribution = float(config.backtest.monthly_contribution or 0.0)

    cash = float(config.backtest.initial_cash)
    qty = dict.fromkeys(symbols, 0.0)
    units = cash
    nav_prev = 1.0
    flows: list[tuple[pd.Timestamp, float]] = [(dates[0], cash)]
    equity_records: dict[pd.Timestamp, float] = {}
    nav_records: dict[pd.Timestamp, float] = {}
    fees_total = 0.0
    orders_total = 0
    last_month = (dates[0].year, dates[0].month)
    last_calendar_year = dates[0].year
    pending: list = []

    for dt in dates:
        if contribution > 0 and (dt.year, dt.month) != last_month:
            cash += contribution
            units += contribution / nav_prev
            flows.append((dt, contribution))
        last_month = (dt.year, dt.month)

        if pending:
            opens = open_df.loc[dt]
            for order in pending:  # ventes d'abord (ordre de `plan_rebalance`)
                price = float(opens[order.symbol])
                if order.side == "sell":
                    q = min(order.qty, qty[order.symbol])
                else:
                    q = min(order.qty, max_affordable_qty(cash, price, rules.commission, rules.whole_shares))
                if q <= 0:
                    continue
                notional = q * price
                fee = rules.commission.fee(notional)
                sign = 1 if order.side == "buy" else -1
                qty[order.symbol] += sign * q
                cash -= sign * notional + fee
                fees_total += fee
                orders_total += 1
            pending = []

        prices = {s: float(close_df.at[dt, s]) for s in symbols}
        equity = cash + sum(qty[s] * prices[s] for s in symbols)
        nav_prev = equity / units if units > 0 else 1.0
        equity_records[dt] = equity
        nav_records[dt] = nav_prev

        calendar_due = calendar_rebalance_due(params, dt, last_calendar_year, calendar)
        if calendar_due:
            last_calendar_year = dt.year
        plan = plan_rebalance(qty, prices, cash, params, rules, calendar_due=calendar_due)
        pending = plan.orders

    equity_curve = pd.Series(equity_records).sort_index()
    nav_curve = pd.Series(nav_records).sort_index()
    metrics = compute_metrics(nav_curve * float(config.backtest.initial_cash))
    mwr = money_weighted_return(flows, equity_curve.index[-1], float(equity_curve.iloc[-1]))
    return BacktestResult(
        equity_curve=equity_curve,
        metrics=metrics,
        final_positions=qty,
        total_fees=fees_total,
        num_orders=orders_total,
        nav_curve=nav_curve,
        total_contributed=sum(amount for _, amount in flows),
        money_weighted_return_pct=None if mwr is None else mwr * 100,
    )

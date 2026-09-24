"""Moteur de backtest event-driven (jour par jour).

Réutilise exactement les mêmes briques que le trading live (stratégies,
allocateur multi-stratégies, gestion du risque, stop suiveur, coupe-circuits,
calendrier de marché) pour garantir une parité maximale entre backtest et
live : ce qui est testé est ce qui est tradé.

Exécution des ordres : le signal (et le dimensionnement du risque qui en
découle) est calculé à partir de la clôture du jour J, mais exécuté à
l'ouverture du jour de bourse suivant J+1 — la quantité est recalculée à ce
moment-là avec l'equity et le prix d'ouverture réels, comme le ferait un
opérateur qui déciderait le soir sur la clôture et passerait ses ordres le
lendemain matin. C'est plus réaliste qu'une exécution immédiate à la clôture
du jour même où le signal est calculé (approximation optimiste qu'on
utilisait auparavant). Le stop suiveur, lui, reste vérifié le jour même
contre le plus bas/plus haut intrajournalier (low/high), ce qui est déjà
réaliste.

Coûts d'exécution (voir `trading_bot.portfolio.fees`) : frais par paliers
optionnels (`backtest.commission_schedule`, sinon `commission_pct`), actions
entières + achats plafonnés au cash disponible (`backtest.whole_shares`,
PEA), et garde-fous anti-frais (`min_order_value`, `max_fee_pct`,
`rebalance_tolerance_pct`) — les mêmes règles que le live. Les ventes d'une
même séance sont exécutées avant les achats (le cash libéré finance les
achats).

Stops (`risk.stop_mode`) : "atr" (historique), "trailing_pct" (sémantique
d'un Stop Suiveur natif de courtier, écart figé à l'entrée, remonté sur les
plus hauts, exécuté au seuil ou à l'ouverture en cas de gap) ou "none".
`risk.stop_reentry: "sma"` bloque une ré-entrée après stop tant que la
clôture n'est pas repassée au-dessus de sa SMA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_bot.backtest.metrics import BacktestMetrics, compute_metrics
from trading_bot.backtest.trades import Trade, TradeTracker
from trading_bot.config import AppConfig
from trading_bot.indicators import atr
from trading_bot.logger import get_logger
from trading_bot.market_calendar import MarketCalendar
from trading_bot.portfolio.allocator import SignalAllocator
from trading_bot.indicators import sma
from trading_bot.portfolio.circuit_breaker import CircuitBreaker, RiskState, apply_halt
from trading_bot.portfolio.fees import CommissionModel, ExecutionRules, decide_order_qty
from trading_bot.portfolio.regime import regime_scale_series
from trading_bot.portfolio.risk import RiskManager
from trading_bot.portfolio.stops import (
    PctTrailingStop,
    StopLevel,
    entry_trail_pct,
    is_triggered,
    pct_stop_exit_price,
    update_stop,
    validate_stop_mode,
)
from trading_bot.portfolio.symbol_track_record import load_track_record
from trading_bot.portfolio.volatility_filter import volatility_scale_series
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
    # Trades individuels RÉALISÉS (voir `trading_bot.backtest.trades`) : les
    # positions encore ouvertes à la fin du backtest n'y figurent pas.
    trades: list[Trade] = field(default_factory=list)
    # Coûts réels de la stratégie : total des frais payés et nombre d'ORDRES
    # exécutés (achats + ventes, stops compris) — ce qu'il faut réellement
    # passer à la main sur un compte sans API.
    total_fees: float = 0.0
    num_orders: int = 0


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


def _execute_at_open(
    pending_target_weights: dict[str, float],
    positions: dict[str, float],
    cash: float,
    open_prices: pd.Series,
    commission_pct: float,
    dt: pd.Timestamp | None = None,
    trade_tracker: TradeTracker | None = None,
    rules: ExecutionRules | None = None,
    stats: dict | None = None,
) -> tuple[dict[str, float], float]:
    """Exécute, au prix d'ouverture `open_prices`, les poids cibles décidés à
    la clôture du jour précédent. Fonction pure (ne mute pas `positions`) :
    renvoie les positions et le cash mis à jour.

    `dt`/`trade_tracker` sont optionnels (et vont toujours de pair) : quand
    fournis, chaque fill déclenché ici est aussi transmis au tracker pour
    alimenter les métriques par trade (voir `trading_bot.backtest.trades`).
    Optionnels pour ne pas casser les appels directs existants (tests) qui
    n'ont pas besoin de ce suivi.

    `rules` (optionnel) : contraintes d'exécution (frais par paliers, actions
    entières, garde-fous, voir `trading_bot.portfolio.fees`). None : frais
    proportionnels `commission_pct`, quantités fractionnaires (historique).
    `stats` (optionnel) : dict mis à jour en place avec "fees" et "orders".
    """
    if rules is None:
        rules = ExecutionRules(commission=CommissionModel(pct=commission_pct))
    equity_at_open = cash + _portfolio_value(positions, open_prices)
    updated_positions = dict(positions)

    # Ventes d'abord : le cash qu'elles libèrent finance les achats de la
    # même séance (indispensable quand les achats sont plafonnés au cash).
    def _is_sell(item: tuple[str, float]) -> bool:
        symbol, target_weight = item
        price = open_prices.get(symbol)
        if price is None or pd.isna(price) or price <= 0:
            return False
        return (target_weight * equity_at_open) / price < updated_positions.get(symbol, 0.0)

    ordered = sorted(pending_target_weights.items(), key=lambda item: 0 if _is_sell(item) else 1)

    for symbol, target_weight in ordered:
        price = open_prices.get(symbol)
        if price is None or pd.isna(price) or price <= 0:
            continue  # pas de prix d'ouverture ce jour-là : ordre perdu (marché fermé pour ce symbole)

        target_qty = (target_weight * equity_at_open) / price if equity_at_open > 0 else 0.0
        current_qty = updated_positions[symbol]
        cash_available = cash if rules.whole_shares else None
        target_qty, _reason = decide_order_qty(current_qty, target_qty, price, equity_at_open, cash_available, rules)
        delta_qty = target_qty - current_qty
        if abs(delta_qty) * price < MIN_TRADE_VALUE:
            continue

        trade_value = delta_qty * price
        commission = rules.commission.fee(trade_value)
        cash -= trade_value + commission
        updated_positions[symbol] = target_qty
        if stats is not None:
            stats["fees"] = stats.get("fees", 0.0) + commission
            stats["orders"] = stats.get("orders", 0) + 1

        if trade_tracker is not None and dt is not None:
            trade_tracker.record_fill(symbol, dt, old_qty=current_qty, new_qty=target_qty, price=price, commission=commission)

    return updated_positions, cash


def run_backtest(
    config: AppConfig,
    data_by_symbol: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame | None = None,
    volatility_benchmark_df: pd.DataFrame | None = None,
) -> BacktestResult:
    """Lance le backtest.

    `benchmark_df` : données OHLCV du symbole de référence du filtre de
    régime (`config.market.regime_filter.symbol`), uniquement nécessaire s'il
    n'est PAS déjà dans `data_by_symbol` (auquel cas il est réutilisé
    directement). Ce symbole n'est jamais tradé pour lui-même via ce
    paramètre : il ne sert qu'à calculer le facteur de régime.

    `volatility_benchmark_df` : même principe pour le proxy de volatilité du
    filtre `config.market.volatility_filter.symbol` (voir
    `trading_bot.portfolio.volatility_filter`).
    """
    if not data_by_symbol:
        raise ValueError("Aucune donnée historique fournie pour le backtest.")

    strategies_with_weights = build_enabled_strategies(config.strategies)
    if not strategies_with_weights:
        raise ValueError("Aucune stratégie active dans la configuration.")

    rotation_configs = {
        s.name: s.universe_rotation for s in config.strategies if s.enabled and s.universe_rotation.enabled
    }
    track_record = None
    if any(r.metric == "track_record" for r in rotation_configs.values()):
        track_record = load_track_record(config.live.track_record_file)
    allocator = SignalAllocator(
        strategies_with_weights,
        allow_short=config.risk.allow_short,
        rotation_configs=rotation_configs or None,
        base_symbols=config.symbols if rotation_configs else None,
        track_record=track_record,
    )
    exposure_series = allocator.target_exposure_series(data_by_symbol)
    atr_series = {sym: atr(df, config.risk.atr_window) for sym, df in data_by_symbol.items()}

    close_df = pd.DataFrame({sym: df["close"] for sym, df in data_by_symbol.items()}).sort_index().ffill()
    open_df = pd.DataFrame({sym: df["open"] for sym, df in data_by_symbol.items()}).sort_index().ffill()

    regime_config = config.market.regime_filter
    regime_scale_by_date: pd.Series | None = None
    if regime_config.enabled:
        bench_source = benchmark_df if benchmark_df is not None else data_by_symbol.get(regime_config.symbol)
        if bench_source is None:
            logger.warning(
                "Filtre de régime activé pour %s mais aucune donnée disponible (ni dans l'univers "
                "tradé, ni via `benchmark_df`) : filtre ignoré pour ce backtest.",
                regime_config.symbol,
            )
        else:
            regime_scale_by_date = regime_scale_series(
                bench_source["close"], regime_config.sma_window, regime_config.bearish_exposure_scale
            )

    volatility_config = config.market.volatility_filter
    volatility_scale_by_date: pd.Series | None = None
    if volatility_config.enabled:
        vol_source = volatility_benchmark_df if volatility_benchmark_df is not None else data_by_symbol.get(
            volatility_config.symbol
        )
        if vol_source is None:
            logger.warning(
                "Filtre de volatilité activé pour %s mais aucune donnée disponible (ni dans l'univers "
                "tradé, ni via `volatility_benchmark_df`) : filtre ignoré pour ce backtest.",
                volatility_config.symbol,
            )
        else:
            volatility_scale_by_date = volatility_scale_series(
                vol_source["close"],
                volatility_config.sma_window,
                volatility_config.spike_threshold_pct,
                volatility_config.spike_exposure_scale,
            )

    start = pd.Timestamp(config.backtest.start_date)
    end = pd.Timestamp(config.backtest.end_date) if config.backtest.end_date else close_df.index.max()

    calendar = MarketCalendar(config.market.calendar)
    trading_days = calendar.trading_days(start, end)
    in_range = (close_df.index >= start) & (close_df.index <= end)
    # Comparaison par DATE (jour de séance), pas par horodatage exact : des
    # bougies journalières tombent déjà à minuit (comportement inchangé),
    # mais des bougies intraday (5min, 1h...) ne tombent jamais exactement à
    # minuit — les comparer à `trading_days` telles quelles éliminerait
    # silencieusement la quasi-totalité des données. `.normalize()` ramène
    # chaque horodatage à sa date pour ne filtrer que sur les jours de
    # bourse valides, quelle que soit la granularité des bougies.
    dates = close_df.index[in_range & close_df.index.normalize().isin(trading_days)]

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
    rules = ExecutionRules.from_backtest_config(config.backtest)
    stop_mode = validate_stop_mode(config.risk.stop_mode)
    reentry_sma = None
    if config.risk.stop_reentry == "sma":
        reentry_sma = {sym: sma(close_df[sym], config.risk.stop_reentry_sma_window) for sym in data_by_symbol}
    elif config.risk.stop_reentry != "immediate":
        raise ValueError("`risk.stop_reentry` doit valoir 'immediate' ou 'sma'.")
    stats: dict = {"fees": 0.0, "orders": 0}

    cash = float(config.backtest.initial_cash)
    positions: dict[str, float] = dict.fromkeys(data_by_symbol, 0.0)
    trailing_stops: dict[str, StopLevel] = {}
    pct_stops: dict[str, PctTrailingStop] = {}
    # Symboles sortis sur stop, bloqués à la ré-entrée (`stop_reentry: sma`).
    stopped_out: set[str] = set()
    equity_records: dict[pd.Timestamp, float] = {}
    risk_state: RiskState | None = None
    num_stop_exits = 0
    trade_tracker = TradeTracker()
    # Poids cibles décidés à la clôture d'un jour, exécutés à l'ouverture du
    # jour de bourse suivant (voir docstring du module).
    pending_target_weights: dict[str, float] | None = None

    for dt in dates:
        # 0) Exécute, au prix d'ouverture d'aujourd'hui, la décision prise à
        # la clôture d'hier (rien à faire le tout premier jour).
        if pending_target_weights is not None:
            positions, cash = _execute_at_open(
                pending_target_weights,
                positions,
                cash,
                open_df.loc[dt],
                commission_pct,
                dt=dt,
                trade_tracker=trade_tracker,
                rules=rules,
                stats=stats,
            )
            pending_target_weights = None

        prices = close_df.loc[dt]
        equity = cash + _portfolio_value(positions, prices)
        if equity <= 0:
            break  # compte "ruiné" : ne devrait pas arriver avec une gestion du risque saine

        if risk_state is None:
            risk_state = RiskState.initial(equity, today=dt.date())

        # 1) Vérifie les stops suiveurs établis la veille contre le range du jour.
        for symbol, qty in list(positions.items()):
            if qty == 0 or stop_mode == "none":
                continue
            df = data_by_symbol.get(symbol)
            if df is None or dt not in df.index:
                continue

            low = float(df.loc[dt, "low"])
            high = float(df.loc[dt, "high"])
            if stop_mode == "trailing_pct":
                pct_stop = pct_stops.get(symbol)
                if pct_stop is None or qty < 0 or low > pct_stop.stop_price:
                    continue
                exit_price = pct_stop_exit_price(pct_stop, float(df.loc[dt, "open"]))
            else:
                stop = trailing_stops.get(symbol)
                if stop is None or not is_triggered(stop, low, high):
                    continue
                exit_price = stop.stop_price

            commission = rules.commission.fee(qty * exit_price)
            cash += qty * exit_price - commission
            stats["fees"] += commission
            stats["orders"] += 1
            trade_tracker.record_fill(
                symbol, dt, old_qty=qty, new_qty=0.0, price=exit_price, commission=commission, exit_reason="stop"
            )
            positions[symbol] = 0.0
            trailing_stops.pop(symbol, None)
            pct_stops.pop(symbol, None)
            num_stop_exits += 1
            if reentry_sma is not None:
                stopped_out.add(symbol)

        equity = cash + _portfolio_value(positions, prices)
        if equity <= 0:
            break

        risk_state = circuit_breaker.update(risk_state, equity, dt.date())

        # 2) Construit les tailles candidates par symbole à partir des signaux du jour,
        # modulées par le filtre de régime de marché et le filtre de
        # volatilité s'ils sont actifs (combinés multiplicativement : un pic
        # de volatilité peut survenir même en régime haussier, voir
        # `trading_bot.portfolio.volatility_filter`).
        regime_scale = 1.0
        if regime_scale_by_date is not None:
            regime_value = regime_scale_by_date.get(dt)
            if regime_value is not None and pd.notna(regime_value):
                regime_scale = float(regime_value)

        vol_scale = 1.0
        if volatility_scale_by_date is not None:
            vol_value = volatility_scale_by_date.get(dt)
            if vol_value is not None and pd.notna(vol_value):
                vol_scale = float(vol_value)

        raw_sizings = {}
        for symbol in data_by_symbol:
            series = exposure_series[symbol]
            exposure = series.loc[dt] if dt in series.index else None
            if exposure is None or pd.isna(exposure):
                continue
            if symbol in stopped_out:
                sma_value = reentry_sma[symbol].get(dt)
                close_value = prices.get(symbol)
                if sma_value is None or pd.isna(sma_value) or pd.isna(close_value) or close_value <= sma_value:
                    continue  # toujours bloqué : pas de ré-entrée tant que la clôture <= SMA
                stopped_out.discard(symbol)
            symbol_regime_scale = 1.0 if symbol in regime_config.exempt_symbols else regime_scale
            symbol_vol_scale = 1.0 if symbol in volatility_config.exempt_symbols else vol_scale
            exposure = float(exposure) * symbol_regime_scale * symbol_vol_scale
            if abs(exposure) <= 1e-9:
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

        # 4) Mémorise les poids cibles décidés aujourd'hui (0 pour les symboles non
        # sélectionnés, pour bien les flatten) : exécutés à l'ouverture de demain (étape 0).
        pending_target_weights = {
            symbol: (final_sizings[symbol].target_weight if symbol in final_sizings else 0.0)
            for symbol in data_by_symbol
        }

        # 5) Met à jour le stop suiveur de chaque position encore ouverte, pour demain.
        for symbol, qty in positions.items():
            if qty == 0:
                trailing_stops.pop(symbol, None)
                pct_stops.pop(symbol, None)
                continue
            if stop_mode == "none":
                continue

            price = prices.get(symbol)
            if price is None or pd.isna(price):
                continue

            atr_value = atr_series[symbol].loc[dt] if dt in atr_series[symbol].index else None
            atr_value = None if atr_value is None or pd.isna(atr_value) else float(atr_value)

            if stop_mode == "trailing_pct":
                if qty < 0:
                    continue  # stop suiveur natif modélisé pour des positions longues uniquement (PEA)
                df = data_by_symbol[symbol]
                day_high = float(df.loc[dt, "high"]) if dt in df.index else float(price)
                existing = pct_stops.get(symbol)
                if existing is None:
                    pct = entry_trail_pct(
                        config.risk.trailing_stop_pct, float(price), atr_value, config.risk.atr_stop_multiple
                    )
                    if pct is not None:
                        pct_stops[symbol] = PctTrailingStop(trail_pct=pct, high_water=day_high)
                else:
                    pct_stops[symbol] = existing.ratchet(day_high)
                continue

            direction = 1 if qty > 0 else -1
            trailing_stops[symbol] = update_stop(
                trailing_stops.get(symbol), direction, float(price), atr_value, config.risk.atr_stop_multiple
            )

        equity_records[dt] = cash + _portfolio_value(positions, prices)

    equity_curve = pd.Series(equity_records).sort_index()
    metrics = compute_metrics(equity_curve, trades=trade_tracker.completed_trades)
    return BacktestResult(
        equity_curve=equity_curve,
        metrics=metrics,
        final_positions=positions,
        num_stop_exits=num_stop_exits,
        final_risk_state=risk_state,
        trades=trade_tracker.completed_trades,
        total_fees=stats["fees"],
        num_orders=stats["orders"],
    )

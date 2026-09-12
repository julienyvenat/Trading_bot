"""Moteur live : boucle de décision tournant contre le compte paper Alpaca.

Intègre, en plus de la génération de signaux :
  - un stop suiveur ATR par position (persisté entre redémarrages) ;
  - les coupe-circuits de risque (perte journalière / drawdown, voir
    `trading_bot.portfolio.circuit_breaker`) ;
  - le vrai calendrier de marché (`trading_bot.market_calendar`) pour ne
    trader que pendant les heures de bourse, en tenant compte des jours
    fériés et fermetures anticipées, et pour dormir intelligemment jusqu'à
    la prochaine séance plutôt que de sonder en boucle.
"""

from __future__ import annotations

import time

import pandas as pd

from trading_bot.config import AppConfig, load_alpaca_credentials
from trading_bot.data.market_data import fetch_latest_bars
from trading_bot.execution.alpaca_broker import AlpacaBroker
from trading_bot.execution.rebalancer import execute_orders, plan_orders
from trading_bot.indicators import atr
from trading_bot.logger import get_logger
from trading_bot.market_calendar import MarketCalendar
from trading_bot.portfolio.allocator import SignalAllocator
from trading_bot.portfolio.circuit_breaker import CircuitBreaker, RiskState, apply_halt, should_flatten
from trading_bot.portfolio.risk import RiskManager
from trading_bot.portfolio.stops import is_triggered, update_stop
from trading_bot.state import LiveState, load_state, save_state
from trading_bot.strategies.registry import build_enabled_strategies

logger = get_logger()

# Ne dort jamais plus de 30 minutes d'un coup en dehors des heures de marché,
# pour rester réactif (logs réguliers, arrêt propre du process, etc.).
MAX_SLEEP_CHUNK_SECONDS = 1800


def _close_all_positions(current_qty: dict[str, float], broker: AlpacaBroker, dry_run: bool, reason: str) -> None:
    for symbol, qty in current_qty.items():
        if qty == 0:
            continue
        side = "sell" if qty > 0 else "buy"
        logger.warning("Flatten (%s) : %s %s %.4f", reason, side, symbol, abs(qty))
        if not dry_run:
            broker.submit_market_order(symbol, abs(qty), side)


def run_once(config: AppConfig, broker: AlpacaBroker, dry_run: bool, state: LiveState) -> LiveState:
    """Exécute un cycle complet : données -> stops -> coupe-circuits ->
    signaux -> allocation -> risque -> ordres. Renvoie l'état mis à jour."""
    credentials = load_alpaca_credentials()

    logger.info("Récupération des données de marché pour %s...", ", ".join(config.symbols))
    data_by_symbol = fetch_latest_bars(config.symbols, config.timeframe, credentials)
    if not data_by_symbol:
        logger.warning("Aucune donnée de marché reçue, cycle ignoré.")
        return state

    account = broker.get_account()
    positions = broker.get_positions()
    current_qty = {symbol: pos.qty for symbol, pos in positions.items()}
    last_prices = {symbol: broker.get_last_price(symbol) for symbol in config.symbols}
    equity = account.equity

    today = pd.Timestamp.now(tz="UTC").date()
    circuit_breaker = CircuitBreaker(config.risk)
    risk_state = state.risk_state or RiskState.initial(equity, today=today)
    risk_state = circuit_breaker.update(risk_state, equity, today)
    state.risk_state = risk_state

    if risk_state.drawdown_halted:
        logger.error(
            "COUPE-CIRCUIT DE DRAWDOWN ACTIF (equity %.2f) : flatten de toutes les positions, "
            "aucune nouvelle entrée tant qu'il n'est pas levé manuellement (voir README).",
            equity,
        )
    elif risk_state.daily_halted:
        logger.warning("Coupe-circuit de perte journalière actif : aucune nouvelle entrée ce cycle.")

    # 1) Stop suiveur : compare le dernier prix connu au stop établi lors d'un
    # cycle précédent. Ne capture pas les mouvements *entre* deux cycles.
    for symbol, qty in list(current_qty.items()):
        if qty == 0:
            continue
        stop = state.trailing_stops.get(symbol)
        price = last_prices.get(symbol)
        if stop is None or price is None:
            continue
        if is_triggered(stop, low=price, high=price):
            logger.warning("Stop suiveur déclenché sur %s au prix %.2f (stop=%.2f).", symbol, price, stop.stop_price)
            side = "sell" if qty > 0 else "buy"
            if not dry_run:
                broker.submit_market_order(symbol, abs(qty), side)
            current_qty[symbol] = 0.0
            state.trailing_stops.pop(symbol, None)

    # 2) Coupe-circuit de drawdown : flatten total, on s'arrête là pour ce cycle.
    if should_flatten(risk_state):
        _close_all_positions(current_qty, broker, dry_run, reason="coupe-circuit drawdown")
        state.trailing_stops.clear()
        return state

    # 3) Signaux multi-stratégies -> exposition cible -> dimensionnement du risque.
    strategies_with_weights = build_enabled_strategies(config.strategies)
    allocator = SignalAllocator(strategies_with_weights, allow_short=config.risk.allow_short)
    target_exposures = allocator.latest_target_exposures(data_by_symbol)
    logger.info("Expositions cibles : %s", {k: round(v, 3) for k, v in target_exposures.items()})

    risk_manager = RiskManager(config.risk)
    sizings = risk_manager.size_positions(target_exposures, data_by_symbol)

    # 4) Coupe-circuit journalier : bloque les nouvelles entrées / augmentations.
    current_weights = {
        sym: (current_qty.get(sym, 0.0) * last_prices[sym] / equity if last_prices.get(sym) and equity > 0 else 0.0)
        for sym in config.symbols
    }
    sizings = apply_halt(sizings, current_weights, risk_state)

    orders = plan_orders(sizings, current_qty, equity, last_prices)
    if orders:
        execute_orders(orders, broker, dry_run=dry_run)
    else:
        logger.info("Aucun ordre à passer ce cycle (portefeuille déjà à la cible).")

    # 5) Met à jour le stop suiveur de chaque position visée, pour le prochain cycle.
    for symbol in config.symbols:
        target_weight = sizings[symbol].target_weight if symbol in sizings else 0.0
        if abs(target_weight) <= 1e-9:
            state.trailing_stops.pop(symbol, None)
            continue

        df = data_by_symbol.get(symbol)
        price = last_prices.get(symbol)
        if df is None or price is None or len(df) <= config.risk.atr_window:
            continue

        atr_value = float(atr(df, config.risk.atr_window).iloc[-1])
        direction = 1 if target_weight > 0 else -1
        state.trailing_stops[symbol] = update_stop(
            state.trailing_stops.get(symbol), direction, price, atr_value, config.risk.atr_stop_multiple
        )

    return state


def run_forever(config: AppConfig, dry_run: bool = False) -> None:
    """Boucle infinie : exécute un cycle à chaque instant actionnable (marché
    ouvert, hors buffer de clôture), et dort intelligemment le reste du temps.
    """
    credentials = load_alpaca_credentials()
    broker = AlpacaBroker(credentials)
    calendar = MarketCalendar(config.market.calendar)

    state = load_state(config.live.state_file)
    if state.risk_state and state.risk_state.drawdown_halted:
        logger.error(
            "Coupe-circuit de DRAWDOWN restauré depuis %s : le bot reste À L'ARRÊT tant qu'il "
            "n'est pas levé manuellement (voir README, section 'Reprise après coupe-circuit').",
            config.live.state_file,
        )

    mode = "PAPER" if credentials.paper else "RÉEL"
    logger.info("Démarrage du moteur live en mode %s (dry_run=%s).", mode, dry_run)
    if not credentials.paper and not dry_run:
        logger.warning("!!! Compte RÉEL détecté : des ordres avec de l'argent réel vont être passés !!!")

    while True:
        try:
            now = pd.Timestamp.now(tz="UTC")
            if config.live.trade_only_when_market_open and not calendar.is_actionable(
                now, config.market.close_buffer_minutes
            ):
                wait_seconds = calendar.seconds_until_actionable(now, config.market.close_buffer_minutes)
                logger.info(
                    "Marché fermé (ou trop proche de la clôture) : prochain instant actionnable dans %.0f min.",
                    wait_seconds / 60,
                )
                time.sleep(min(wait_seconds, MAX_SLEEP_CHUNK_SECONDS))
                continue

            state = run_once(config, broker, dry_run=dry_run, state=state)
            save_state(config.live.state_file, state)
        except Exception:  # noqa: BLE001 - on ne veut jamais crasher la boucle live
            logger.exception("Erreur pendant le cycle de trading, on continue.")

        time.sleep(config.live.loop_interval_seconds)

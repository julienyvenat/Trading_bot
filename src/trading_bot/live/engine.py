"""Moteur live : boucle de décision tournant contre le compte paper Alpaca.

Intègre, en plus de la génération de signaux :
  - un stop suiveur ATR par position, délégué à un ordre STOP natif posé chez
    le broker (réagit en continu, contrairement à une vérification qui
    n'aurait lieu qu'une fois par cycle) — persisté entre redémarrages ;
  - les coupe-circuits de risque (perte journalière / drawdown, voir
    `trading_bot.portfolio.circuit_breaker`) ;
  - le vrai calendrier de marché (`trading_bot.market_calendar`) pour ne
    trader que pendant les heures de bourse, en tenant compte des jours
    fériés et fermetures anticipées, et pour dormir intelligemment jusqu'à
    la prochaine séance plutôt que de sonder en boucle.

Limite connue : après avoir soumis un ordre de rebalancement, on relit
immédiatement les positions chez le broker pour poser le stop sur la
quantité réelle. Un ordre marché au paper trading Alpaca fill quasiment tout
de suite, mais rien ne garantit qu'il soit déjà reflété au moment de cette
relecture (pas d'attente bloquante du fill) : dans de rares cas, le stop du
cycle pourrait porter sur la quantité d'avant l'ordre plutôt qu'après. Le
cycle suivant corrige la situation (ratchet + remplacement du stop).
"""

from __future__ import annotations

import time

import pandas as pd

from trading_bot.config import AppConfig, load_alpaca_credentials
from trading_bot.data.market_data import fetch_latest_bars
from trading_bot.execution.alpaca_broker import AlpacaBroker
from trading_bot.execution.broker_base import Broker
from trading_bot.execution.rebalancer import execute_orders, plan_orders
from trading_bot.indicators import atr
from trading_bot.logger import get_logger
from trading_bot.market_calendar import MarketCalendar
from trading_bot.portfolio.allocator import SignalAllocator
from trading_bot.portfolio.circuit_breaker import CircuitBreaker, RiskState, apply_halt, should_flatten
from trading_bot.portfolio.regime import latest_regime_scale
from trading_bot.portfolio.risk import RiskManager
from trading_bot.portfolio.stops import StopLevel, update_stop
from trading_bot.state import LiveState, load_state, save_state
from trading_bot.strategies.registry import build_enabled_strategies

logger = get_logger()

# Ne dort jamais plus de 30 minutes d'un coup en dehors des heures de marché,
# pour rester réactif (logs réguliers, arrêt propre du process, etc.).
MAX_SLEEP_CHUNK_SECONDS = 1800


def _cancel_stop_order(symbol: str, state: LiveState, broker: Broker, dry_run: bool) -> None:
    """Annule (best-effort) le stop natif existant d'un symbole, s'il y en a un."""
    order_id = state.stop_order_ids.pop(symbol, None)
    if order_id is None:
        return
    if dry_run:
        logger.info("DRY-RUN CANCEL STOP %s (ordre %s)", symbol, order_id)
        return
    broker.cancel_order(order_id)


def _submit_stop_order(symbol: str, qty: float, stop: StopLevel, state: LiveState, broker: Broker, dry_run: bool) -> None:
    side = "sell" if stop.direction > 0 else "buy"
    if dry_run:
        logger.info("DRY-RUN STOP %s %s %.4f @ %.2f", side.upper(), symbol, abs(qty), stop.stop_price)
        return
    order_id = broker.submit_stop_order(symbol, abs(qty), side, stop.stop_price)
    state.stop_order_ids[symbol] = order_id


def _close_all_positions(current_qty: dict[str, float], broker: Broker, dry_run: bool, reason: str) -> None:
    for symbol, qty in current_qty.items():
        if qty == 0:
            continue
        side = "sell" if qty > 0 else "buy"
        logger.warning("Flatten (%s) : %s %s %.4f", reason, side, symbol, abs(qty))
        if not dry_run:
            broker.submit_market_order(symbol, abs(qty), side)


def run_once(config: AppConfig, broker: Broker, dry_run: bool, state: LiveState) -> LiveState:
    """Exécute un cycle complet : données -> coupe-circuits -> signaux ->
    allocation -> risque -> ordres -> stops natifs. Renvoie l'état mis à jour."""
    credentials = load_alpaca_credentials()

    logger.info("Récupération des données de marché pour %s...", ", ".join(config.symbols))
    data_by_symbol = fetch_latest_bars(config.symbols, config.timeframe, credentials)
    if not data_by_symbol:
        logger.warning("Aucune donnée de marché reçue, cycle ignoré.")
        return state

    account = broker.get_account()
    positions = broker.get_positions()
    current_qty = {symbol: pos.qty for symbol, pos in positions.items()}

    # On récupère aussi le prix des positions ouvertes en dehors de l'univers
    # configuré (ex. symbole retiré de config.yaml, ou position ouverte
    # manuellement) : sans leur prix, `plan_orders` ne peut jamais les
    # liquider et elles restent orphelines indéfiniment, sans stop suiveur.
    orphan_symbols = set(current_qty) - set(config.symbols)
    if orphan_symbols:
        logger.warning(
            "Position(s) hors de l'univers configuré détectée(s) : %s. Elles seront liquidées "
            "(aucune stratégie ni stop suiveur ne les gère tant qu'elles ne sont pas dans "
            "config.yaml).",
            ", ".join(sorted(orphan_symbols)),
        )
    price_symbols = set(config.symbols) | orphan_symbols
    last_prices = {symbol: broker.get_last_price(symbol) for symbol in price_symbols}
    equity = account.equity

    # Le stop suiveur est désormais un ordre natif posé chez le broker (voir
    # plus bas) : s'il n'y a plus de position pour un symbole qu'on suivait,
    # c'est que son stop a fillé (ou que la position a été close autrement)
    # depuis le dernier cycle. On nettoie l'état en conséquence.
    for symbol in list(state.stop_order_ids):
        if current_qty.get(symbol, 0.0) == 0.0:
            logger.info("Stop natif sur %s considéré exécuté (plus de position) : nettoyage de l'état.", symbol)
            state.stop_order_ids.pop(symbol, None)
            state.trailing_stops.pop(symbol, None)

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

    # 1) Coupe-circuit de drawdown : annule tous les stops natifs puis flatten,
    # on s'arrête là pour ce cycle.
    if should_flatten(risk_state):
        for symbol, qty in current_qty.items():
            if qty != 0:
                _cancel_stop_order(symbol, state, broker, dry_run)
        _close_all_positions(current_qty, broker, dry_run, reason="coupe-circuit drawdown")
        state.trailing_stops.clear()
        return state

    # 2) Signaux multi-stratégies -> exposition cible, modulée par le filtre
    # de régime de marché s'il est actif.
    strategies_with_weights = build_enabled_strategies(config.strategies)
    allocator = SignalAllocator(strategies_with_weights, allow_short=config.risk.allow_short)
    target_exposures = allocator.latest_target_exposures(data_by_symbol)

    regime_config = config.market.regime_filter
    if regime_config.enabled:
        bench_df = data_by_symbol.get(regime_config.symbol)
        if bench_df is None:
            logger.warning(
                "Filtre de régime activé pour %s mais ce symbole n'est pas dans l'univers configuré "
                "(config.yaml -> universe.symbols) : filtre ignoré ce cycle.",
                regime_config.symbol,
            )
        else:
            scale = latest_regime_scale(bench_df["close"], regime_config.sma_window, regime_config.bearish_exposure_scale)
            if scale < 1.0:
                logger.warning(
                    "Filtre de régime : %s en tendance baissière, expositions réduites d'un facteur %.2f.",
                    regime_config.symbol,
                    scale,
                )
            target_exposures = {sym: exp * scale for sym, exp in target_exposures.items()}

    logger.info("Expositions cibles : %s", {k: round(v, 3) for k, v in target_exposures.items()})

    # 3) Dimensionnement du risque -> coupe-circuit journalier -> ordres.
    risk_manager = RiskManager(config.risk)
    sizings = risk_manager.size_positions(target_exposures, data_by_symbol)

    current_weights = {
        sym: (current_qty.get(sym, 0.0) * last_prices[sym] / equity if last_prices.get(sym) and equity > 0 else 0.0)
        for sym in config.symbols
    }
    sizings = apply_halt(sizings, current_weights, risk_state)

    orders = plan_orders(sizings, current_qty, equity, last_prices)

    # Annule le stop natif des symboles dont la position va changer AVANT
    # d'envoyer l'ordre de rebalancement : le broker réserve les actions
    # couvertes par un ordre stop ouvert et refuserait sinon un ordre
    # concurrent sur la même position.
    for order in orders:
        _cancel_stop_order(order.symbol, state, broker, dry_run)

    if orders:
        execute_orders(orders, broker, dry_run=dry_run)
        if not dry_run:
            # Relit les positions réelles après exécution : le stop natif doit
            # porter sur la quantité effectivement détenue, pas une estimation
            # (voir limite connue en tête de fichier).
            current_qty = {symbol: pos.qty for symbol, pos in broker.get_positions().items()}
    else:
        logger.info("Aucun ordre à passer ce cycle (portefeuille déjà à la cible).")

    # 4) Pose ou renouvelle le stop suiveur ATR (ordre natif) de chaque
    # position visée, pour qu'il reste actif en continu jusqu'au prochain cycle.
    for symbol in config.symbols:
        target_weight = sizings[symbol].target_weight if symbol in sizings else 0.0
        if abs(target_weight) <= 1e-9:
            state.trailing_stops.pop(symbol, None)
            _cancel_stop_order(symbol, state, broker, dry_run)
            continue

        df = data_by_symbol.get(symbol)
        price = last_prices.get(symbol)
        if df is None or price is None or len(df) <= config.risk.atr_window:
            continue

        atr_value = float(atr(df, config.risk.atr_window).iloc[-1])
        direction = 1 if target_weight > 0 else -1
        previous_stop = state.trailing_stops.get(symbol)
        new_stop = update_stop(previous_stop, direction, price, atr_value, config.risk.atr_stop_multiple)
        if new_stop is None:
            continue
        state.trailing_stops[symbol] = new_stop

        # Quantité réellement détenue après le rebalancement de ce cycle. En
        # dry-run (aucun ordre réel), on estime la quantité visée pour quand
        # même journaliser un stop plausible.
        qty = current_qty.get(symbol, 0.0)
        if qty == 0 and price:
            qty = (target_weight * equity) / price
        if qty == 0:
            continue

        stop_moved = previous_stop is None or previous_stop.stop_price != new_stop.stop_price
        if stop_moved or symbol not in state.stop_order_ids:
            _cancel_stop_order(symbol, state, broker, dry_run)
            _submit_stop_order(symbol, qty, new_stop, state, broker, dry_run)

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

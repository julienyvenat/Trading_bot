"""Moteur live : boucle de décision tournant contre le compte paper Alpaca."""

from __future__ import annotations

import time

from trading_bot.config import AppConfig, load_alpaca_credentials
from trading_bot.data.market_data import fetch_latest_bars
from trading_bot.execution.alpaca_broker import AlpacaBroker
from trading_bot.execution.rebalancer import execute_orders, plan_orders
from trading_bot.logger import get_logger
from trading_bot.portfolio.allocator import SignalAllocator
from trading_bot.portfolio.risk import RiskManager
from trading_bot.strategies.registry import build_enabled_strategies

logger = get_logger()


def run_once(config: AppConfig, broker: AlpacaBroker, dry_run: bool) -> None:
    """Exécute un cycle complet : données -> signaux -> allocation -> risque -> ordres."""
    credentials = load_alpaca_credentials()

    logger.info("Récupération des données de marché pour %s...", ", ".join(config.symbols))
    data_by_symbol = fetch_latest_bars(config.symbols, config.timeframe, credentials)
    if not data_by_symbol:
        logger.warning("Aucune donnée de marché reçue, cycle ignoré.")
        return

    strategies_with_weights = build_enabled_strategies(config.strategies)
    allocator = SignalAllocator(strategies_with_weights, allow_short=config.risk.allow_short)
    target_exposures = allocator.latest_target_exposures(data_by_symbol)
    logger.info("Expositions cibles : %s", {k: round(v, 3) for k, v in target_exposures.items()})

    risk_manager = RiskManager(config.risk)
    sizings = risk_manager.size_positions(target_exposures, data_by_symbol)

    account = broker.get_account()
    positions = broker.get_positions()
    current_qty = {symbol: pos.qty for symbol, pos in positions.items()}
    last_prices = {symbol: broker.get_last_price(symbol) for symbol in config.symbols}

    orders = plan_orders(sizings, current_qty, account.equity, last_prices)
    if not orders:
        logger.info("Aucun ordre à passer ce cycle (portefeuille déjà à la cible).")
        return

    execute_orders(orders, broker, dry_run=dry_run)


def run_forever(config: AppConfig, dry_run: bool = False) -> None:
    """Boucle infinie : exécute un cycle toutes les `loop_interval_seconds`,
    en ne tradant que pendant les heures de marché si configuré ainsi.
    """
    credentials = load_alpaca_credentials()
    broker = AlpacaBroker(credentials)

    mode = "PAPER" if credentials.paper else "RÉEL"
    logger.info("Démarrage du moteur live en mode %s (dry_run=%s).", mode, dry_run)
    if not credentials.paper and not dry_run:
        logger.warning("!!! Compte RÉEL détecté : des ordres avec de l'argent réel vont être passés !!!")

    while True:
        try:
            if config.live.trade_only_when_market_open and not broker.is_market_open():
                logger.info("Marché fermé, cycle ignoré.")
            else:
                run_once(config, broker, dry_run=dry_run)
        except Exception:  # noqa: BLE001 - on ne veut jamais crasher la boucle live
            logger.exception("Erreur pendant le cycle de trading, on continue.")

        time.sleep(config.live.loop_interval_seconds)

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

Cette même course peut aussi faire rejeter transitoirement la pose du stop
par Alpaca ("potential wash trade detected" juste après le fill d'un ordre de
rebalancement, le temps que son carnet d'ordres interne se mette à jour) ;
`_submit_stop_order` retente donc automatiquement quelques fois avec un court
délai avant d'abandonner *ce symbole précis* : un rejet ne doit jamais faire
échouer le cycle entier ni empêcher la pose des stops des autres positions.

Autre contrainte Alpaca, structurelle celle-ci (vue en pratique lors du
premier test en paper trading réel) : les ordres stop/stop_limit en quantité
FRACTIONNAIRE ne sont acceptés qu'en `TimeInForce.DAY`, jamais `GTC` — or le
dimensionnement basé sur l'ATR (`RiskManager`) produit quasi systématiquement
des quantités fractionnaires. Chaque stop natif expire donc à la clôture de
la séance où il a été posé (voir `AlpacaBroker.submit_stop_order`), et
`state.stop_order_dates` (voir `trading_bot.state`) permet à `run_once` de le
reposer à chaque nouvelle séance même quand son prix n'a pas bougé, plutôt
que de laisser une position sans protection dès le lendemain.
"""

from __future__ import annotations

import time

import pandas as pd

from trading_bot.config import AppConfig, load_alpaca_credentials
from trading_bot.data.market_data import fetch_latest_bars
from trading_bot.data.news_sentiment import fetch_recent_sentiment
from trading_bot.execution.broker_base import Broker
from trading_bot.execution.rebalancer import execute_orders, plan_orders
from trading_bot.indicators import atr
from trading_bot.live.trade_realization import detect_realized_trades, snapshot_positions
from trading_bot.logger import get_logger
from trading_bot.market_calendar import MarketCalendar
from trading_bot.portfolio.allocator import SignalAllocator
from trading_bot.portfolio.circuit_breaker import CircuitBreaker, RiskState, apply_halt, should_flatten
from trading_bot.portfolio.regime import latest_regime_scale
from trading_bot.portfolio.risk import RiskManager
from trading_bot.portfolio.stops import StopLevel, update_stop
from trading_bot.portfolio.symbol_track_record import load_track_record, merge_trades, save_track_record
from trading_bot.portfolio.volatility_filter import latest_volatility_scale
from trading_bot.state import LiveState, load_state, save_state
from trading_bot.strategies.registry import build_enabled_strategies

logger = get_logger()

# Ne dort jamais plus de 30 minutes d'un coup en dehors des heures de marché,
# pour rester réactif (logs réguliers, arrêt propre du process, etc.).
MAX_SLEEP_CHUNK_SECONDS = 1800


def _update_track_record(config: AppConfig, trades: list) -> None:
    """Alimente la base persistante de performance par symbole (voir
    `trading_bot.portfolio.symbol_track_record`) avec les trades RÉALISÉS ce
    cycle — c'est la façon dont l'algorithme "apprend" dans le temps sur des
    données réellement nouvelles (contrairement à un backtest rejoué, voir
    la docstring du module). No-op si `trades` est vide (pas d'I/O inutile)."""
    if not trades:
        return
    path = config.live.track_record_file
    updated = merge_trades(load_track_record(path), trades)
    save_track_record(path, updated)
    logger.info("Base de suivi par symbole mise à jour (%s) : %d trade(s) réalisé(s) ce cycle.", path, len(trades))


def _cancel_stop_order(symbol: str, state: LiveState, broker: Broker, dry_run: bool) -> None:
    """Annule (best-effort) le stop natif existant d'un symbole, s'il y en a un."""
    order_id = state.stop_order_ids.pop(symbol, None)
    state.stop_order_dates.pop(symbol, None)
    if order_id is None:
        return
    if dry_run:
        logger.info("DRY-RUN CANCEL STOP %s (ordre %s)", symbol, order_id)
        return
    broker.cancel_order(order_id)


# Nombre de tentatives et délai avant retry en cas de rejet transitoire par
# Alpaca ("wash trade" faussement détecté juste après le fill d'un ordre
# marché de rebalancement, le temps que le carnet d'ordres se mette à jour
# côté broker — voir limite connue en tête de fichier).
_STOP_ORDER_MAX_ATTEMPTS = 3
_STOP_ORDER_RETRY_DELAY_SECONDS = 2.0


def _submit_stop_order(
    symbol: str, qty: float, stop: StopLevel, state: LiveState, broker: Broker, dry_run: bool, today_str: str
) -> None:
    side = "sell" if stop.direction > 0 else "buy"
    if dry_run:
        logger.info("DRY-RUN STOP %s %s %.4f @ %.2f", side.upper(), symbol, abs(qty), stop.stop_price)
        return

    last_error: Exception | None = None
    for attempt in range(1, _STOP_ORDER_MAX_ATTEMPTS + 1):
        try:
            order_id = broker.submit_stop_order(symbol, abs(qty), side, stop.stop_price)
            state.stop_order_ids[symbol] = order_id
            state.stop_order_dates[symbol] = today_str
            return
        except Exception as exc:  # noqa: BLE001 - on catégorise via retry, pas via type
            last_error = exc
            if attempt < _STOP_ORDER_MAX_ATTEMPTS:
                logger.warning(
                    "Échec de la pose du stop sur %s (tentative %d/%d), nouvelle tentative dans %.0fs : %s",
                    symbol,
                    attempt,
                    _STOP_ORDER_MAX_ATTEMPTS,
                    _STOP_ORDER_RETRY_DELAY_SECONDS,
                    exc,
                )
                time.sleep(_STOP_ORDER_RETRY_DELAY_SECONDS)

    # Toutes les tentatives ont échoué : on logue et on abandonne CE symbole
    # sans interrompre le cycle (les autres positions doivent quand même
    # récupérer leur stop). La position reste sans stop natif jusqu'au
    # prochain cycle, qui retentera (stop_order_ids ne contient pas ce
    # symbole, donc `stop_moved or symbol not in state.stop_order_ids` sera
    # vrai et redéclenchera une tentative).
    logger.error(
        "Impossible de poser le stop natif sur %s après %d tentatives, position non protégée "
        "jusqu'au prochain cycle : %s",
        symbol,
        _STOP_ORDER_MAX_ATTEMPTS,
        last_error,
    )


def _fetch_live_bars(config: AppConfig, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Récupère les dernières bougies OHLCV pour un cycle live, via Alpaca ou
    yfinance selon `config.live.broker` : Alpaca ne couvrant pas les actions
    européennes, un broker "manual" (ex: PEA sans API, voir
    `trading_bot.execution.manual_broker`) doit s'appuyer sur yfinance."""
    if config.live.broker == "manual":
        from trading_bot.data.historical import fetch_latest_data, yfinance_interval_for_timeframe

        return fetch_latest_data(symbols, yfinance_interval_for_timeframe(config.timeframe))

    return fetch_latest_bars(symbols, config.timeframe, load_alpaca_credentials())


def build_broker(config: AppConfig) -> Broker:
    """Instancie le broker configuré par `config.live.broker` : "alpaca"
    (défaut, inchangé) ou "manual" (voir `trading_bot.execution.manual_broker`,
    pour un compte sans API de courtage comme un PEA)."""
    if config.live.broker == "manual":
        from trading_bot.execution.manual_broker import ManualBroker

        return ManualBroker(config.live.manual.account_file, calendar_name=config.market.calendar)
    if config.live.broker != "alpaca":
        raise ValueError(f"`live.broker` inconnu '{config.live.broker}'. Valeurs supportées : 'alpaca', 'manual'.")

    from trading_bot.execution.alpaca_broker import AlpacaBroker

    return AlpacaBroker(load_alpaca_credentials())


def _fetch_filter_reference(
    symbol: str,
    data_by_symbol: dict[str, pd.DataFrame],
    config: AppConfig,
) -> pd.DataFrame | None:
    """Renvoie les bougies d'un symbole de référence utilisé par un filtre
    (régime ou volatilité), en le récupérant séparément s'il n'est pas déjà
    dans l'univers tradé (`config.symbols`) — ce symbole de référence n'a pas
    besoin d'y figurer, voir `RegimeFilterConfig`/`VolatilityFilterConfig`."""
    if symbol in data_by_symbol:
        return data_by_symbol[symbol]
    extra = _fetch_live_bars(config, [symbol])
    return extra.get(symbol)


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
    logger.info("Récupération des données de marché pour %s...", ", ".join(config.symbols))
    data_by_symbol = _fetch_live_bars(config, config.symbols)
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

    # Trades RÉALISÉS depuis la fin du cycle précédent (ex: un stop natif
    # déclenché entre deux cycles, voir trading_bot.live.trade_realization),
    # détectés en comparant le dernier instantané persisté aux positions
    # fraîchement lues ci-dessus. Alimente `trading_bot.portfolio.
    # symbol_track_record` en fin de cycle (voir plus bas), sur TOUS les
    # chemins de sortie de cette fonction (y compris le flatten sur
    # coupe-circuit ci-dessous) pour ne jamais perdre un trade réalisé.
    now_ts = pd.Timestamp.now(tz="UTC")
    positions_before_this_cycle_orders = positions
    realized_trades = detect_realized_trades(state.last_known_positions, positions, last_prices, now_ts)

    # Le stop suiveur est désormais un ordre natif posé chez le broker (voir
    # plus bas) : s'il n'y a plus de position pour un symbole qu'on suivait,
    # c'est que son stop a fillé (ou que la position a été close autrement)
    # depuis le dernier cycle. On nettoie l'état en conséquence.
    for symbol in list(state.stop_order_ids):
        if current_qty.get(symbol, 0.0) == 0.0:
            logger.info("Stop natif sur %s considéré exécuté (plus de position) : nettoyage de l'état.", symbol)
            state.stop_order_ids.pop(symbol, None)
            state.stop_order_dates.pop(symbol, None)
            state.trailing_stops.pop(symbol, None)

    today = pd.Timestamp.now(tz="UTC").date()
    today_str = today.isoformat()
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
        # `state.last_known_positions` n'est PAS mis à jour ici (volontaire) :
        # le flatten qui vient d'être déclenché sera détecté comme réalisé au
        # PROCHAIN cycle, par comparaison avec cet instantané resté inchangé
        # (positions d'avant flatten) — plus simple que de re-vérifier ici
        # que chaque ordre de flatten a bien fillé avant de l'enregistrer.
        _update_track_record(config, realized_trades)
        return state

    # 2) Signaux multi-stratégies -> exposition cible, modulée par le filtre
    # de régime de marché s'il est actif.
    strategies_with_weights = build_enabled_strategies(config.strategies)
    allocator = SignalAllocator(strategies_with_weights, allow_short=config.risk.allow_short)
    target_exposures = allocator.latest_target_exposures(data_by_symbol)

    regime_config = config.market.regime_filter
    if regime_config.enabled:
        bench_df = _fetch_filter_reference(regime_config.symbol, data_by_symbol, config)
        if bench_df is None:
            logger.warning(
                "Filtre de régime activé pour %s mais aucune donnée disponible pour ce symbole : "
                "filtre ignoré ce cycle.",
                regime_config.symbol,
            )
        else:
            scale = latest_regime_scale(bench_df["close"], regime_config.sma_window, regime_config.bearish_exposure_scale)
            if scale < 1.0:
                logger.warning(
                    "Filtre de régime : %s en tendance baissière, expositions réduites d'un facteur %.2f "
                    "(hors symboles exemptés : %s).",
                    regime_config.symbol,
                    scale,
                    ", ".join(regime_config.exempt_symbols) or "aucun",
                )
            target_exposures = {
                sym: (exp if sym in regime_config.exempt_symbols else exp * scale)
                for sym, exp in target_exposures.items()
            }

    # Filtre de volatilité : complémentaire au filtre de régime ci-dessus
    # (celui-ci réagit à l'amplitude des mouvements récents plutôt qu'à la
    # direction du marché, voir `trading_bot.portfolio.volatility_filter`).
    volatility_config = config.market.volatility_filter
    if volatility_config.enabled:
        vol_df = _fetch_filter_reference(volatility_config.symbol, data_by_symbol, config)
        if vol_df is None:
            logger.warning(
                "Filtre de volatilité activé pour %s mais aucune donnée disponible pour ce symbole : "
                "filtre ignoré ce cycle.",
                volatility_config.symbol,
            )
        else:
            vol_scale = latest_volatility_scale(
                vol_df["close"],
                volatility_config.sma_window,
                volatility_config.spike_threshold_pct,
                volatility_config.spike_exposure_scale,
            )
            if vol_scale < 1.0:
                logger.warning(
                    "Filtre de volatilité : pic détecté sur %s, expositions réduites d'un facteur %.2f "
                    "(hors symboles exemptés : %s).",
                    volatility_config.symbol,
                    vol_scale,
                    ", ".join(volatility_config.exempt_symbols) or "aucun",
                )
            target_exposures = {
                sym: (exp if sym in volatility_config.exempt_symbols else exp * vol_scale)
                for sym, exp in target_exposures.items()
            }

    logger.info("Expositions cibles : %s", {k: round(v, 3) for k, v in target_exposures.items()})

    # 3) Dimensionnement du risque -> coupe-circuit journalier -> ordres.
    risk_manager = RiskManager(config.risk)
    sizings = risk_manager.size_positions(target_exposures, data_by_symbol)

    current_weights = {
        sym: (current_qty.get(sym, 0.0) * last_prices[sym] / equity if last_prices.get(sym) and equity > 0 else 0.0)
        for sym in config.symbols
    }
    sizings = apply_halt(sizings, current_weights, risk_state)

    # 3bis) Filtre de sentiment de news : bloque uniquement les NOUVELLES
    # entrées (symbole actuellement flat) dont les news récentes sont
    # majoritairement négatives — ne s'applique jamais à une position déjà
    # ouverte (jamais de blocage d'une réduction de risque). Voir
    # `trading_bot.data.news_sentiment` pour la justification de l'approche
    # (API officielle Alpaca, pas de scraping) : indisponible en mode
    # `live.broker: "manual"` (pas d'identifiants Alpaca dans ce mode).
    news_config = config.news_sentiment
    if news_config.enabled and config.live.broker == "manual":
        logger.warning(
            "Filtre de sentiment de news activé mais indisponible en mode `live.broker: manual` "
            "(API News Alpaca uniquement) : filtre ignoré ce cycle."
        )
    elif news_config.enabled:
        credentials = load_alpaca_credentials()
        for symbol in list(sizings):
            sizing = sizings[symbol]
            is_new_entry = current_qty.get(symbol, 0.0) == 0.0 and abs(sizing.target_weight) > 1e-9
            if not is_new_entry:
                continue
            try:
                sentiment = fetch_recent_sentiment(symbol, credentials, news_config.lookback_hours)
            except Exception:  # noqa: BLE001 - une panne de l'API news ne doit jamais bloquer tout le cycle
                logger.exception(
                    "Impossible de récupérer le sentiment de news pour %s : entrée autorisée par défaut.", symbol
                )
                continue
            if sentiment.num_articles >= news_config.min_articles and sentiment.score <= news_config.block_threshold:
                logger.warning(
                    "Filtre de sentiment : nouvelle entrée sur %s bloquée (score %.2f sur %d articles récents).",
                    symbol,
                    sentiment.score,
                    sentiment.num_articles,
                )
                sizings.pop(symbol, None)

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
            # (voir limite connue en tête de fichier). Capture aussi les
            # trades réalisés par CE rebalancement (réduction/clôture), voir
            # trading_bot.live.trade_realization.
            positions = broker.get_positions()
            current_qty = {symbol: pos.qty for symbol, pos in positions.items()}
            realized_trades += detect_realized_trades(
                snapshot_positions(positions_before_this_cycle_orders), positions, last_prices, now_ts
            )
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
        # même journaliser un stop plausible. En réel, une quantité nulle ici
        # signifie que l'ordre de rebalancement n'a pas (encore) été reflété
        # côté broker — voire a été rejeté après plusieurs tentatives, voir
        # `execution.rebalancer._submit_market_order_with_retry` — et il ne
        # faut surtout pas fabriquer un stop sur une position qui n'existe
        # pas réellement : le cycle suivant reposera le stop dès que la
        # position sera confirmée.
        qty = current_qty.get(symbol, 0.0)
        if qty == 0 and price and dry_run:
            qty = (target_weight * equity) / price
        if qty == 0:
            continue

        stop_moved = previous_stop is None or previous_stop.stop_price != new_stop.stop_price
        # Un ordre stop natif posé un jour de séance antérieur est déjà expiré
        # côté broker (TimeInForce.DAY, voir AlpacaBroker.submit_stop_order) :
        # il faut le reposer même si son prix n'a pas bougé depuis.
        stop_expired = state.stop_order_dates.get(symbol) != today_str
        if stop_moved or symbol not in state.stop_order_ids or stop_expired:
            _cancel_stop_order(symbol, state, broker, dry_run)
            _submit_stop_order(symbol, qty, new_stop, state, broker, dry_run, today_str)

    state.last_known_positions = snapshot_positions(positions)
    _update_track_record(config, realized_trades)

    return state


def run_forever(config: AppConfig, dry_run: bool = False) -> None:
    """Boucle infinie : exécute un cycle à chaque instant actionnable (marché
    ouvert, hors buffer de clôture), et dort intelligemment le reste du temps.
    """
    broker = build_broker(config)
    calendar = MarketCalendar(config.market.calendar)

    state = load_state(config.live.state_file)
    if state.risk_state and state.risk_state.drawdown_halted:
        logger.error(
            "Coupe-circuit de DRAWDOWN restauré depuis %s : le bot reste À L'ARRÊT tant qu'il "
            "n'est pas levé manuellement (voir README, section 'Reprise après coupe-circuit').",
            config.live.state_file,
        )

    if config.live.broker == "manual":
        logger.info(
            "Démarrage du moteur live en mode MANUEL (dry_run=%s) : chaque ordre/stop sera affiché à "
            "exécuter toi-même sur ton courtier, jamais envoyé automatiquement.",
            dry_run,
        )
    else:
        credentials = load_alpaca_credentials()
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

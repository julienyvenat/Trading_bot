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

Mode PEA manuel (`live.broker: "manual"`) — ajouts :
  - `live.manual.native_trailing_stop` : le stop est un "Stop Suiveur" natif
    du courtier (écart en % figé à l'entrée, remonté en continu par le
    courtier) : le bot ne notifie qu'à la pose, à l'annulation, et quand le
    plus bas d'une bougie a vraisemblablement touché le seuil (il comptabilise
    alors la vente et prévient de vérifier) — plus jamais "remplace ton stop"
    à chaque cycle. `risk.stop_mode: none` : aucun stop, message explicite ;
  - mêmes garde-fous d'exécution que le backtest (frais par paliers, actions
    entières, cash, ordres trop petits, voir `trading_bot.portfolio.fees`) ;
  - `risk.stop_reentry: sma` : pas de ré-entrée après stop tant que la
    clôture n'est pas repassée au-dessus de sa SMA ;
  - `live.alerts` : alertes d'information (clôture sous SMA, drawdown du
    compte), notifiées au franchissement uniquement ;
  - `live.daily_run_after` : un seul cycle par jour de bourse, après la
    clôture (voir `seconds_until_daily_run`).
"""

from __future__ import annotations

import time

import pandas as pd

from trading_bot.config import AppConfig, load_alpaca_credentials
from trading_bot.data.market_data import fetch_latest_bars
from trading_bot.data.news_sentiment import fetch_recent_sentiment
from trading_bot.execution.broker_base import Broker
from trading_bot.execution.rebalancer import apply_execution_rules, execute_orders, plan_orders
from trading_bot.indicators import atr, sma
from trading_bot.live.trade_realization import detect_realized_trades, snapshot_positions
from trading_bot.logger import get_logger
from trading_bot.market_calendar import MarketCalendar
from trading_bot.notify.pushover import PushoverNotifier, truncate_lines
from trading_bot.portfolio.allocator import SignalAllocator
from trading_bot.portfolio.circuit_breaker import CircuitBreaker, RiskState, apply_halt, should_flatten
from trading_bot.portfolio.fees import CommissionModel, ExecutionRules
from trading_bot.portfolio.regime import latest_regime_scale
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

        return ManualBroker(
            config.live.manual.account_file,
            calendar_name=config.market.calendar,
            commission=CommissionModel.from_backtest_config(config.backtest),
            limit_offset_pct=config.live.manual.limit_offset_pct,
        )
    if config.live.broker != "alpaca":
        raise ValueError(f"`live.broker` inconnu '{config.live.broker}'. Valeurs supportées : 'alpaca', 'manual'.")

    from trading_bot.execution.alpaca_broker import AlpacaBroker

    return AlpacaBroker(load_alpaca_credentials())


def build_notifier(config: AppConfig) -> PushoverNotifier:
    """Notifier Pushover configuré par `config.live.notifications.pushover`
    (inactif par défaut, voir `trading_bot.notify.pushover`)."""
    return PushoverNotifier(config.live.notifications.pushover)


def notify_cycle_instructions(notifier: PushoverNotifier, broker: Broker) -> None:
    """Envoie UN SEUL push récapitulant toutes les instructions manuelles
    (ordres, stops, stops à annuler) affichées pendant le cycle par un broker
    manuel (voir `ManualBroker.drain_instructions`). Rien n'est envoyé s'il
    n'y a rien à faire, ni pour un broker automatique (Alpaca)."""
    drain = getattr(broker, "drain_instructions", None)
    if drain is None:
        return
    try:
        instructions = drain()
        if not instructions:
            return
        prefixes = {"order": "ORDRE : ", "stop": "STOP : ", "cancel": "", "info": "INFO : ", "alert": "ALERTE : "}
        lines = [prefixes.get(i.kind, "") + i.text for i in instructions]
        actions = sum(1 for i in instructions if i.kind in ("order", "stop"))
        cancels = sum(1 for i in instructions if i.kind == "cancel")
        if actions or cancels:
            title = f"{actions or cancels} ordre(s) à passer"
        elif any(i.kind == "alert" for i in instructions):
            title = "Alerte (à vérifier)"
        else:
            title = "Info (aucun ordre)"
        notifier.send(title, truncate_lines(lines))
    except Exception:  # noqa: BLE001 - une notification ne doit jamais casser la boucle live
        logger.warning("Échec de la préparation de la notification des ordres manuels.", exc_info=True)


def notify_risk_transitions(
    notifier: PushoverNotifier, before: RiskState | None, after: RiskState | None
) -> None:
    """Alerte (priorité `alert_priority`) au moment où un coupe-circuit se
    DÉCLENCHE — pas à chaque cycle où il reste actif, pour ne pas spammer."""
    if after is None:
        return
    priority = notifier.alert_priority
    if after.drawdown_halted and not (before and before.drawdown_halted):
        notifier.send(
            "COUPE-CIRCUIT DRAWDOWN",
            "Coupe-circuit de drawdown déclenché : flatten de toutes les positions, plus aucune entrée "
            "tant qu'il n'est pas levé manuellement (voir README).",
            priority=priority,
        )
    elif after.daily_halted and not (before and before.daily_halted):
        notifier.send(
            "Coupe-circuit journalier",
            "Perte journalière max atteinte : aucune nouvelle entrée jusqu'à la prochaine séance.",
            priority=priority,
        )


def notify_cycle_error(notifier: PushoverNotifier, exc: BaseException) -> None:
    notifier.send(
        "Erreur de cycle",
        truncate_lines([f"Le cycle de trading a échoué : {type(exc).__name__}: {exc}", "Voir les logs du bot."]),
        priority=notifier.alert_priority,
    )


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


def _effective_stop_mode(config: AppConfig) -> tuple[str, bool]:
    """(mode de stop effectif, stop suiveur natif ?). Un stop "trailing_pct"
    n'est supporté en live qu'en Stop Suiveur natif d'un broker manuel ; en
    mode natif, un stop "atr" est converti en écart % figé à l'entrée."""
    mode = validate_stop_mode(config.risk.stop_mode)
    native = config.live.broker == "manual" and config.live.manual.native_trailing_stop
    if native and mode == "atr":
        mode = "trailing_pct"
    if mode == "trailing_pct" and not native:
        raise ValueError(
            "`risk.stop_mode: trailing_pct` n'est supporté en live qu'avec `live.broker: manual` et "
            "`live.manual.native_trailing_stop: true` (Stop Suiveur natif du courtier)."
        )
    if config.risk.stop_reentry not in ("immediate", "sma"):
        raise ValueError("`risk.stop_reentry` doit valoir 'immediate' ou 'sma'.")
    return mode, native


def _note(broker: Broker, text: str, kind: str = "info") -> None:
    add_note = getattr(broker, "add_note", None)
    if add_note is None:
        logger.warning(text)
    else:
        add_note(text, kind=kind)


def _check_native_stops(
    state: LiveState, current_qty: dict[str, float], data_by_symbol: dict[str, pd.DataFrame], broker: Broker
) -> bool:
    """Rejoue, bougie par bougie depuis la dernière examinée, chaque Stop
    Suiveur natif posé chez le courtier : remonte son plus haut de référence
    (comme le courtier) et détecte un déclenchement probable (plus bas <=
    seuil). Renvoie True si au moins une sortie a été comptabilisée."""
    any_exit = False
    for symbol, info in list(state.native_stops.items()):
        qty = current_qty.get(symbol, 0.0)
        if qty <= 0:
            logger.info("Stop suiveur natif sur %s : plus de position, suivi arrêté.", symbol)
            state.native_stops.pop(symbol, None)
            continue
        df = data_by_symbol.get(symbol)
        if df is None or df.empty:
            continue
        stop = PctTrailingStop(trail_pct=float(info["trail_pct"]), high_water=float(info["high_water"]))
        last_date = pd.Timestamp(info["last_date"]) if info.get("last_date") else None
        new_bars = df[df.index > last_date] if last_date is not None else df.iloc[-1:]
        for dt, bar in new_bars.iterrows():
            if float(bar["low"]) <= stop.stop_price:
                exit_price = pct_stop_exit_price(stop, float(bar["open"]))
                broker.record_stop_exit(symbol, qty, exit_price, stop.stop_price, float(bar["low"]))
                state.native_stops.pop(symbol, None)
                state.stopped_out[symbol] = pd.Timestamp(dt).date().isoformat()
                any_exit = True
                break
            stop = stop.ratchet(float(bar["high"]))
            info["high_water"] = stop.high_water
            info["last_date"] = pd.Timestamp(dt).date().isoformat()
    return any_exit


def _apply_reentry_block(
    config: AppConfig, state: LiveState, target_exposures: dict[str, float], data_by_symbol: dict[str, pd.DataFrame]
) -> dict[str, float]:
    """`risk.stop_reentry: sma` : exposition forcée à 0 sur un symbole sorti
    sur stop tant que sa clôture n'est pas repassée au-dessus de sa SMA."""
    if config.risk.stop_reentry != "sma":
        state.stopped_out.clear()
        return target_exposures
    adjusted = dict(target_exposures)
    for symbol in list(state.stopped_out):
        df = data_by_symbol.get(symbol)
        if df is None or len(df) < config.risk.stop_reentry_sma_window:
            adjusted[symbol] = 0.0
            continue
        close = float(df["close"].iloc[-1])
        sma_value = float(sma(df["close"], config.risk.stop_reentry_sma_window).iloc[-1])
        if close > sma_value:
            logger.info("%s repassé au-dessus de sa SMA%d : ré-entrée autorisée.", symbol, config.risk.stop_reentry_sma_window)
            state.stopped_out.pop(symbol, None)
        else:
            adjusted[symbol] = 0.0
    return adjusted


def check_alerts(
    config: AppConfig, state: LiveState, data_by_symbol: dict[str, pd.DataFrame], equity: float, broker: Broker
) -> None:
    """Alertes d'INFORMATION (`live.alerts`), jamais des ordres : clôture de
    `symbol` sous sa SMA, drawdown du compte depuis son plus haut. Notifiées
    uniquement au franchissement (et au retour), pas à chaque cycle."""
    alerts = config.live.alerts
    if not alerts.enabled:
        return
    ticker = alerts.symbol.split(".")[0]
    df = data_by_symbol.get(alerts.symbol)
    if df is not None and len(df) >= alerts.sma_window:
        close = float(df["close"].iloc[-1])
        sma_value = float(sma(df["close"], alerts.sma_window).iloc[-1])
        key = f"below_sma:{alerts.symbol}"
        below = close < sma_value
        if below and not state.alert_flags.get(key):
            _note(
                broker,
                f"{ticker} a clôturé à {close:.2f} €, SOUS sa SMA{alerts.sma_window} ({sma_value:.2f} €). "
                "Information seulement, aucun ordre à passer.",
                kind="alert",
            )
        elif not below and state.alert_flags.get(key):
            _note(broker, f"{ticker} est repassé au-dessus de sa SMA{alerts.sma_window} ({close:.2f} € > {sma_value:.2f} €).")
        state.alert_flags[key] = below

    peak = state.risk_state.equity_peak if state.risk_state else equity
    if peak > 0:
        drawdown = 1.0 - equity / peak
        in_drawdown = drawdown >= alerts.drawdown_pct
        if in_drawdown and not state.alert_flags.get("drawdown"):
            _note(
                broker,
                f"Compte à -{drawdown:.1%} de son plus haut ({equity:.2f} € vs {peak:.2f} €), seuil d'alerte "
                f"{alerts.drawdown_pct:.0%}. Information seulement, aucun ordre à passer.",
                kind="alert",
            )
        state.alert_flags["drawdown"] = in_drawdown


def _manage_replaced_stops(
    config: AppConfig,
    state: LiveState,
    sizings: dict,
    current_qty: dict[str, float],
    data_by_symbol: dict[str, pd.DataFrame],
    last_prices: dict[str, float],
    broker: Broker,
    dry_run: bool,
    equity: float,
    today_str: str,
) -> None:
    """Pose ou renouvelle le stop suiveur ATR (ordre stop natif remplacé à
    chaque déplacement) de chaque position visée, pour qu'il reste actif en
    continu jusqu'au prochain cycle (comportement historique)."""
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

        # Le marché a déjà franchi le niveau du stop ratcheté ENTRE deux
        # cycles (ex: aucun stop natif resté posé chez le broker pendant
        # cette fenêtre, suite à un rejet transitoire au cycle précédent) :
        # soumettre un ordre stop natif à ce prix serait systématiquement
        # rejeté par Alpaca ("stop price must be less than current price" /
        # l'équivalent pour un short), la condition de déclenchement étant
        # déjà vraie. Retenter `_submit_stop_order` en boucle ne ferait que
        # répéter cet échec indéfiniment (voir bug XLE du 2026-09-21) en
        # laissant la position sans AUCUNE protection. On flatten donc
        # immédiatement au marché à la place, et on laisse
        # `detect_realized_trades` constater la clôture au PROCHAIN cycle par
        # comparaison avec `state.last_known_positions` (même pattern que le
        # flatten de coupe-circuit plus haut).
        if is_triggered(new_stop, low=price, high=price):
            side = "sell" if direction > 0 else "buy"
            logger.error(
                "Stop suiveur sur %s déjà franchi (prix %.2f, stop %.2f) avant qu'un ordre stop natif "
                "n'ait pu rester posé chez le broker : flatten immédiat au marché plutôt qu'une pose "
                "vouée à l'échec.",
                symbol,
                price,
                new_stop.stop_price,
            )
            _cancel_stop_order(symbol, state, broker, dry_run)
            if dry_run:
                logger.info("DRY-RUN FLATTEN (stop déjà franchi) %s %s %.4f", side.upper(), symbol, abs(qty))
            else:
                broker.submit_market_order(symbol, abs(qty), side)
            state.trailing_stops.pop(symbol, None)
            continue

        stop_moved = previous_stop is None or previous_stop.stop_price != new_stop.stop_price
        # Un ordre stop natif posé un jour de séance antérieur est déjà expiré
        # côté broker (TimeInForce.DAY, voir AlpacaBroker.submit_stop_order) :
        # il faut le reposer même si son prix n'a pas bougé depuis.
        stop_expired = state.stop_order_dates.get(symbol) != today_str
        if stop_moved or symbol not in state.stop_order_ids or stop_expired:
            _cancel_stop_order(symbol, state, broker, dry_run)
            _submit_stop_order(symbol, qty, new_stop, state, broker, dry_run, today_str)



def _place_native_stops(
    config: AppConfig,
    state: LiveState,
    sizings: dict,
    current_qty: dict[str, float],
    data_by_symbol: dict[str, pd.DataFrame],
    last_prices: dict[str, float],
    broker: Broker,
    dry_run: bool,
    bought_this_cycle: set[str] | None = None,
) -> None:
    """Mode Stop Suiveur natif : invite à poser un stop une seule fois par
    position (écart figé en %), puis laisse le courtier le remonter. Rien à
    notifier tant que la position ne change pas."""
    for symbol in config.symbols:
        target_weight = sizings[symbol].target_weight if symbol in sizings else 0.0
        qty = current_qty.get(symbol, 0.0)
        if qty <= 0 or target_weight <= 0:
            continue
        if symbol in state.native_stops:
            continue  # déjà posé chez le courtier : il le remonte tout seul
        df = data_by_symbol.get(symbol)
        price = last_prices.get(symbol)
        if df is None or not price:
            continue
        atr_value = float(atr(df, config.risk.atr_window).iloc[-1]) if len(df) > config.risk.atr_window else None
        trail_pct = entry_trail_pct(config.risk.trailing_stop_pct, price, atr_value, config.risk.atr_stop_multiple)
        if trail_pct is None:
            continue
        stop = PctTrailingStop(trail_pct=trail_pct, high_water=float(price))
        if dry_run:
            logger.info("DRY-RUN STOP SUIVEUR %s %.0f écart %.1f%% seuil %.2f", symbol, qty, trail_pct * 100, stop.stop_price)
            continue
        broker.submit_trailing_stop_order(
            symbol, qty, trail_pct, stop.stop_price, after_buy=symbol in (bought_this_cycle or set())
        )
        state.native_stops[symbol] = {
            "trail_pct": trail_pct,
            "high_water": stop.high_water,
            "qty": qty,
            "last_date": pd.Timestamp(df.index[-1]).date().isoformat(),
        }


def run_once(config: AppConfig, broker: Broker, dry_run: bool, state: LiveState) -> LiveState:
    """Exécute un cycle complet : données -> coupe-circuits -> signaux ->
    allocation -> risque -> ordres -> stops natifs. Renvoie l'état mis à jour."""
    # Broker manuel : les prix yfinance sont mis en cache par cycle, jamais
    # d'un cycle à l'autre (sinon, dans `run_forever`, le broker construit
    # une seule fois garderait indéfiniment les prix du premier cycle).
    clear_cache = getattr(broker, "clear_price_cache", None)
    if clear_cache is not None:
        clear_cache()
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

    stop_mode, native_stops = _effective_stop_mode(config)
    rules = ExecutionRules.from_backtest_config(config.backtest)
    if native_stops and not dry_run and _check_native_stops(state, current_qty, data_by_symbol, broker):
        account = broker.get_account()
        positions = broker.get_positions()
        current_qty = {symbol: pos.qty for symbol, pos in positions.items()}
        equity = account.equity

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

    if state.stopped_out:
        target_exposures = _apply_reentry_block(config, state, target_exposures, data_by_symbol)

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
    orders = apply_execution_rules(orders, current_qty, equity, last_prices, account.cash, rules)
    entered_this_cycle = {o.symbol for o in orders if o.side == "buy" and current_qty.get(o.symbol, 0.0) == 0.0}
    bought_this_cycle = {o.symbol for o in orders if o.side == "buy"}

    # Annule le stop natif des symboles dont la position va changer AVANT
    # d'envoyer l'ordre de rebalancement : le broker réserve les actions
    # couvertes par un ordre stop ouvert et refuserait sinon un ordre
    # concurrent sur la même position.
    for order in orders:
        _cancel_stop_order(order.symbol, state, broker, dry_run)
        if native_stops and order.symbol in state.native_stops and not dry_run:
            broker.cancel_trailing_stop(order.symbol, "AVANT de passer l'ordre ci-dessous : il bloquerait les titres")
            state.native_stops.pop(order.symbol, None)

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

    if stop_mode == "none":
        # Aucun stop : on nettoie d'éventuels stops hérités d'une ancienne
        # config, et on le dit explicitement à chaque nouvelle entrée.
        for symbol in list(state.stop_order_ids):
            _cancel_stop_order(symbol, state, broker, dry_run)
        state.trailing_stops.clear()
        for symbol in sorted(entered_this_cycle):
            if current_qty.get(symbol, 0.0) > 0:
                _note(
                    broker,
                    f"Pas de stop à poser sur {symbol.split('.')[0]} : risk.stop_mode vaut none dans cette config "
                    "(justification dans ses commentaires et dans le README)."
                    + (" Alertes d'information actives (SMA / drawdown)." if config.live.alerts.enabled else ""),
                )
    elif native_stops:
        _place_native_stops(
            config, state, sizings, current_qty, data_by_symbol, last_prices, broker, dry_run, bought_this_cycle
        )
    else:
        _manage_replaced_stops(config, state, sizings, current_qty, data_by_symbol, last_prices, broker, dry_run, equity, today_str)

    check_alerts(config, state, data_by_symbol, equity, broker)

    state.last_known_positions = snapshot_positions(positions)
    _update_track_record(config, realized_trades)

    return state


def seconds_until_daily_run(
    now: pd.Timestamp, calendar: MarketCalendar, run_after: str, last_run_date: str | None, search_days: int = 21
) -> tuple[float, str]:
    """Mode `live.daily_run_after` : renvoie (secondes à attendre, date de
    séance visée au format ISO). 0 seconde = exécuter maintenant le cycle de
    la séance du jour (jour de bourse, heure locale >= `run_after`, pas
    encore exécuté). Sinon, attente jusqu'au prochain jour de bourse à
    `run_after` (heure locale de la place, ex. Europe/Paris)."""
    hour, minute = (int(part) for part in run_after.split(":"))
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    local_now = now.tz_convert(calendar.timezone)
    for offset in range(search_days):
        day = (local_now + pd.Timedelta(days=offset)).normalize()
        day_str = day.date().isoformat()
        if day_str == last_run_date or not calendar.is_trading_day(day.tz_localize(None)):
            continue
        target = day + pd.Timedelta(hours=hour, minutes=minute)
        if offset == 0 and local_now >= target:
            return 0.0, day_str
        if target > local_now:
            return (target - local_now).total_seconds(), day_str
    raise RuntimeError(f"Aucun jour de bourse trouvé dans les {search_days} prochains jours.")


def run_forever(config: AppConfig, dry_run: bool = False) -> None:
    """Boucle infinie : exécute un cycle à chaque instant actionnable (marché
    ouvert, hors buffer de clôture), et dort intelligemment le reste du temps.
    """
    broker = build_broker(config)
    calendar = MarketCalendar(config.market.calendar)
    notifier = build_notifier(config)
    notifier.warn_if_misconfigured()
    # Alerte de crash envoyée seulement à la PREMIÈRE erreur d'une série :
    # une panne persistante (ex: yfinance indisponible) ne doit pas envoyer
    # un push à chaque cycle.
    consecutive_errors = 0

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

    daily_run_after = config.live.daily_run_after
    if daily_run_after:
        logger.info(
            "Mode quotidien : un seul cycle par jour de bourse, après %s (heure de %s).",
            daily_run_after,
            calendar.timezone,
        )

    while True:
        if daily_run_after:
            wait_seconds, session_str = seconds_until_daily_run(
                pd.Timestamp.now(tz="UTC"), calendar, daily_run_after, state.last_daily_run
            )
            if wait_seconds > 0:
                logger.info("Prochain cycle quotidien (%s) dans %.0f min.", session_str, wait_seconds / 60)
                time.sleep(min(wait_seconds, MAX_SLEEP_CHUNK_SECONDS))
                continue
            try:
                risk_before = state.risk_state
                state = run_once(config, broker, dry_run=dry_run, state=state)
                consecutive_errors = 0
                notify_risk_transitions(notifier, risk_before, state.risk_state)
            except Exception as exc:  # noqa: BLE001 - on ne veut jamais crasher la boucle live
                logger.exception("Erreur pendant le cycle quotidien, nouvel essai au prochain réveil.")
                save_state(config.live.state_file, state)  # ne pas perdre ce qui a déjà été affiché/comptabilisé
                consecutive_errors += 1
                if consecutive_errors == 1:
                    notify_cycle_error(notifier, exc)
                notify_cycle_instructions(notifier, broker)
                time.sleep(MAX_SLEEP_CHUNK_SECONDS)
                continue
            state.last_daily_run = session_str
            save_state(config.live.state_file, state)
            notify_cycle_instructions(notifier, broker)
            continue

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

            risk_before = state.risk_state
            state = run_once(config, broker, dry_run=dry_run, state=state)
            save_state(config.live.state_file, state)
            consecutive_errors = 0
            notify_risk_transitions(notifier, risk_before, state.risk_state)
        except Exception as exc:  # noqa: BLE001 - on ne veut jamais crasher la boucle live
            logger.exception("Erreur pendant le cycle de trading, on continue.")
            consecutive_errors += 1
            if consecutive_errors == 1:
                notify_cycle_error(notifier, exc)
        finally:
            # Même si le cycle a crashé en cours de route, les ordres DÉJÀ
            # affichés (et déjà comptabilisés dans `account_file`) doivent
            # être notifiés : sinon ils ne seraient jamais passés.
            notify_cycle_instructions(notifier, broker)

        time.sleep(config.live.loop_interval_seconds)

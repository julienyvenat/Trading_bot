"""Point d'entrée en ligne de commande du bot.

Usage :
    python -m trading_bot backtest [--config config/config.yaml] [--report chemin.html]
    python -m trading_bot walk-forward [--train-days N] [--test-days N] [--step-days N] [--optimize]
    python -m trading_bot optimize [--metric sharpe_ratio] [--max-workers N] [--top N]
    python -m trading_bot paper [--once] [--dry-run]
    python -m trading_bot notify-test [--config ...]
    python -m trading_bot morning-brief [--config ...] [--print]
"""

from __future__ import annotations

import argparse
import sys

from trading_bot.config import AppConfig, load_config
from trading_bot.logger import setup_logging


def _build_param_grids(config: AppConfig) -> list:
    """Convertit `config.optimization.grids` (dict brut issu de config.yaml)
    en `list[ParamGrid]` exploitable par `trading_bot.backtest.optimizer`."""
    from trading_bot.backtest.optimizer import ParamGrid

    return [
        ParamGrid(strategy_name=strategy_name, params=params)
        for strategy_name, params in config.optimization.grids.items()
    ]


def _fetch_data(config: AppConfig, symbols: list[str]) -> dict:
    """Télécharge l'historique pour `symbols`, à la granularité dérivée de
    `universe.timeframe` (même convention que côté live/Alpaca, voir
    `trading_bot.data.market_data`) : '1Day' -> quotidien (comportement
    historique, inchangé), '5Min'/'15Min'/'1Hour'/... -> intraday pour une
    stratégie comme `momentum_breakout` reparamétrée en bougies intraday.

    Source pilotée par `backtest.data_source` (voir `BacktestConfig`) :
    "yfinance" (défaut) ou "alpaca" pour un historique intraday plus profond
    que la limite de yfinance.

    `backtest.warmup_days` > 0 : télécharge aussi ce nombre de jours AVANT
    `start_date` pour chauffer les indicateurs (le moteur de backtest ne
    simule de toute façon qu'à partir de `start_date`)."""
    start_date = config.backtest.start_date
    if config.backtest.warmup_days > 0:
        import pandas as pd

        start_date = (pd.Timestamp(start_date) - pd.Timedelta(days=config.backtest.warmup_days)).strftime("%Y-%m-%d")
    if config.backtest.data_source == "alpaca":
        from trading_bot.config import load_alpaca_credentials
        from trading_bot.data.market_data import fetch_historical_bars

        return fetch_historical_bars(
            symbols,
            start_date=start_date,
            end_date=config.backtest.end_date,
            timeframe=config.timeframe,
            credentials=load_alpaca_credentials(),
        )
    if config.backtest.data_source != "yfinance":
        raise ValueError(
            f"`backtest.data_source` inconnu '{config.backtest.data_source}'. Valeurs supportées : "
            "'yfinance', 'alpaca'."
        )

    from trading_bot.data.historical import fetch_historical_data, yfinance_interval_for_timeframe

    interval = yfinance_interval_for_timeframe(config.timeframe)
    return fetch_historical_data(
        symbols, start_date=start_date, end_date=config.backtest.end_date, interval=interval
    )


def _fetch_filter_benchmarks(config: AppConfig, data_by_symbol: dict, logger) -> tuple:
    """Récupère, pour le backtest et le walk-forward, les données des
    symboles de référence utilisés par les filtres de régime et de
    volatilité quand ils ne sont pas déjà dans l'univers tradé (voir
    `RegimeFilterConfig`/`VolatilityFilterConfig` : ces symboles n'ont pas
    besoin de figurer dans `universe.symbols`, ils ne servent qu'à calculer
    un facteur d'exposition, jamais tradés pour eux-mêmes via ce mécanisme).
    """
    benchmark_df = None
    regime_config = config.market.regime_filter
    if regime_config.enabled and regime_config.symbol not in data_by_symbol:
        bench_data = _fetch_data(config, [regime_config.symbol])
        benchmark_df = bench_data.get(regime_config.symbol)
        if benchmark_df is None:
            logger.warning(
                "Impossible de récupérer les données de %s pour le filtre de régime : filtre ignoré.",
                regime_config.symbol,
            )

    volatility_benchmark_df = None
    volatility_config = config.market.volatility_filter
    if volatility_config.enabled and volatility_config.symbol not in data_by_symbol:
        vol_data = _fetch_data(config, [volatility_config.symbol])
        volatility_benchmark_df = vol_data.get(volatility_config.symbol)
        if volatility_benchmark_df is None:
            logger.warning(
                "Impossible de récupérer les données de %s pour le filtre de volatilité : filtre ignoré.",
                volatility_config.symbol,
            )

    return benchmark_df, volatility_benchmark_df


def _fetch_rotation_candidates(config: AppConfig, data_by_symbol: dict, logger) -> dict:
    """Récupère, en plus de `universe.symbols`, les pools de candidats de
    rotation déclarés sous `strategies: -> universe_rotation.candidates`
    (voir `UniverseRotationConfig`) qui ne sont pas déjà dans `data_by_symbol`.
    Renvoie `data_by_symbol` complété (nouveau dict, ne mute pas l'original).
    Sans stratégie à rotation active, renvoie `data_by_symbol` inchangé."""
    missing = {
        symbol
        for s in config.strategies
        if s.enabled and s.universe_rotation.enabled
        for symbol in s.universe_rotation.candidates
        if symbol not in data_by_symbol
    }
    if not missing:
        return data_by_symbol

    logger.info("Téléchargement des pools de candidats de rotation : %s...", ", ".join(sorted(missing)))
    extra = _fetch_data(config, sorted(missing))
    still_missing = missing - set(extra)
    if still_missing:
        logger.warning(
            "Candidat(s) de rotation introuvable(s), ignoré(s) pour la sélection périodique : %s.",
            ", ".join(sorted(still_missing)),
        )
    return {**data_by_symbol, **extra}


def cmd_backtest(args: argparse.Namespace) -> None:
    from trading_bot.backtest.engine import run_backtest

    logger = setup_logging()
    config = load_config(args.config)

    logger.info(
        "Téléchargement des données historiques (%s -> %s, granularité %s)...",
        config.backtest.start_date,
        config.backtest.end_date or "aujourd'hui",
        config.timeframe,
    )
    data_by_symbol = _fetch_data(config, config.symbols)
    if not data_by_symbol:
        logger.error("Aucune donnée téléchargée. Vérifie les symboles et ta connexion réseau.")
        sys.exit(1)

    benchmark_df, volatility_benchmark_df = _fetch_filter_benchmarks(config, data_by_symbol, logger)
    data_by_symbol = _fetch_rotation_candidates(config, data_by_symbol, logger)

    logger.info("Lancement du backtest sur %d symboles...", len(data_by_symbol))
    result = run_backtest(
        config, data_by_symbol, benchmark_df=benchmark_df, volatility_benchmark_df=volatility_benchmark_df
    )

    print("\n=== Résultats du backtest ===")
    print(result.metrics.summary())
    print(f"Sorties sur stop suiveur : {result.num_stop_exits}")
    years = max(result.metrics.num_trading_days / 252, 1e-9)
    print(f"Ordres exécutés         : {result.num_orders} ({result.num_orders / years:.1f}/an)")
    print(f"Frais payés (total)     : {result.total_fees:.2f}")
    if result.money_weighted_return_pct is not None and result.total_contributed > config.backtest.initial_cash:
        final_value = float(result.equity_curve.iloc[-1])
        print(
            f"Apports : {result.total_contributed:.2f} versés au total, valeur finale {final_value:.2f} "
            f"(x{final_value / result.total_contributed:.2f}) ; TRI (pondéré par l'argent) : "
            f"{result.money_weighted_return_pct:+.2f}%/an. Les métriques ci-dessus portent sur la NAV par "
            "part (hors effet des apports)."
        )
    if result.final_risk_state and result.final_risk_state.drawdown_halted:
        print(
            "⚠️  Coupe-circuit de DRAWDOWN déclenché à un moment du backtest "
            "(position soldée, plus aucune entrée après ce point)."
        )
    if result.final_risk_state and result.final_risk_state.daily_halted:
        print("⚠️  Coupe-circuit de perte JOURNALIÈRE actif sur la dernière séance simulée.")

    if args.output:
        result.equity_curve.to_csv(args.output, header=["equity"])
        logger.info("Courbe d'equity sauvegardée dans %s", args.output)

    if args.report:
        from trading_bot.backtest.report import generate_html_report

        generate_html_report(result, args.report)
        logger.info("Rapport HTML sauvegardé dans %s", args.report)

    if args.update_track_record:
        from trading_bot.portfolio.symbol_track_record import load_track_record, merge_trades, save_track_record

        path = config.live.track_record_file
        before = load_track_record(path)
        trades_before = sum(s.trade_count for s in before.by_symbol.values())
        track_record = merge_trades(before, result.trades)
        trades_after = sum(s.trade_count for s in track_record.by_symbol.values())
        save_track_record(path, track_record)
        logger.info(
            "Base de suivi par symbole mise à jour (%s) : %d nouveau(x) trade(s) ajouté(s) sur %d de ce "
            "backtest (%d déjà connus, filtrés — voir trading_bot.portfolio.symbol_track_record).",
            path,
            trades_after - trades_before,
            len(result.trades),
            len(result.trades) - (trades_after - trades_before),
        )


def cmd_walk_forward(args: argparse.Namespace) -> None:
    from trading_bot.backtest.walk_forward import run_walk_forward

    logger = setup_logging()
    config = load_config(args.config)

    logger.info(
        "Téléchargement des données historiques (%s -> %s, granularité %s)...",
        config.backtest.start_date,
        config.backtest.end_date or "aujourd'hui",
        config.timeframe,
    )
    data_by_symbol = _fetch_data(config, config.symbols)
    if not data_by_symbol:
        logger.error("Aucune donnée téléchargée. Vérifie les symboles et ta connexion réseau.")
        sys.exit(1)

    benchmark_df, volatility_benchmark_df = _fetch_filter_benchmarks(config, data_by_symbol, logger)
    data_by_symbol = _fetch_rotation_candidates(config, data_by_symbol, logger)

    param_grids = None
    if args.optimize:
        param_grids = _build_param_grids(config)
        if not param_grids:
            logger.error(
                "--optimize demandé mais `optimization.grids` est vide dans config.yaml. "
                "Renseigne au moins une grille de paramètres (voir l'exemple commenté dans config.yaml)."
            )
            sys.exit(1)

    step_days = args.step_days if args.step_days is not None else args.test_days
    logger.info(
        "Lancement du walk-forward (entraînement=%dj, test=%dj, pas=%dj%s)...",
        args.train_days,
        args.test_days,
        step_days,
        ", avec ré-optimisation par fenêtre" if param_grids else "",
    )
    try:
        result = run_walk_forward(
            config,
            data_by_symbol,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=step_days,
            benchmark_df=benchmark_df,
            volatility_benchmark_df=volatility_benchmark_df,
            param_grids=param_grids,
            optimization_metric=args.metric or config.optimization.metric,
            max_workers=args.max_workers,
        )
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    print("\n" + result.summary())


def cmd_optimize(args: argparse.Namespace) -> None:
    from trading_bot.backtest.optimizer import optimize

    logger = setup_logging()
    config = load_config(args.config)

    param_grids = _build_param_grids(config)
    if not param_grids:
        logger.error(
            "`optimization.grids` est vide dans config.yaml. Renseigne au moins une grille de "
            "paramètres (voir l'exemple commenté dans config.yaml) avant de lancer `optimize`."
        )
        sys.exit(1)

    logger.info(
        "Téléchargement des données historiques (%s -> %s, granularité %s)...",
        config.backtest.start_date,
        config.backtest.end_date or "aujourd'hui",
        config.timeframe,
    )
    data_by_symbol = _fetch_data(config, config.symbols)
    if not data_by_symbol:
        logger.error("Aucune donnée téléchargée. Vérifie les symboles et ta connexion réseau.")
        sys.exit(1)

    benchmark_df, volatility_benchmark_df = _fetch_filter_benchmarks(config, data_by_symbol, logger)
    data_by_symbol = _fetch_rotation_candidates(config, data_by_symbol, logger)

    metric = args.metric or config.optimization.metric
    logger.info("Lancement de la recherche par grille (métrique optimisée : %s)...", metric)
    try:
        results = optimize(
            config,
            data_by_symbol,
            param_grids,
            metric=metric,
            benchmark_df=benchmark_df,
            volatility_benchmark_df=volatility_benchmark_df,
            max_workers=args.max_workers,
        )
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    print(f"\n=== Top {min(args.top, len(results))} combinaisons (sur {len(results)} testées, triées par {metric}) ===")
    for rank, result in enumerate(results[: args.top], start=1):
        print(f"{rank}. {metric} = {getattr(result.metrics, metric):.3f} | {result.params_by_strategy}")

    if args.output:
        import csv

        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", metric, "params_by_strategy"])
            for rank, result in enumerate(results, start=1):
                writer.writerow([rank, getattr(result.metrics, metric), result.params_by_strategy])
        logger.info("Résultats complets sauvegardés dans %s", args.output)


def cmd_paper(args: argparse.Namespace) -> None:
    from trading_bot.live.engine import (
        build_broker,
        build_notifier,
        notify_cycle_error,
        notify_cycle_instructions,
        notify_risk_transitions,
        run_forever,
        run_once,
    )
    from trading_bot.state import load_state, save_state

    logger = setup_logging()
    config = load_config(args.config)

    rotating_strategies = [s.name for s in config.strategies if s.enabled and s.universe_rotation.enabled]
    if rotating_strategies:
        logger.error(
            "`universe_rotation` est activé pour %s mais n'est pour l'instant supporté qu'en "
            "backtest/optimize/walk-forward (trading_bot.portfolio.allocator), pas en live : "
            "démarrer `paper` tel quel traderait uniquement `universe.symbols` en ignorant la "
            "rotation, ce qui romprait la parité backtest/live. Désactive `universe_rotation` "
            "sur ces stratégies avant de passer en paper trading, ou attends le support live.",
            ", ".join(rotating_strategies),
        )
        sys.exit(1)

    if args.once:
        broker = build_broker(config)
        notifier = build_notifier(config)
        notifier.warn_if_misconfigured()
        state = load_state(config.live.state_file)
        risk_before = state.risk_state
        try:
            state = run_once(config, broker, dry_run=args.dry_run, state=state)
        except Exception as exc:
            # Comme `run_forever`, on ne veut pas perdre l'état déjà mis à
            # jour par le cycle (stops posés, coupe-circuits) juste parce
            # qu'une erreur inattendue survient en fin de cycle : mieux vaut
            # sauvegarder ce qu'on a et faire remonter l'erreur que de tout
            # perdre silencieusement.
            logger.exception("Erreur pendant le cycle de trading.")
            save_state(config.live.state_file, state)
            notify_cycle_error(notifier, exc)
            raise
        finally:
            notify_cycle_instructions(notifier, broker)
        save_state(config.live.state_file, state)
        notify_risk_transitions(notifier, risk_before, state.risk_state)
    else:
        run_forever(config, dry_run=args.dry_run)


def cmd_notify_test(args: argparse.Namespace) -> None:
    """Envoie une notification Pushover de test, pour valider les
    identifiants (`PUSHOVER_APP_TOKEN` / `PUSHOVER_USER_KEY`) avant de
    compter dessus en live. Fonctionne même si `live.notifications.pushover.
    enabled` est encore à false dans la config (pour tester avant d'activer)."""
    from trading_bot.notify.pushover import APP_TOKEN_ENV, USER_KEY_ENV, PushoverNotifier

    logger = setup_logging(log_file=None)
    config = load_config(args.config)
    pushover_config = config.live.notifications.pushover
    notifier = PushoverNotifier(pushover_config)

    if not notifier.has_credentials:
        logger.error("%s et/ou %s absent(s) de l'environnement (ou de .env).", APP_TOKEN_ENV, USER_KEY_ENV)
        sys.exit(1)
    if not pushover_config.enabled:
        logger.warning(
            "live.notifications.pushover.enabled est à false dans cette config : le test est envoyé quand "
            "même, mais le bot n'enverra rien en live tant que ce n'est pas activé."
        )

    if notifier.send("Test", "Notification de test : les identifiants Pushover fonctionnent.", force=True):
        logger.info("Notification de test envoyée : vérifie ton téléphone.")
    else:
        logger.error("Échec de l'envoi de la notification de test (voir l'avertissement ci-dessus).")
        sys.exit(1)


def cmd_morning_brief(args: argparse.Namespace) -> None:
    """Génère UN aperçu du matin à la demande (voir `trading_bot.live.
    morning_brief`) : `--print` l'affiche sans rien envoyer, sinon il est
    envoyé via Pushover (même si `live.morning_brief.enabled` ou
    `live.notifications.pushover.enabled` valent false, pour tester). Lecture
    seule : ni le fichier de compte ni l'état du bot ne sont modifiés (la
    date du dernier aperçu planifié n'est pas touchée)."""
    from trading_bot.live.morning_brief import build_morning_brief, send_morning_brief
    from trading_bot.state import load_state

    logger = setup_logging(log_file=None)
    config = load_config(args.config)
    try:
        brief = build_morning_brief(config, load_state(config.live.state_file))
    except (ValueError, OSError) as exc:
        logger.error("Aperçu du matin impossible : %s", exc)
        sys.exit(1)
    if args.print:
        print(brief.title)
        print(brief.message)
        return
    if send_morning_brief(config, brief, force=True):
        logger.info("Aperçu du matin envoyé (%d caractères) : vérifie ton téléphone.", len(brief.message))
    else:
        logger.error("Échec de l'envoi de l'aperçu du matin (voir l'avertissement ci-dessus).")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Chemin vers config.yaml (par défaut: config/config.yaml)")

    parser = argparse.ArgumentParser(
        prog="trading_bot",
        description="Bot de trading automatique multi-stratégies.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser(
        "backtest", help="Lance un backtest sur données historiques.", parents=[config_parser]
    )
    backtest_parser.add_argument("-o", "--output", default=None, help="Chemin CSV où sauvegarder la courbe d'equity.")
    backtest_parser.add_argument(
        "--report", default=None, help="Chemin du rapport HTML autonome à générer (nécessite l'extra 'report')."
    )
    backtest_parser.add_argument(
        "--update-track-record",
        action="store_true",
        help=(
            "Enregistre les trades de ce backtest dans la base de suivi par symbole "
            "(config.live.track_record_file, voir trading_bot.portfolio.symbol_track_record). "
            "Idempotent sur une période déjà connue, mais rejouer le même backtest en boucle "
            "n'apporte aucune information nouvelle : préfère l'activer sur des fenêtres réellement "
            "nouvelles (ou laisser le live l'alimenter au fil du temps)."
        ),
    )
    backtest_parser.set_defaults(func=cmd_backtest)

    walk_forward_parser = subparsers.add_parser(
        "walk-forward",
        help="Valide la robustesse de la stratégie par fenêtres glissantes entraînement/test.",
        parents=[config_parser],
    )
    walk_forward_parser.add_argument(
        "--train-days", type=int, default=504, help="Taille de la fenêtre d'entraînement en jours calendaires (def: 504, ~2 ans)."
    )
    walk_forward_parser.add_argument(
        "--test-days",
        type=int,
        default=126,
        help="Taille de la fenêtre de test hors échantillon en jours calendaires (def: 126, ~6 mois).",
    )
    walk_forward_parser.add_argument(
        "--step-days",
        type=int,
        default=None,
        help="Avancée entre deux fenêtres en jours (def: = --test-days, fenêtres de test contiguës non chevauchantes).",
    )
    walk_forward_parser.add_argument(
        "--optimize",
        action="store_true",
        help="Ré-optimise les paramètres sur chaque fenêtre d'entraînement (voir config.yaml -> optimization.grids) "
        "avant de valider hors échantillon sur la fenêtre de test correspondante.",
    )
    walk_forward_parser.add_argument(
        "--metric",
        default=None,
        help="Attribut de BacktestMetrics à maximiser lors de --optimize (def: config.yaml -> optimization.metric).",
    )
    walk_forward_parser.add_argument(
        "--max-workers", type=int, default=None, help="Nombre de processus parallèles pour --optimize (def: tous les cœurs)."
    )
    walk_forward_parser.set_defaults(func=cmd_walk_forward)

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Recherche par grille des meilleurs paramètres de stratégie (voir config.yaml -> optimization).",
        parents=[config_parser],
    )
    optimize_parser.add_argument(
        "--metric", default=None, help="Attribut de BacktestMetrics à maximiser (def: config.yaml -> optimization.metric)."
    )
    optimize_parser.add_argument(
        "--max-workers", type=int, default=None, help="Nombre de processus parallèles (def: tous les cœurs)."
    )
    optimize_parser.add_argument("--top", type=int, default=10, help="Nombre de meilleures combinaisons à afficher (def: 10).")
    optimize_parser.add_argument("-o", "--output", default=None, help="Chemin CSV où sauvegarder TOUS les résultats.")
    optimize_parser.set_defaults(func=cmd_optimize)

    paper_parser = subparsers.add_parser(
        "paper",
        help="Lance le trading live (broker choisi via config.yaml -> live.broker : Alpaca ou manuel).",
        parents=[config_parser],
    )
    paper_parser.add_argument("--once", action="store_true", help="N'exécute qu'un seul cycle puis s'arrête.")
    paper_parser.add_argument(
        "--dry-run", action="store_true", help="Calcule les ordres mais ne les envoie pas au broker."
    )
    paper_parser.set_defaults(func=cmd_paper)

    notify_test_parser = subparsers.add_parser(
        "notify-test",
        help="Envoie une notification Pushover de test (valide PUSHOVER_APP_TOKEN / PUSHOVER_USER_KEY).",
        parents=[config_parser],
    )
    notify_test_parser.set_defaults(func=cmd_notify_test)

    morning_brief_parser = subparsers.add_parser(
        "morning-brief",
        help="Génère un aperçu du matin à la demande (Pushover, ou --print pour l'afficher sans l'envoyer).",
        parents=[config_parser],
    )
    morning_brief_parser.add_argument("--print", action="store_true", help="Affiche l'aperçu sans l'envoyer.")
    morning_brief_parser.set_defaults(func=cmd_morning_brief)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

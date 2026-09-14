"""Point d'entrée en ligne de commande du bot.

Usage :
    python -m trading_bot backtest [--config config/config.yaml] [--report chemin.html]
    python -m trading_bot walk-forward [--train-days N] [--test-days N] [--step-days N] [--optimize]
    python -m trading_bot optimize [--metric sharpe_ratio] [--max-workers N] [--top N]
    python -m trading_bot paper [--once] [--dry-run]
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


def _fetch_filter_benchmarks(config: AppConfig, data_by_symbol: dict, logger) -> tuple:
    """Récupère, pour le backtest et le walk-forward, les données des
    symboles de référence utilisés par les filtres de régime et de
    volatilité quand ils ne sont pas déjà dans l'univers tradé (voir
    `RegimeFilterConfig`/`VolatilityFilterConfig` : ces symboles n'ont pas
    besoin de figurer dans `universe.symbols`, ils ne servent qu'à calculer
    un facteur d'exposition, jamais tradés pour eux-mêmes via ce mécanisme).
    """
    from trading_bot.data.historical import fetch_historical_data

    benchmark_df = None
    regime_config = config.market.regime_filter
    if regime_config.enabled and regime_config.symbol not in data_by_symbol:
        bench_data = fetch_historical_data(
            [regime_config.symbol], start_date=config.backtest.start_date, end_date=config.backtest.end_date
        )
        benchmark_df = bench_data.get(regime_config.symbol)
        if benchmark_df is None:
            logger.warning(
                "Impossible de récupérer les données de %s pour le filtre de régime : filtre ignoré.",
                regime_config.symbol,
            )

    volatility_benchmark_df = None
    volatility_config = config.market.volatility_filter
    if volatility_config.enabled and volatility_config.symbol not in data_by_symbol:
        vol_data = fetch_historical_data(
            [volatility_config.symbol], start_date=config.backtest.start_date, end_date=config.backtest.end_date
        )
        volatility_benchmark_df = vol_data.get(volatility_config.symbol)
        if volatility_benchmark_df is None:
            logger.warning(
                "Impossible de récupérer les données de %s pour le filtre de volatilité : filtre ignoré.",
                volatility_config.symbol,
            )

    return benchmark_df, volatility_benchmark_df


def cmd_backtest(args: argparse.Namespace) -> None:
    from trading_bot.backtest.engine import run_backtest
    from trading_bot.data.historical import fetch_historical_data

    logger = setup_logging()
    config = load_config(args.config)

    logger.info("Téléchargement des données historiques (%s -> %s)...", config.backtest.start_date, config.backtest.end_date or "aujourd'hui")
    data_by_symbol = fetch_historical_data(
        config.symbols,
        start_date=config.backtest.start_date,
        end_date=config.backtest.end_date,
    )
    if not data_by_symbol:
        logger.error("Aucune donnée téléchargée. Vérifie les symboles et ta connexion réseau.")
        sys.exit(1)

    benchmark_df, volatility_benchmark_df = _fetch_filter_benchmarks(config, data_by_symbol, logger)

    logger.info("Lancement du backtest sur %d symboles...", len(data_by_symbol))
    result = run_backtest(
        config, data_by_symbol, benchmark_df=benchmark_df, volatility_benchmark_df=volatility_benchmark_df
    )

    print("\n=== Résultats du backtest ===")
    print(result.metrics.summary())
    print(f"Sorties sur stop suiveur : {result.num_stop_exits}")
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


def cmd_walk_forward(args: argparse.Namespace) -> None:
    from trading_bot.backtest.walk_forward import run_walk_forward
    from trading_bot.data.historical import fetch_historical_data

    logger = setup_logging()
    config = load_config(args.config)

    logger.info(
        "Téléchargement des données historiques (%s -> %s)...",
        config.backtest.start_date,
        config.backtest.end_date or "aujourd'hui",
    )
    data_by_symbol = fetch_historical_data(
        config.symbols, start_date=config.backtest.start_date, end_date=config.backtest.end_date
    )
    if not data_by_symbol:
        logger.error("Aucune donnée téléchargée. Vérifie les symboles et ta connexion réseau.")
        sys.exit(1)

    benchmark_df, volatility_benchmark_df = _fetch_filter_benchmarks(config, data_by_symbol, logger)

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
    from trading_bot.data.historical import fetch_historical_data

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
        "Téléchargement des données historiques (%s -> %s)...",
        config.backtest.start_date,
        config.backtest.end_date or "aujourd'hui",
    )
    data_by_symbol = fetch_historical_data(
        config.symbols, start_date=config.backtest.start_date, end_date=config.backtest.end_date
    )
    if not data_by_symbol:
        logger.error("Aucune donnée téléchargée. Vérifie les symboles et ta connexion réseau.")
        sys.exit(1)

    benchmark_df, volatility_benchmark_df = _fetch_filter_benchmarks(config, data_by_symbol, logger)

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
    from trading_bot.execution.alpaca_broker import AlpacaBroker
    from trading_bot.live.engine import run_forever, run_once
    from trading_bot.state import load_state, save_state

    logger = setup_logging()
    config = load_config(args.config)

    if args.once:
        from trading_bot.config import load_alpaca_credentials

        broker = AlpacaBroker(load_alpaca_credentials())
        state = load_state(config.live.state_file)
        try:
            state = run_once(config, broker, dry_run=args.dry_run, state=state)
        except Exception:
            # Comme `run_forever`, on ne veut pas perdre l'état déjà mis à
            # jour par le cycle (stops posés, coupe-circuits) juste parce
            # qu'une erreur inattendue survient en fin de cycle : mieux vaut
            # sauvegarder ce qu'on a et faire remonter l'erreur que de tout
            # perdre silencieusement.
            logger.exception("Erreur pendant le cycle de trading.")
            save_state(config.live.state_file, state)
            raise
        save_state(config.live.state_file, state)
    else:
        run_forever(config, dry_run=args.dry_run)


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
        "paper", help="Lance le trading en paper trading (ou réel) via Alpaca.", parents=[config_parser]
    )
    paper_parser.add_argument("--once", action="store_true", help="N'exécute qu'un seul cycle puis s'arrête.")
    paper_parser.add_argument(
        "--dry-run", action="store_true", help="Calcule les ordres mais ne les envoie pas au broker."
    )
    paper_parser.set_defaults(func=cmd_paper)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

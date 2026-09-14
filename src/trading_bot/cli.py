"""Point d'entrée en ligne de commande du bot.

Usage :
    python -m trading_bot backtest [--config config/config.yaml]
    python -m trading_bot walk-forward [--train-days N] [--test-days N] [--step-days N]
    python -m trading_bot paper [--once] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

from trading_bot.config import AppConfig, load_config
from trading_bot.logger import setup_logging


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

    step_days = args.step_days if args.step_days is not None else args.test_days
    logger.info(
        "Lancement du walk-forward (entraînement=%dj, test=%dj, pas=%dj)...",
        args.train_days,
        args.test_days,
        step_days,
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
        )
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    print("\n" + result.summary())


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
    walk_forward_parser.set_defaults(func=cmd_walk_forward)

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

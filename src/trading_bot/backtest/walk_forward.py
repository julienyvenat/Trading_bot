"""Validation walk-forward : découpe l'historique en fenêtres glissantes
train/test (in-sample/out-of-sample) et exécute le backtest sur chacune, pour
vérifier que la performance ne tient pas qu'à la période testée en un seul
bloc (le risque principal d'un backtest classique sur toute la période :
"tuner" implicitement des paramètres sur une seule tranche d'historique sans
jamais les valider ailleurs).

Deux modes :
  - `param_grids=None` (défaut) : ré-exécute la même config, à paramètres
    FIXES, sur chaque sous-période. Valide la STABILITÉ de la performance
    dans le temps, mais ne dit rien sur l'overfitting du choix des
    paramètres eux-mêmes.
  - `param_grids` fourni : pour chaque fenêtre, `trading_bot.backtest.
    optimizer.optimize` cherche les meilleurs paramètres sur la fenêtre
    d'ENTRAÎNEMENT uniquement (jamais sur le test, pour ne pas fuiter
    d'information du futur), puis les applique tels quels à la fenêtre de
    TEST hors échantillon. C'est un walk-forward analysis au sens strict.
    Si les paramètres optimaux varient énormément d'une fenêtre à l'autre
    (voir `WalkForwardFold.best_params` dans le résumé), c'est en soi un
    signal d'alerte d'overfitting sur la grille choisie.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.metrics import BacktestMetrics, compute_metrics
from trading_bot.backtest.optimizer import ParamGrid, apply_params, optimize
from trading_bot.config import AppConfig


@dataclass
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    # Paramètres retenus pour cette fenêtre : None si `param_grids` n'a pas
    # été fourni à `run_walk_forward` (paramètres fixes de `config.yaml`).
    best_params: dict[str, dict[str, Any]] | None = None


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    combined_out_of_sample_equity: pd.Series
    combined_out_of_sample_metrics: BacktestMetrics

    def summary(self) -> str:
        lines = ["=== Walk-forward : par fenêtre (entraînement -> test hors échantillon) ==="]
        for i, fold in enumerate(self.folds, start=1):
            lines.append(
                f"Fenêtre {i} : entraînement {fold.train_start.date()} -> {fold.train_end.date()} "
                f"(rendement {fold.train_metrics.total_return_pct:+.2f}%, Sharpe {fold.train_metrics.sharpe_ratio:.2f}) "
                f"| test {fold.test_start.date()} -> {fold.test_end.date()} "
                f"(rendement {fold.test_metrics.total_return_pct:+.2f}%, Sharpe {fold.test_metrics.sharpe_ratio:.2f}, "
                f"max DD {fold.test_metrics.max_drawdown_pct:.2f}%)"
            )
            if fold.best_params is not None:
                lines.append(f"    Paramètres retenus (optimisés sur l'entraînement) : {fold.best_params}")
        lines.append("")
        lines.append("=== Cumulé sur toutes les fenêtres de TEST (hors échantillon uniquement) ===")
        lines.append(self.combined_out_of_sample_metrics.summary())
        return "\n".join(lines)


def make_folds(
    start: pd.Timestamp, end: pd.Timestamp, train_days: int, test_days: int, step_days: int
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Découpe [start, end] en fenêtres glissantes (train_start, train_end,
    test_start, test_end). `step_days` avance chaque fenêtre suivante (mettre
    `step_days = test_days` pour des fenêtres de test contiguës et non
    chevauchantes, recommandé pour que la courbe hors échantillon cumulée
    reste interprétable)."""
    folds = []
    train_start = start
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > end:
            break
        folds.append((train_start, train_end, test_start, test_end))
        train_start = train_start + pd.Timedelta(days=step_days)
    return folds


def run_walk_forward(
    config: AppConfig,
    data_by_symbol: dict[str, pd.DataFrame],
    train_days: int,
    test_days: int,
    step_days: int,
    benchmark_df: pd.DataFrame | None = None,
    volatility_benchmark_df: pd.DataFrame | None = None,
    param_grids: list[ParamGrid] | None = None,
    optimization_metric: str = "sharpe_ratio",
    max_workers: int | None = None,
) -> WalkForwardResult:
    """Lance un backtest répété sur des fenêtres glissantes train/test.

    Sans `param_grids` (défaut), réutilise `run_backtest` tel quel sur
    chaque sous-période (mêmes stratégies, même config de risque) : voir la
    docstring du module sur ce que ce mode valide (stabilité dans le temps)
    et ne valide pas (absence d'overfitting sur le choix des paramètres
    eux-mêmes).

    Avec `param_grids`, ré-optimise les paramètres sur chaque fenêtre
    d'entraînement (voir `trading_bot.backtest.optimizer.optimize`) puis les
    applique à la fenêtre de test correspondante — jamais l'inverse, pour ne
    jamais utiliser d'information de la période de test au moment de choisir
    les paramètres.
    """
    full_index: pd.Index | None = None
    for df in data_by_symbol.values():
        full_index = df.index if full_index is None else full_index.union(df.index)
    if full_index is None or len(full_index) == 0:
        raise ValueError("Aucune donnée disponible pour construire les fenêtres walk-forward.")

    start, end = full_index.min(), full_index.max()
    raw_folds = make_folds(start, end, train_days, test_days, step_days)
    if not raw_folds:
        raise ValueError(
            f"Aucune fenêtre walk-forward complète ne tient dans l'historique disponible "
            f"({start.date()} -> {end.date()}) avec train_days={train_days}, test_days={test_days}. "
            "Réduis --train-days/--test-days ou fournis un historique plus long."
        )

    folds: list[WalkForwardFold] = []
    test_equity_segments: list[pd.Series] = []

    for train_start, train_end, test_start, test_end in raw_folds:
        train_config = _with_backtest_dates(config, train_start, train_end)
        test_config = _with_backtest_dates(config, test_start, test_end)

        best_params: dict[str, dict[str, Any]] | None = None
        if param_grids:
            optimization_results = optimize(
                train_config,
                data_by_symbol,
                param_grids,
                metric=optimization_metric,
                benchmark_df=benchmark_df,
                volatility_benchmark_df=volatility_benchmark_df,
                max_workers=max_workers,
            )
            best = optimization_results[0]
            best_params = best.params_by_strategy
            # Réutilise le résultat déjà calculé par `optimize` (le meilleur
            # combo EST le backtest d'entraînement) plutôt que de relancer un
            # `run_backtest` d'entraînement redondant.
            train_metrics = best.metrics
            test_config = apply_params(test_config, best_params)
        else:
            train_result = run_backtest(
                train_config, data_by_symbol, benchmark_df=benchmark_df, volatility_benchmark_df=volatility_benchmark_df
            )
            train_metrics = train_result.metrics

        test_result = run_backtest(
            test_config, data_by_symbol, benchmark_df=benchmark_df, volatility_benchmark_df=volatility_benchmark_df
        )

        folds.append(
            WalkForwardFold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_metrics=train_metrics,
                test_metrics=test_result.metrics,
                best_params=best_params,
            )
        )
        if len(test_result.equity_curve) > 0:
            test_equity_segments.append(test_result.equity_curve)

    combined_equity = _chain_equity_curves(test_equity_segments)
    # Volontairement sans `trades=` ici : les trades de chaque fenêtre de
    # test portent sur une equity qui a été RECALÉE (voir
    # `_chain_equity_curves`), donc leur P&L en $ ne serait plus cohérent
    # avec cette courbe combinée. Les stats par trade (num_trades,
    # trade_win_rate_pct...) du résumé cumulé valent donc 0 par construction
    # — regarde les métriques par trade de chaque fenêtre individuellement
    # (`fold.test_metrics`) si tu veux ces stats.
    combined_metrics = compute_metrics(combined_equity)

    return WalkForwardResult(
        folds=folds,
        combined_out_of_sample_equity=combined_equity,
        combined_out_of_sample_metrics=combined_metrics,
    )


def _with_backtest_dates(config: AppConfig, start: pd.Timestamp, end: pd.Timestamp) -> AppConfig:
    """Renvoie une copie de `config` avec `backtest.start_date`/`end_date`
    remplacés (les autres paramètres, notamment `initial_cash`, restent
    identiques entre fenêtres pour rester comparables)."""
    new_config = copy.deepcopy(config)
    new_config.backtest.start_date = start.strftime("%Y-%m-%d")
    new_config.backtest.end_date = end.strftime("%Y-%m-%d")
    return new_config


def _chain_equity_curves(segments: list[pd.Series]) -> pd.Series:
    """Concatène des segments d'equity hors échantillon en une seule courbe
    continue, en recalant chaque segment pour repartir de la valeur finale du
    précédent (chaque fenêtre de test repart de `initial_cash` dans
    `run_backtest` ; ce recalage simule un réinvestissement continu du
    capital d'une fenêtre de test à l'autre, comme un seul passage continu)."""
    if not segments:
        return pd.Series(dtype=float)

    chained = [segments[0]]
    running_value = float(segments[0].iloc[-1])
    for segment in segments[1:]:
        if len(segment) == 0 or segment.iloc[0] == 0:
            continue
        scale = running_value / float(segment.iloc[0])
        rescaled = segment * scale
        chained.append(rescaled)
        running_value = float(rescaled.iloc[-1])

    return pd.concat(chained).sort_index()

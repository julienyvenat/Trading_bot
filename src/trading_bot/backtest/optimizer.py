"""Recherche de paramètres par grille (grid search), en orchestrant plusieurs
appels indépendants à `trading_bot.backtest.engine.run_backtest` sur des
processus séparés.

Le moteur event-driven lui-même n'est ni modifié ni dupliqué : chaque
combinaison de paramètres tourne exactement le même `run_backtest` que le
reste du bot (stratégies, allocateur, risque, stops, coupe-circuits), pour
garantir que le résultat de l'optimisation correspond à ce qui tournerait
réellement en live avec ces paramètres — pas d'approximation vectorisée qui
introduirait un écart entre "ce qui est optimisé" et "ce qui est tradé".

Parallélisation : `multiprocessing.Pool` avec un `initializer` qui pousse la
config/les données dans chaque worker UNE SEULE FOIS à la création du pool,
plutôt que de les re-sérialiser à chaque combinaison testée (qui serait
prohibitif dès que l'historique ou le nombre de combinaisons grandit).
"""

from __future__ import annotations

import copy
import itertools
import multiprocessing
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.metrics import BacktestMetrics
from trading_bot.config import AppConfig

ParamCombo = dict[str, dict[str, Any]]  # {nom_stratégie: {param: valeur}}


@dataclass
class ParamGrid:
    strategy_name: str
    # Ex: {"fast_window": [10, 20, 30], "slow_window": [50, 100, 150]}
    params: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    params_by_strategy: ParamCombo
    metrics: BacktestMetrics


def generate_param_combinations(grids: list[ParamGrid]) -> list[ParamCombo]:
    """Produit cartésien de toutes les grilles fournies, combinées ENTRE
    stratégies (attention à l'explosion combinatoire : le nombre total de
    runs est le produit du nombre de combinaisons de chaque grille)."""
    per_strategy_combos: list[list[tuple[str, dict[str, Any]]]] = []

    for grid in grids:
        if not grid.params:
            per_strategy_combos.append([(grid.strategy_name, {})])
            continue
        keys = list(grid.params.keys())
        value_lists = [grid.params[k] for k in keys]
        combos_for_strategy = [
            (grid.strategy_name, dict(zip(keys, values))) for values in itertools.product(*value_lists)
        ]
        per_strategy_combos.append(combos_for_strategy)

    if not per_strategy_combos:
        return []

    result: list[ParamCombo] = []
    for combo_tuple in itertools.product(*per_strategy_combos):
        result.append({strategy_name: params for strategy_name, params in combo_tuple})
    return result


def apply_params(config: AppConfig, combo: ParamCombo) -> AppConfig:
    """Copie profonde de `config` avec les paramètres de chaque stratégie
    concernée FUSIONNÉS (pas remplacés) dans `StrategyConfig.params`, pour ne
    pas perdre des paramètres non grillés."""
    new_config = copy.deepcopy(config)
    for strategy_config in new_config.strategies:
        overrides = combo.get(strategy_config.name)
        if overrides:
            strategy_config.params = {**strategy_config.params, **overrides}
    return new_config


# État process-local peuplé une seule fois par `_init_worker` : évite de
# repickler `data_by_symbol` (potentiellement volumineux) à chaque tâche.
_worker_state: dict[str, Any] = {}


def _init_worker(
    config: AppConfig,
    data_by_symbol: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame | None,
    volatility_benchmark_df: pd.DataFrame | None,
) -> None:
    _worker_state["config"] = config
    _worker_state["data_by_symbol"] = data_by_symbol
    _worker_state["benchmark_df"] = benchmark_df
    _worker_state["volatility_benchmark_df"] = volatility_benchmark_df


def _evaluate_combo(combo: ParamCombo) -> OptimizationResult:
    state = _worker_state
    combo_config = apply_params(state["config"], combo)
    result = run_backtest(
        combo_config,
        state["data_by_symbol"],
        benchmark_df=state["benchmark_df"],
        volatility_benchmark_df=state["volatility_benchmark_df"],
    )
    return OptimizationResult(params_by_strategy=combo, metrics=result.metrics)


def optimize(
    config: AppConfig,
    data_by_symbol: dict[str, pd.DataFrame],
    grids: list[ParamGrid],
    metric: str = "sharpe_ratio",
    benchmark_df: pd.DataFrame | None = None,
    volatility_benchmark_df: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> list[OptimizationResult]:
    """Lance `run_backtest` pour chaque combinaison de paramètres de `grids`
    (produit cartésien), en parallèle. Renvoie les résultats triés du
    meilleur au pire selon `metric` (n'importe quel attribut numérique de
    `BacktestMetrics`, ex: "sharpe_ratio", "calmar_ratio", "total_return_pct").

    `max_workers=1` force une exécution séquentielle dans le process courant
    (pas de `Pool`) : pratique pour déboguer ou dans un environnement qui
    interdit le multiprocessing (ex: certains runners de CI).
    """
    combos = generate_param_combinations(grids)
    if not combos:
        raise ValueError("Aucune combinaison de paramètres à tester (grilles vides).")

    if max_workers == 1:
        _init_worker(config, data_by_symbol, benchmark_df, volatility_benchmark_df)
        results = [_evaluate_combo(combo) for combo in combos]
    else:
        with multiprocessing.Pool(
            processes=max_workers,
            initializer=_init_worker,
            initargs=(config, data_by_symbol, benchmark_df, volatility_benchmark_df),
        ) as pool:
            results = pool.map(_evaluate_combo, combos)

    results.sort(key=lambda r: getattr(r.metrics, metric), reverse=True)
    return results


def best_params(results: list[OptimizationResult]) -> ParamCombo:
    """Renvoie les paramètres de la meilleure combinaison (`results` déjà
    trié, voir `optimize`)."""
    if not results:
        raise ValueError("Aucun résultat d'optimisation à partir duquel choisir les meilleurs paramètres.")
    return results[0].params_by_strategy

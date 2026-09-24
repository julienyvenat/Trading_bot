"""Compare, à frais réels Fortuneo, les modes PEA (a) buy & hold PSP5 (avec
ou sans stop suiveur large) et (b) rotation mensuelle dual momentum entre
ETF PEA, sur 2018→aujourd'hui puis en échantillon (2018-2022) / hors
échantillon (2023→aujourd'hui).

Usage (réseau requis : yfinance) :
    python scripts/pea_compare.py [--cache /tmp/pea_data.pkl]

Rien n'est écrit dans le dépôt : sortie console (tableaux Markdown).
"""

from __future__ import annotations

import argparse
import copy
import pickle
from dataclasses import replace
from pathlib import Path

import pandas as pd

from trading_bot.backtest.engine import run_backtest
from trading_bot.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BUYHOLD_CONFIG = ROOT / "config" / "config_pea_buyhold.example.yaml"
MOMENTUM_CONFIG = ROOT / "config" / "config_pea_etf_momentum.example.yaml"
DATA_START = "2016-06-01"


def _load_data(symbols: list[str], cache: str | None) -> dict[str, pd.DataFrame]:
    if cache and Path(cache).exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
        if set(symbols) <= set(data):
            return data
    from trading_bot.data.historical import fetch_historical_data

    data = fetch_historical_data(symbols, start_date=DATA_START)
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(data, f)
    return data


def _run(config, data, start: str, end: str | None):
    config = copy.deepcopy(config)
    config.backtest = replace(config.backtest, start_date=start, end_date=end)
    subset = {s: data[s] for s in config.symbols if s in data}
    result = run_backtest(config, subset)
    m = result.metrics
    years = max(m.num_trading_days / 252, 1e-9)
    return {
        "total": m.total_return_pct,
        "cagr": m.annualized_return_pct,
        "maxdd": m.max_drawdown_pct,
        "sharpe": m.sharpe_ratio,
        "calmar": m.calmar_ratio,
        "orders_y": result.num_orders / years,
        "fees": result.total_fees,
        "stops": result.num_stop_exits,
    }


def _row(label: str, r: dict) -> str:
    return (
        f"| {label} | {r['total']:+.1f} % | {r['cagr']:+.1f} % | {r['maxdd']:.1f} % | {r['sharpe']:.2f} | "
        f"{r['calmar']:.2f} | {r['orders_y']:.1f} | {r['fees']:.2f} € |"
    )


HEADER = (
    "| Variante | Total | CAGR | Max DD | Sharpe | Calmar | Ordres/an | Frais totaux |\n"
    "|---|---|---|---|---|---|---|---|"
)


def buyhold_variants(base):
    variants = [("Buy & hold PSP5 (sans stop)", replace(base.risk, stop_mode="none"))]
    for k in (4, 6, 8):
        variants.append(
            (f"PSP5 + stop suiveur ATR×{k} (figé à l'entrée)",
             replace(base.risk, stop_mode="trailing_pct", trailing_stop_pct=None, atr_stop_multiple=float(k)))
        )
    for pct in (0.15, 0.20, 0.25):
        variants.append(
            (f"PSP5 + stop suiveur {pct:.0%}", replace(base.risk, stop_mode="trailing_pct", trailing_stop_pct=pct))
        )
    out = []
    for label, risk in variants:
        cfg = copy.deepcopy(base)
        cfg.risk = risk
        out.append((label, cfg))
    return out


def momentum_variants(base):
    out = []
    for lookback in (126, 252):
        for top_n in (1, 2):
            cfg = copy.deepcopy(base)
            cfg.strategies[0].params = {**cfg.strategies[0].params, "lookback_window": lookback, "top_n": top_n}
            out.append((f"Dual momentum {lookback // 21} mois, top {top_n}", cfg))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=None, help="Fichier pickle de cache des données yfinance.")
    args = parser.parse_args()

    buyhold = load_config(BUYHOLD_CONFIG)
    momentum = load_config(MOMENTUM_CONFIG)
    data = _load_data(sorted(set(buyhold.symbols) | set(momentum.symbols)), args.cache)
    last = max(df.index.max() for df in data.values()).date()
    print(f"Données jusqu'au {last}. Capital initial {buyhold.backtest.initial_cash} €, frais Fortuneo par paliers.\n")

    periods = [("2018-01-01 → aujourd'hui", "2018-01-01", None),
               ("En échantillon 2018-2022", "2018-01-01", "2022-12-31"),
               ("Hors échantillon 2023 → aujourd'hui", "2023-01-01", None)]

    for title, variants in (("(a) Buy & hold PSP5 et stops", buyhold_variants(buyhold)),
                            ("(b) Dual momentum ETF PEA", momentum_variants(momentum))):
        for period_label, start, end in periods:
            print(f"### {title} — {period_label}\n")
            print(HEADER)
            if title.startswith("(b)"):
                print(_row("Référence : buy & hold PSP5", _run(buyhold_variants(buyhold)[0][1], data, start, end)))
            for label, cfg in variants:
                print(_row(label, _run(cfg, data, start, end)))
            print()


if __name__ == "__main__":
    main()

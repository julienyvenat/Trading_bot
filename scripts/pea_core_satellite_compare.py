"""Compare, à frais réels Fortuneo et en actions entières, le plan PEA
cœur-satellite (55 % MSCI World / 20 % S&P 500 / 25 % levier x2 USA) à ses
variantes (0 / 25 / 50 % de levier) et aux buy & hold 100 % PSP5 et 100 %
World, avec 2 306,59 € ou 20 000 € de départ, avec ou sans 100 €/mois.

Deux jeux de données :
  (a) ETF réels (yfinance) depuis la première date commune : DCAM.PA n'a
      d'historique que depuis le 04/03/2025, donc AVANT cette date la poche
      World utilise CW8.PA (Amundi MSCI World, même indice, en EUR), remis à
      l'échelle du cours de DCAM (actions entières réalistes). CL2.PA :
      cotations aberrantes de yfinance (x300 un jour sans volume, série non
      ajustée avant 2012) nettoyées, voir `clean_glitches`.
  (b) Stress test synthétique 1990→2026 et sous-périodes 2000→2012,
      2007→2009 (calendrier NYSE, USD, change ignoré) :
        - S&P 500 = ^SP500TR (dividendes réinvestis) - 0,15 %/an de frais ;
        - levier x2 = 2 x r(^SP500TR) - (^IRX/100 + 0,6 %)/252 par jour ;
        - World = ^990100-USD-STRD (MSCI World, indice PRIX) + 2,3 %/an de
          dividendes (écart mesuré entre URTH et cet indice sur 2012-2026) -
          0,20 %/an de frais.
      Chaque série synthétique démarre à un cours réaliste (6 / 60 / 30).

Métriques : CAGR = rendement pondéré par le temps (NAV par part, hors effet
des apports), annualisé sur le temps calendaire ; TRI = rendement pondéré par l'argent (versements datés) ;
Sharpe sans taux sans risque (rf = 0) ; pire 12 mois et drawdown sur la NAV.

Usage (réseau requis : yfinance) :
    python scripts/pea_core_satellite_compare.py [--cache /tmp/cs_data.pkl] [--json sortie.json]

Rien n'est écrit dans le dépôt : sortie console (tableaux Markdown).
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest.engine import run_backtest
from trading_bot.backtest.metrics import max_drawdown_recovery, worst_rolling_return
from trading_bot.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "config_pea_core_satellite.example.yaml"
W, SP, LEV = "DCAM.PA", "PSP5.PA", "CL2.PA"
TICKERS = [W, "CW8.PA", SP, LEV, "^SP500TR", "^IRX", "^990100-USD-STRD"]

WORLD_DIVIDEND = 0.023
WORLD_TER = 0.002
SP_TER = 0.0015
LEV_SPREAD = 0.006


def variants() -> list[tuple[str, dict[str, float]]]:
    core = {W: 55 / 75, SP: 20 / 75}  # répartition World/S&P du plan hors levier

    def with_lev(lev: float) -> dict[str, float]:
        weights = {s: w * (1 - lev) for s, w in core.items()}
        if lev:
            weights[LEV] = lev
        return weights

    return [
        ("**Plan 55/20/25**", {W: 0.55, SP: 0.20, LEV: 0.25}),
        ("0 % levier (73/27)", with_lev(0.0)),
        ("50 % levier (37/13/50)", with_lev(0.5)),
        ("100 % PSP5 (buy & hold)", {SP: 1.0}),
        ("100 % World (buy & hold)", {W: 1.0}),
    ]


def _download(cache: str | None) -> dict[str, pd.DataFrame]:
    if cache and Path(cache).exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
        if set(TICKERS) <= set(data):
            return data
    from trading_bot.data.historical import fetch_historical_data

    data = fetch_historical_data(TICKERS, start_date="1985-01-01")
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(data, f)
    return data


def clean_glitches(df: pd.DataFrame, start: str | None = None) -> pd.DataFrame:
    """Retire les cotations aberrantes (clôture à plus de 3x ou moins d'1/3
    de la médiane glissante) et remplace les ouvertures nulles."""
    df = df.loc[start:] if start else df
    median = df["close"].rolling(21, center=True, min_periods=1).median()
    ok = (df["close"] <= 3 * median) & (df["close"] >= median / 3)
    df = df[ok].copy()
    bad_open = (df["open"] <= 0) | (df["open"] > 3 * df["close"]) | (df["open"] < df["close"] / 3)
    df.loc[bad_open, "open"] = df["close"].shift(1)[bad_open]
    return df.dropna()


def real_data(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    dcam, cw8 = raw[W], raw["CW8.PA"]
    splice = dcam.index[0]
    scale = float(dcam["close"].iloc[0]) / float(cw8["close"].asof(splice))
    before = cw8.loc[cw8.index < splice].copy()
    before[["open", "high", "low", "close"]] *= scale
    world = pd.concat([before, dcam]).sort_index()
    lev = clean_glitches(raw[LEV], start="2012-01-02")
    data = {W: world, SP: raw[SP], LEV: lev}
    info = {
        "splice": splice.date().isoformat(),
        "start": max(df.index[0] for df in data.values()).date().isoformat(),
        "end": min(df.index[-1] for df in data.values()).date().isoformat(),
        "lev_removed": int(len(raw[LEV].loc["2012-01-02":]) - len(lev)),
    }
    return data, info


def synthetic_data(raw: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Rendements quotidiens synthétiques (voir docstring du module)."""
    frame = pd.concat(
        {
            "sptr": raw["^SP500TR"]["close"],
            "msci": raw["^990100-USD-STRD"]["close"],
            "irx": raw["^IRX"]["close"],
        },
        axis=1,
    ).sort_index()
    frame = frame[frame["sptr"].notna()].ffill().dropna()
    r_sp = frame["sptr"].pct_change()
    r_world = frame["msci"].pct_change() + (WORLD_DIVIDEND - WORLD_TER) / 252
    r_lev = 2 * r_sp - (frame["irx"] / 100 + LEV_SPREAD) / 252
    return {W: r_world.dropna(), SP: (r_sp - SP_TER / 252).dropna(), LEV: r_lev.dropna()}


def synthetic_ohlcv(returns: dict[str, pd.Series], start: str, end: str) -> dict[str, pd.DataFrame]:
    start_price = {W: 6.0, SP: 60.0, LEV: 30.0}
    out = {}
    for symbol, r in returns.items():
        r = r.loc[start:end]
        close = start_price[symbol] * (1 + r).cumprod() / (1 + r.iloc[0])
        out[symbol] = pd.DataFrame(
            {"open": close.shift(1).fillna(close.iloc[0]), "high": close, "low": close, "close": close, "volume": 1}
        )
    return out


def run(base, data, weights, start, end, cash, contribution, calendar):
    config = copy.deepcopy(base)
    config.symbols = list(weights)
    config.strategies[0].params = {**config.strategies[0].params, "targets": weights}
    config.market = replace(config.market, calendar=calendar)
    config.backtest = replace(
        config.backtest, start_date=start, end_date=end, initial_cash=cash, monthly_contribution=contribution
    )
    result = run_backtest(config, {s: data[s] for s in weights})
    nav = result.nav_curve
    m = result.metrics
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    peak, trough, recovered = max_drawdown_recovery(nav)
    worst = worst_rolling_return(nav)
    return {
        # Annualisé sur le temps calendaire (comme le TRI), pas sur 252 séances.
        "cagr": ((float(nav.iloc[-1]) / float(nav.iloc[0])) ** (1 / years) - 1) * 100,
        "irr": result.money_weighted_return_pct,
        "invested": result.total_contributed,
        "final": float(result.equity_curve.iloc[-1]),
        "maxdd": m.max_drawdown_pct,
        "sharpe": m.sharpe_ratio,
        "worst12": None if worst is None else worst * 100,
        "dd_peak": None if peak is None else peak.date().isoformat(),
        "dd_trough": None if trough is None else trough.date().isoformat(),
        "recovery_years": None if recovered is None else (recovered - peak).days / 365.25,
        "orders_y": result.num_orders / max(years, 1e-9),
        "fees": result.total_fees,
        "start": nav.index[0].date().isoformat(),
        "end": nav.index[-1].date().isoformat(),
    }


def fmt(x, pattern="{:+.1f} %"):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else pattern.format(x).replace(".", ",")


def eur0(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ") + " €"


def row(label, r):
    rec = "non récupéré" if r["recovery_years"] is None and r["dd_peak"] else fmt(r["recovery_years"], "{:.1f} ans")
    return (
        f"| {label} | {fmt(r['cagr'])} | {fmt(r['irr'])} | {eur0(r['invested'])} → {eur0(r['final'])} | "
        f"{fmt(r['maxdd'])} | {fmt(r['sharpe'], '{:.2f}')} | {fmt(r['worst12'])} | {rec} | "
        f"{fmt(r['orders_y'], '{:.1f}')} | {eur0(r['fees'])} |"
    )


HEADER = (
    "| Variante | CAGR (NAV) | TRI | Versé → final | Max DD | Sharpe | Pire 12 mois | Récup. max DD | Ordres/an | Frais |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=None, help="Fichier pickle de cache des données yfinance.")
    parser.add_argument("--json", default=None, help="Écrit aussi tous les résultats dans ce fichier JSON.")
    args = parser.parse_args()

    base = load_config(CONFIG)
    raw = _download(args.cache)
    real, info = real_data(raw)
    synth = synthetic_data(raw)
    print(
        f"(a) ETF réels : {info['start']} → {info['end']} ; World = CW8.PA remis à l'échelle avant le "
        f"{info['splice']}, DCAM.PA ensuite ; {info['lev_removed']} cotation(s) aberrante(s) de CL2.PA retirée(s)."
    )
    print(f"(b) Synthétique : {synth[SP].index[0].date()} → {synth[SP].index[-1].date()} (USD, change ignoré).\n")

    periods = [
        ("(a) ETF réels", real, info["start"], None, "XPAR"),
        ("(a') ETF réels, DCAM seul (sans proxy)", real, info["splice"], None, "XPAR"),
        ("(b) Synthétique 1990→2026", None, "1990-01-02", None, "NYSE"),
        ("(b) Synthétique 2000→2012", None, "2000-01-03", "2012-12-31", "NYSE"),
        ("(b) Synthétique 2007→2009", None, "2007-01-02", "2009-12-31", "NYSE"),
    ]
    results = []
    for title, data, start, end, calendar in periods:
        if data is None:
            data = synthetic_ohlcv(synth, start, end or str(synth[SP].index[-1].date()))
        for cash in (2306.59, 20000.0):
            for contribution in (0.0, 100.0):
                label = f"{cash:,.2f} €".replace(",", " ") + (f" + {contribution:.0f} €/mois" if contribution else "")
                print(f"### {title} — départ {label}\n")
                print(HEADER)
                for name, weights in variants():
                    r = run(base, data, weights, start, end, cash, contribution, calendar)
                    results.append({"period": title, "cash": cash, "contribution": contribution, "variant": name, **r})
                    print(row(name, r))
                print(f"\n(période simulée : {r['start']} → {r['end']})\n")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

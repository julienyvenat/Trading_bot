"""Base de suivi PERSISTANTE, par symbole, des trades réalisés au fil des
cycles (backtest ou live) — un historique de performance réelle qui
s'accumule d'un run à l'autre, plutôt que d'être recalculé à froid depuis
zéro à chaque fois (contrairement à `trading_bot.portfolio.universe_rotation`,
qui ne regarde qu'un historique de PRIX). Utilisable comme `metric:
"track_record"` dans `UniverseRotationConfig` : au lieu de classer les
candidats sur un proxy technique (volatilité/momentum), on les classe sur
leur VÉCU réel avec ce bot.

⚠️ Cette accumulation n'apporte de l'information réellement nouvelle que si
les trades enregistrés proviennent de périodes RÉELLEMENT nouvelles (paper
trading live, jour après jour, ou fenêtres de walk-forward qui ne se
recouvrent pas). Rejouer plusieurs fois un backtest sur EXACTEMENT la même
période ne fait qu'écraser les mêmes trades avec les mêmes trades : `merge`
est idempotent (basé sur une marque d'eau `last_exit_date` par symbole, pas
un compteur qui grossirait à chaque replay) précisément pour éviter de
gonfler artificiellement `trade_count` en relançant le même backtest en
boucle. Voir aussi `trading_bot.cli` (`--update-track-record`, opt-in
explicite plutôt qu'une mise à jour silencieuse à chaque backtest).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    # Import différé : `trading_bot.backtest` importe (via son __init__)
    # `trading_bot.backtest.engine`, qui importe `trading_bot.portfolio.
    # allocator`, qui importe `trading_bot.portfolio.universe_rotation`, qui
    # importe CE module — un import direct en tête de fichier créerait un
    # cycle. `Trade` n'est utilisé ici que comme annotation de type (voir
    # `from __future__ import annotations` ci-dessus, qui les rend
    # paresseuses), donc cet import `TYPE_CHECKING` suffit.
    from trading_bot.backtest.trades import Trade


@dataclass
class SymbolStats:
    trade_count: int = 0
    win_count: int = 0
    total_pnl_pct: float = 0.0  # somme des pnl_pct (voir Trade.pnl_pct) : permet la moyenne
    total_pnl_pct_sq: float = 0.0  # somme des carrés : permet l'écart-type sans stocker chaque trade
    last_exit_date: str | None = None  # ISO date : marque d'eau, voir docstring du module

    @property
    def win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count else 0.0

    @property
    def avg_pnl_pct(self) -> float:
        return self.total_pnl_pct / self.trade_count if self.trade_count else 0.0

    @property
    def pnl_pct_stdev(self) -> float:
        if self.trade_count < 2:
            return 0.0
        mean = self.avg_pnl_pct
        variance = max(0.0, self.total_pnl_pct_sq / self.trade_count - mean * mean)
        return variance**0.5

    def expectancy_score(self, prior_trades: int = 20) -> float:
        """Gain moyen par trade (`avg_pnl_pct`), atténué vers 0 (a priori
        neutre : "pas d'edge connu") tant que l'échantillon est petit —
        shrinkage bayésien simple pour qu'un symbole avec 2 trades gagnants
        sur 2 ne paraisse pas déjà "excellent". `prior_trades` est le
        nombre de trades "virtuels" du prior neutre : plus il est grand,
        plus il faut de trades réels pour que le score s'éloigne de 0."""
        weight = self.trade_count / (self.trade_count + prior_trades)
        return weight * self.avg_pnl_pct

    def to_dict(self) -> dict:
        return {
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "total_pnl_pct": self.total_pnl_pct,
            "total_pnl_pct_sq": self.total_pnl_pct_sq,
            "last_exit_date": self.last_exit_date,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SymbolStats:
        return cls(
            trade_count=data.get("trade_count", 0),
            win_count=data.get("win_count", 0),
            total_pnl_pct=data.get("total_pnl_pct", 0.0),
            total_pnl_pct_sq=data.get("total_pnl_pct_sq", 0.0),
            last_exit_date=data.get("last_exit_date"),
        )


@dataclass
class SymbolTrackRecord:
    by_symbol: dict[str, SymbolStats] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {symbol: stats.to_dict() for symbol, stats in self.by_symbol.items()}

    @classmethod
    def from_dict(cls, data: dict) -> SymbolTrackRecord:
        return cls(by_symbol={symbol: SymbolStats.from_dict(v) for symbol, v in data.items()})


def load_track_record(path: str | Path) -> SymbolTrackRecord:
    file_path = Path(path)
    if not file_path.exists():
        return SymbolTrackRecord()
    with open(file_path, encoding="utf-8") as f:
        return SymbolTrackRecord.from_dict(json.load(f))


def save_track_record(path: str | Path, record: SymbolTrackRecord) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)


def merge_trades(record: SymbolTrackRecord, trades: list[Trade]) -> SymbolTrackRecord:
    """Intègre `trades` dans `record` (fonction pure, renvoie un nouveau
    `SymbolTrackRecord`). Idempotent : un trade dont `exit_date` est
    antérieure ou égale à la marque d'eau `last_exit_date` déjà connue pour
    ce symbole AVANT cet appel est ignoré (déjà comptabilisé lors d'un run
    précédent, ou rejeu du même backtest) — voir l'avertissement en tête de
    module."""
    by_symbol = {symbol: SymbolStats.from_dict(stats.to_dict()) for symbol, stats in record.by_symbol.items()}
    # Marques d'eau figées AVANT la boucle (snapshot), jamais relues pendant
    # qu'on avance : sans ça, deux trades du MÊME symbole partageant (ou se
    # suivant à) la même exit_date au sein de CET APPEL se filtreraient
    # mutuellement — le premier traité avancerait la marque d'eau, faisant
    # paraître le second "déjà connu" alors qu'il ne l'était pas avant cet
    # appel.
    watermarks_before = {
        symbol: pd.Timestamp(stats.last_exit_date) for symbol, stats in by_symbol.items() if stats.last_exit_date
    }

    for trade in sorted(trades, key=lambda t: t.exit_date):
        stats = by_symbol.setdefault(trade.symbol, SymbolStats())
        watermark = watermarks_before.get(trade.symbol)
        if watermark is not None and trade.exit_date <= watermark:
            continue

        stats.trade_count += 1
        stats.win_count += 1 if trade.pnl > 0 else 0
        stats.total_pnl_pct += trade.pnl_pct
        stats.total_pnl_pct_sq += trade.pnl_pct**2
        stats.last_exit_date = trade.exit_date.isoformat()

    return SymbolTrackRecord(by_symbol=by_symbol)

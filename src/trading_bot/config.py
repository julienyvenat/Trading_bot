"""Chargement de la configuration (YAML + variables d'environnement)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclass
class StrategyConfig:
    name: str
    enabled: bool
    weight: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    allow_short: bool
    max_gross_exposure_pct: float
    max_position_weight_pct: float
    risk_per_trade_pct: float
    atr_stop_multiple: float
    atr_window: int
    max_open_positions: int


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str | None
    initial_cash: float
    commission_pct: float


@dataclass
class LiveConfig:
    loop_interval_seconds: int
    trade_only_when_market_open: bool


@dataclass
class AppConfig:
    symbols: list[str]
    timeframe: str
    strategies: list[StrategyConfig]
    risk: RiskConfig
    backtest: BacktestConfig
    live: LiveConfig

    def enabled_strategies(self) -> list[StrategyConfig]:
        return [s for s in self.strategies if s.enabled]


def load_config(path: str | Path | None = None) -> AppConfig:
    """Charge la configuration YAML du bot et les variables d'environnement (.env)."""
    load_dotenv()

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    strategies = [
        StrategyConfig(
            name=s["name"],
            enabled=s.get("enabled", True),
            weight=float(s.get("weight", 1.0)),
            params=s.get("params", {}) or {},
        )
        for s in raw.get("strategies", [])
    ]

    return AppConfig(
        symbols=list(raw["universe"]["symbols"]),
        timeframe=raw["universe"].get("timeframe", "1Day"),
        strategies=strategies,
        risk=RiskConfig(**raw["risk"]),
        backtest=BacktestConfig(**raw["backtest"]),
        live=LiveConfig(**raw["live"]),
    )


@dataclass
class AlpacaCredentials:
    api_key: str
    secret_key: str
    paper: bool


def load_alpaca_credentials() -> AlpacaCredentials:
    """Lit les identifiants Alpaca depuis l'environnement (.env)."""
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    paper = os.getenv("ALPACA_PAPER", "true").strip().lower() not in ("false", "0", "no")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Clés API Alpaca manquantes. Copie .env.example vers .env et renseigne "
            "ALPACA_API_KEY / ALPACA_SECRET_KEY (voir https://app.alpaca.markets/paper/dashboard/overview)."
        )

    return AlpacaCredentials(api_key=api_key, secret_key=secret_key, paper=paper)

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
    max_daily_loss_pct: float
    max_drawdown_pct: float


@dataclass
class RegimeFilterConfig:
    """Filtre de régime de marché (voir `trading_bot.portfolio.regime`).

    Réduit l'exposition du portefeuille quand `symbol` clôture sous sa
    moyenne mobile `sma_window` (régime baissier). Désactivé par défaut pour
    rester rétro-compatible avec un config.yaml qui ne déclare pas cette
    section.
    """

    enabled: bool = False
    symbol: str = "SPY"
    sma_window: int = 200
    bearish_exposure_scale: float = 0.3
    # Symboles jamais réduits par ce filtre (ex: un actif défensif utilisé par
    # `defensive_rotation`, dont l'exposition doit au contraire AUGMENTER en
    # régime baissier — la réduction globale de ce filtre irait à l'encontre
    # de cet objectif).
    exempt_symbols: list[str] = field(default_factory=list)


@dataclass
class VolatilityFilterConfig:
    """Filtre de volatilité (voir `trading_bot.portfolio.volatility_filter`).

    Réduit l'exposition du portefeuille quand `symbol` (un proxy de
    volatilité négociable, ex. VIXY qui réplique des futures VIX court
    terme) s'envole au-dessus de sa moyenne mobile `sma_window`, signe d'un
    pic de stress de marché. Complémentaire à `RegimeFilterConfig` : celui-ci
    réagit à la direction du marché, celui-là à l'amplitude des mouvements
    récents. Désactivé par défaut pour rester rétro-compatible avec un
    config.yaml qui ne déclare pas cette section.
    """

    enabled: bool = False
    symbol: str = "VIXY"
    sma_window: int = 20
    spike_threshold_pct: float = 0.15
    spike_exposure_scale: float = 0.5
    exempt_symbols: list[str] = field(default_factory=list)


@dataclass
class MarketConfig:
    calendar: str
    close_buffer_minutes: int
    regime_filter: RegimeFilterConfig = field(default_factory=RegimeFilterConfig)
    volatility_filter: VolatilityFilterConfig = field(default_factory=VolatilityFilterConfig)


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
    state_file: str = "state/live_state.json"


@dataclass
class NewsSentimentConfig:
    """Filtre de sentiment de news (voir `trading_bot.data.news_sentiment`).

    Bloque les NOUVELLES entrées sur un symbole dont les news récentes
    (API officielle Alpaca, pas de scraping) sont majoritairement négatives
    (score sous `block_threshold`, avec au moins `min_articles` articles
    disponibles pour éviter de juger sur un seul titre isolé). Ne s'applique
    qu'au trading live (voir limite dans `trading_bot.data.news_sentiment`).
    Désactivé par défaut.
    """

    enabled: bool = False
    lookback_hours: int = 48
    block_threshold: float = -0.3
    min_articles: int = 2


@dataclass
class AppConfig:
    symbols: list[str]
    timeframe: str
    strategies: list[StrategyConfig]
    risk: RiskConfig
    market: MarketConfig
    backtest: BacktestConfig
    live: LiveConfig
    news_sentiment: NewsSentimentConfig = field(default_factory=NewsSentimentConfig)

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

    market_raw = dict(raw["market"])
    regime_raw = market_raw.pop("regime_filter", None) or {}
    volatility_raw = market_raw.pop("volatility_filter", None) or {}
    news_sentiment_raw = raw.get("news_sentiment", None) or {}

    return AppConfig(
        symbols=list(raw["universe"]["symbols"]),
        timeframe=raw["universe"].get("timeframe", "1Day"),
        strategies=strategies,
        risk=RiskConfig(**raw["risk"]),
        market=MarketConfig(
            **market_raw,
            regime_filter=RegimeFilterConfig(**regime_raw),
            volatility_filter=VolatilityFilterConfig(**volatility_raw),
        ),
        backtest=BacktestConfig(**raw["backtest"]),
        live=LiveConfig(**raw["live"]),
        news_sentiment=NewsSentimentConfig(**news_sentiment_raw),
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

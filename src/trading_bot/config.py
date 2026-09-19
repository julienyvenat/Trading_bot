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
class UniverseRotationConfig:
    """Rotation périodique du sous-ensemble de symboles tradé par UNE
    stratégie (voir `trading_bot.portfolio.universe_rotation`).

    Désactivée par défaut (rétrocompatible : une config existante qui ne
    déclare pas `universe_rotation` sous une stratégie continue de la faire
    porter sur `universe.symbols` en entier, sans aucun changement de
    comportement). Quand activée, la stratégie ne considère plus
    `universe.symbols` mais exclusivement `candidates` (un pool dédié,
    éventuellement plus large et distinct de l'univers principal), et n'est
    réellement exposée qu'aux `top_n` d'entre eux jugés les plus intéressants
    par `metric`, réévalués tous les `rebalance_every` bougies — pour éviter
    de rester indéfiniment sur des titres devenus peu intéressants (ex: une
    volatilité qui s'est tarie) simplement parce qu'ils étaient dans la
    config au départ.
    """

    enabled: bool = False
    # Pool de symboles candidats, propre à cette stratégie. Peut contenir des
    # symboles absents de `universe.symbols` : ils sont alors récupérés en
    # plus (voir `trading_bot.cli`) uniquement pour cette rotation.
    candidates: list[str] = field(default_factory=list)
    # "volatility" (écart-type des rendements, favorise les titres qui
    # bougent le plus, pertinent pour une stratégie de scalping/retour à la
    # moyenne), "momentum" (rendement glissant, pertinent pour une stratégie
    # de suivi de tendance) ou "dollar_volume" (liquidité, pour éviter les
    # titres trop étroits). Dans les 3 cas, "plus haut = plus intéressant".
    metric: str = "volatility"
    # Sélection par SEUIL de qualité plutôt que par compte fixe (nombre de
    # titres retenus volontairement variable dans le temps, voir
    # `trading_bot.portfolio.universe_rotation.compute_confidence`) : un
    # candidat est retenu si sa "confiance" — la moyenne de son rang
    # percentile sur `metric` parmi les autres candidats, sur les
    # `stability_window` dernières réévaluations — atteint `min_confidence`
    # (échelle [0, 1], 1.0 = systématiquement le meilleur candidat du pool).
    # Le nombre de titres réellement ouverts reste de toute façon borné en
    # aval par `risk.max_open_positions`/`max_gross_exposure_pct`.
    min_confidence: float = 0.6
    # Moyenner sur plusieurs réévaluations plutôt que de ne regarder que la
    # dernière évite de retenir un candidat qui n'aurait été intéressant
    # qu'une seule fois par hasard (ex: un pic de volatilité isolé) : il faut
    # avoir été bon de façon répétée, pas juste au dernier instant T.
    stability_window: int = 3
    lookback_window: int = 20
    rebalance_every: int = 20


@dataclass
class StrategyConfig:
    name: str
    enabled: bool
    weight: float
    params: dict[str, Any] = field(default_factory=dict)
    universe_rotation: UniverseRotationConfig = field(default_factory=UniverseRotationConfig)


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
    # "yfinance" (défaut, inchangé) ou "alpaca". yfinance ne garantit qu'un
    # historique intraday limité (~60 jours en 5Min/15Min/30Min, 7 jours en
    # 1Min — voir `trading_bot.data.historical`) : au-delà, ou pour un
    # walk-forward avec assez de fenêtres hors échantillon sur une stratégie
    # intraday, utiliser "alpaca" (voir `trading_bot.data.market_data.
    # fetch_historical_bars`, plusieurs années d'historique même sur un
    # compte gratuit/paper) — nécessite les identifiants Alpaca dans `.env`
    # (voir `trading_bot.config.load_alpaca_credentials`), pas seulement
    # pour le live.
    data_source: str = "yfinance"


@dataclass
class ManualBrokerConfig:
    """Paramètres du broker "manuel" (voir `trading_bot.execution.
    manual_broker.ManualBroker`), utilisé quand aucune API de courtage n'est
    disponible (ex: PEA Fortuneo) : le bot calcule les ordres mais ne les
    envoie jamais lui-même, il affiche l'instruction exacte à exécuter à la
    main et tient sa comptabilité (cash, positions) dans `account_file`,
    que l'utilisateur initialise avec le solde/les positions réels de son
    compte avant le premier cycle."""

    account_file: str = "state/manual_account.json"


@dataclass
class LiveConfig:
    loop_interval_seconds: int
    trade_only_when_market_open: bool
    state_file: str = "state/live_state.json"
    # "alpaca" (défaut, inchangé) : ordres envoyés automatiquement via l'API
    # Alpaca. "manual" : aucune API de courtage (ex: PEA Fortuneo), le bot
    # affiche chaque ordre/stop à exécuter à la main sur le site du courtier
    # et utilise yfinance (pas Alpaca) comme source de données, voir
    # `trading_bot.execution.manual_broker` et `ManualBrokerConfig`.
    broker: str = "alpaca"
    manual: ManualBrokerConfig = field(default_factory=ManualBrokerConfig)
    # Base persistante de performance réelle par symbole (voir
    # `trading_bot.portfolio.symbol_track_record`), utilisée par une
    # stratégie dont `universe_rotation.metric == "track_record"`. Partagée
    # entre backtest et live (même fichier) pour que l'algorithme
    # s'améliore avec le temps sur les DEUX : chaque cycle live y ajoute des
    # trades réellement nouveaux, un backtest peut la lire (et, avec
    # `--update-track-record`, y écrire) mais sans apporter d'information
    # nouvelle en rejouant une période déjà connue — voir la docstring du
    # module pour cet avertissement.
    track_record_file: str = "state/symbol_track_record.json"


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
class OptimizationConfig:
    """Grilles de paramètres pour la recherche par grille (voir
    `trading_bot.backtest.optimizer`), utilisées par `python -m trading_bot
    optimize` et par `walk-forward --optimize`.

    `grids` : {nom_stratégie: {param: [valeurs...]}}. Une stratégie absente
    d'ici garde ses paramètres tels que définis dans `strategies:` plus haut
    (elle n'est simplement pas grillée). Vide par défaut (rétrocompatible :
    aucun effet tant que la section n'est pas renseignée).
    """

    metric: str = "sharpe_ratio"
    grids: dict[str, dict[str, list[Any]]] = field(default_factory=dict)


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
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

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
            universe_rotation=UniverseRotationConfig(**(s.get("universe_rotation", {}) or {})),
        )
        for s in raw.get("strategies", [])
    ]

    market_raw = dict(raw["market"])
    regime_raw = market_raw.pop("regime_filter", None) or {}
    volatility_raw = market_raw.pop("volatility_filter", None) or {}
    news_sentiment_raw = raw.get("news_sentiment", None) or {}
    optimization_raw = raw.get("optimization", None) or {}

    live_raw = dict(raw["live"])
    manual_raw = live_raw.pop("manual", None) or {}

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
        live=LiveConfig(**live_raw, manual=ManualBrokerConfig(**manual_raw)),
        news_sentiment=NewsSentimentConfig(**news_sentiment_raw),
        optimization=OptimizationConfig(**optimization_raw),
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

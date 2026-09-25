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
    # Type de stop suiveur (voir `trading_bot.portfolio.stops`) :
    #   - "atr" (défaut, inchangé) : stop = clôture - `atr_stop_multiple` x
    #     ATR, recalculé et "ratcheté" à chaque pas de temps sur la clôture ;
    #   - "trailing_pct" : sémantique d'un ordre "Stop Suiveur" natif de
    #     courtier (ex: Fortuneo) — écart en % FIGÉ à l'entrée
    #     (`trailing_stop_pct`, ou à défaut `atr_stop_multiple` x ATR / prix
    #     au moment de l'entrée), stop = plus haut atteint depuis l'entrée x
    #     (1 - écart), le courtier le remontant en continu ;
    #   - "none" : aucun stop (ex: buy & hold pur).
    stop_mode: str = "atr"
    trailing_stop_pct: float | None = None
    # Après une sortie sur stop, quand ré-autoriser une entrée sur ce symbole :
    #   - "immediate" (défaut, inchangé) : dès le signal suivant ;
    #   - "sma" : seulement quand la clôture repasse au-dessus de sa SMA
    #     `stop_reentry_sma_window` (évite, pour une stratégie toujours
    #     investie comme `buy_and_hold`, de racheter dès le lendemain d'un
    #     stop, ce qui rendrait le stop inutile tout en payant les frais).
    stop_reentry: str = "immediate"
    stop_reentry_sma_window: int = 200


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
    # Barème de frais par paliers (voir `trading_bot.portfolio.fees`),
    # prioritaire sur `commission_pct` quand renseigné. Format :
    #   {"tiers": [{"up_to": 500, "pct": 0.005, "min": 1.95},
    #              {"up_to": 2000, "fixed": 1.95},
    #              {"up_to": null, "pct": 0.002}],
    #    "minimum": 0.0}
    # None (défaut) : `commission_pct` proportionnel pur (inchangé).
    # Utilisé AUSSI en live (broker manuel : frais estimés affichés et
    # déduits du cash, et garde-fou `max_fee_pct`) pour garder la parité
    # backtest/live.
    commission_schedule: dict[str, Any] | None = None
    # Actions entières uniquement (PEA) : quantités arrondies vers le bas, et
    # achats plafonnés au cash disponible frais compris (pas de marge sur un
    # PEA). False (défaut) : quantités fractionnaires, comportement inchangé.
    whole_shares: bool = False
    # Garde-fous anti-frais (backtest ET live), jamais appliqués à une
    # clôture complète de position (une sortie décidée par la stratégie doit
    # toujours passer) : ordre ignoré si sa valeur < `min_order_value`, ou si
    # ses frais dépassent `max_fee_pct` de sa valeur, ou (ajustement d'une
    # position existante, pas une entrée) si l'écart de poids cible/actuel
    # est < `rebalance_tolerance_pct` — évite les micro-rééquilibrages dus à
    # la simple dérive des prix.
    min_order_value: float = 0.0
    max_fee_pct: float | None = None
    rebalance_tolerance_pct: float = 0.0
    # Jours calendaires d'historique téléchargés AVANT `start_date` pour
    # "chauffer" les indicateurs (SMA200, momentum 12 mois...) : sans ça, une
    # stratégie à long lookback reste inactive la première année du
    # backtest. 0 (défaut) : comportement inchangé.
    warmup_days: int = 0
    # Apport mensuel simulé (en €), versé en cash à l'ouverture du premier
    # jour de bourse de chaque mois (pas le mois de départ). Uniquement
    # supporté par le mode `core_satellite` (voir `trading_bot.backtest.
    # core_satellite`), qui calcule alors un rendement pondéré par le temps
    # (NAV par part, hors effet des apports) ET un TRI (rendement pondéré
    # par l'argent). 0 (défaut) : aucun apport, comportement inchangé.
    monthly_contribution: float = 0.0


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
    # Sémantique "Stop Suiveur" natif du courtier (ex: Fortuneo) : l'écart
    # est figé en % à l'entrée et c'est le COURTIER qui remonte le stop en
    # continu. Le bot ne notifie alors qu'à la pose (entrée), à l'annulation
    # (sortie/changement de quantité) et quand le stop a vraisemblablement
    # été déclenché — jamais "remplace ton stop" à chaque cycle. Implique
    # `risk.stop_mode: trailing_pct` (un stop "atr" est alors traité comme
    # tel, écart = `atr_stop_multiple` x ATR / prix à l'entrée).
    native_trailing_stop: bool = False
    # Écart proposé pour un ordre à cours limité alternatif à l'ordre au
    # marché (achat : dernier cours + écart ; vente : dernier cours - écart).
    limit_offset_pct: float = 0.005


@dataclass
class AlertsConfig:
    """Alertes d'INFORMATION (jamais des ordres), ex: mode buy & hold sans
    stop : push quand `symbol` clôture sous sa SMA `sma_window`, ou quand
    l'equity du compte passe sous `drawdown_pct` de son plus haut. Une seule
    notification par franchissement (et une au retour), pas à chaque cycle."""

    enabled: bool = False
    symbol: str = "PSP5.PA"
    sma_window: int = 200
    drawdown_pct: float = 0.20
    # Mode `core_satellite` uniquement (voir `trading_bot.live.engine`) :
    # paliers de drawdown du portefeuille (NAV par part, donc hors effet des
    # apports), chacun notifié une seule fois au franchissement, avec un
    # rappel du plan ("ne vends pas, c'est prévu") ; et paliers de baisse
    # d'une poche depuis son plus haut, ex {"CL2.PA": [0.30, 0.50]}. Vides
    # (défaut) : rien de plus que les alertes ci-dessus.
    drawdown_levels: list[float] = field(default_factory=list)
    symbol_drop_levels: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class PushoverConfig:
    """Notifications push via Pushover (https://pushover.net/api, voir
    `trading_bot.notify.pushover`), surtout utiles en mode `live.broker:
    "manual"` : un seul push par cycle récapitule les ordres/stops à passer
    à la main sur le courtier, plus une alerte en cas de crash de cycle ou de
    déclenchement d'un coupe-circuit.

    Désactivé par défaut (rétrocompatible). Les identifiants ne sont JAMAIS
    lus depuis le YAML : uniquement depuis les variables d'environnement
    `PUSHOVER_APP_TOKEN` / `PUSHOVER_USER_KEY` (éventuellement via `.env`).
    """

    enabled: bool = False
    # Priorité Pushover (-2 à 2) des récapitulatifs d'ordres ; `alert_priority`
    # pour les alertes (crash, coupe-circuit). 2 = "emergency" (répété jusqu'à
    # acquittement), voir la doc Pushover.
    priority: int = 0
    alert_priority: int = 1
    title: str = "Trading Bot"


@dataclass
class MorningBriefConfig:
    """Aperçu du matin (voir `trading_bot.live.morning_brief`) : UN push
    concis avant l'ouverture — valeur du portefeuille, poids vs cibles,
    ordres du cycle de la veille à passer, cash, contexte de marché. Lecture
    seule : n'envoie ni ne modifie aucun ordre, ne touche ni au fichier de
    compte ni à l'état du plan. Exécuté par le même process que le cycle du
    soir (`live.daily_run_after` requis) ; mode `core_satellite` + broker
    manuel uniquement.

    Désactivé par défaut (rétrocompatible)."""

    enabled: bool = False
    # Heure locale d'envoi (fuseau du calendrier de marché, Europe/Paris pour XPAR).
    at: str = "08:30"
    # False : envoyé aussi les jours de fermeture (il le dit alors).
    only_trading_days: bool = True
    # Priorité Pushover : -1 = silencieux (pas de son ni de vibration).
    priority: int = -1
    # None : titre des notifications + " — aperçu du matin".
    title: str | None = None
    # Bot (re)démarré après `at` sans aperçu envoyé ce jour-là : envoyé tout
    # de suite s'il est encore avant cette heure locale, sinon sauté.
    catch_up_until: str = "12:00"
    # Libellés courts des positions hors plan (ex. {"CS.PA": "AXA"}), affichés
    # dans l'aperçu ; à défaut, le ticker sans suffixe de place ("CS").
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class NotificationsConfig:
    pushover: PushoverConfig = field(default_factory=PushoverConfig)


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
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    # Heure locale (fuseau du calendrier de marché, ex "18:30" à Paris pour
    # XPAR) à partir de laquelle exécuter UN SEUL cycle par jour de bourse,
    # après la clôture : signaux sur la clôture définitive du jour, ordres à
    # passer le lendemain à l'ouverture (même convention que le backtest).
    # Remplace alors `loop_interval_seconds`/`trade_only_when_market_open`.
    # None (défaut) : boucle à intervalle fixe, inchangée.
    daily_run_after: str | None = None
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    morning_brief: MorningBriefConfig = field(default_factory=MorningBriefConfig)


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
    notifications_raw = live_raw.pop("notifications", None) or {}
    pushover_raw = notifications_raw.get("pushover", None) or {}
    alerts_raw = live_raw.pop("alerts", None) or {}
    morning_brief_raw = live_raw.pop("morning_brief", None) or {}

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
        live=LiveConfig(
            **live_raw,
            manual=ManualBrokerConfig(**manual_raw),
            notifications=NotificationsConfig(pushover=PushoverConfig(**pushover_raw)),
            alerts=AlertsConfig(**alerts_raw),
            morning_brief=MorningBriefConfig(**morning_brief_raw),
        ),
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

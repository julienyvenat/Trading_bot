# Trading Bot — bot de trading automatique multi-stratégies

Application Python de trading automatique pour actions et ETF, conçue pour
démarrer en **paper trading** (simulation, aucun argent réel) via
[Alpaca](https://alpaca.markets/), avec un moteur de **backtest** qui réutilise
exactement les mêmes briques que le trading live.

## Fonctionnalités

- **Multi-stratégies** : plusieurs stratégies tournent en parallèle sur les
  mêmes symboles, chacune avec un poids configurable ; leurs signaux sont
  combinés par un allocateur.
  - `sma_crossover` : suivi de tendance (croisement de moyennes mobiles)
  - `rsi_mean_reversion` : retour à la moyenne (RSI survente/surachat), avec
    un filtre de tendance optionnel (`trend_filter_window`, SMA 200 par
    défaut) qui n'achète un creux de RSI que si le prix est déjà au-dessus de
    cette moyenne longue — pour éviter d'acheter des creux en pleine tendance
    baissière ("couteau qui tombe")
  - `momentum_breakout` : breakout de momentum (canal de Donchian)
  - `relative_strength` : rotation sectorielle / force relative — classe les
    symboles de l'univers *entre eux* (plutôt que dans l'absolu) et ne reste
    investi que sur les plus forts, pour une vraie diversification de
    mécanisme par rapport aux trois stratégies ci-dessus.
  - `defensive_rotation` : rotation défensive — passe sur un actif peu
    corrélé aux actions (or via GLD par défaut) quand le marché large est en
    tendance baissière, pour une diversification de classe d'actif plutôt que
    de signal.
  - `bollinger_scalping` : retour à la moyenne sur bandes de Bollinger,
    pensée pour de l'**intraday** (bougies 5 min...) — voir
    [Intraday (ex-scalping)](#intraday-ex-scalping) : testée sans edge
    démontrable sur l'univers volatil de l'exemple fourni, désactivée par
    défaut dans cet exemple, au profit de `momentum_breakout` reparamétrée
    en bougies 1h.
  - Facile d'en ajouter de nouvelles (voir [Ajouter une stratégie](#ajouter-une-stratégie)).
- **Granularité configurable** (`universe.timeframe`) : quotidien par défaut
  (`1Day`), ou intraday (`1Hour`, `30Min`, `15Min`, `5Min`, `1Min`) — le même
  moteur event-driven tourne à l'identique quelle que soit la granularité
  (voir [Intraday (ex-scalping)](#intraday-ex-scalping)).
- **Gestion du risque** :
  - Dimensionnement des positions basé sur l'ATR (risque max par trade), poids
    max par position, exposition brute max du portefeuille, nombre max de
    positions ouvertes.
  - **Stop-loss suiveur (trailing ATR)** sur chaque position : le stop ne se
    déplace jamais en défaveur de la position, il ne fait que "ratchet" dans
    le sens favorable au fur et à mesure qu'elle progresse. En live, il est
    délégué à un **ordre stop natif posé chez le broker** (réagit en continu),
    plutôt que vérifié uniquement à chaque cycle côté bot.
  - **Filtre de régime de marché** : réduit (ou coupe) l'exposition du
    portefeuille quand un indice de référence (SPY par défaut) est en
    tendance baissière (sous sa SMA 200), pour limiter les pertes en marché
    baissier généralisé — le bot est long-only et sans ce filtre rien ne
    réduit l'exposition dans ce cas. Voir `config.yaml -> market.regime_filter`.
  - **Filtre de volatilité** : réduit l'exposition quand un proxy négociable
    de la volatilité de marché (VIXY par défaut) s'envole au-dessus de sa
    moyenne mobile récente, signe d'un pic de stress — complémentaire au
    filtre de régime (celui-ci réagit à la direction du marché, celui-là à
    l'amplitude des mouvements récents, un choc pouvant survenir même en
    tendance haussière). Désactivé par défaut, voir
    `config.yaml -> market.volatility_filter`.
  - **Filtre de sentiment de news** (live uniquement) : bloque les nouvelles
    entrées sur un symbole dont les news récentes — récupérées via l'API News
    officielle d'Alpaca, pas de scraping de forums/réseaux sociaux — sont
    majoritairement négatives (score sous un seuil configurable, mots-clés
    transparents et auditables plutôt qu'un modèle boîte noire). Désactivé
    par défaut, voir `config.yaml -> news_sentiment`.
  - **Coupe-circuit de perte journalière** : bloque toute nouvelle entrée
    (ou augmentation de position) pour le reste de la séance si la perte du
    jour dépasse un seuil configurable ; se réinitialise à la séance suivante.
  - **Coupe-circuit de drawdown** : liquide tout le portefeuille et arrête le
    bot si l'equity chute de plus d'un certain pourcentage depuis son plus
    haut historique ; ne se réinitialise **jamais** tout seul (reprise
    manuelle obligatoire, voir [Reprise après coupe-circuit](#reprise-après-coupe-circuit-de-drawdown)).
- **Calendrier de marché réel** (NYSE par défaut, via `pandas-market-calendars`) :
  jours fériés et fermetures anticipées gérés nativement, aucune nouvelle
  position ouverte dans les dernières minutes avant la clôture, et le moteur
  live dort intelligemment jusqu'à la prochaine séance plutôt que de sonder en
  boucle.
- **Backtest** event-driven sur données historiques (via `yfinance`), avec
  courbe d'equity et métriques (rendement, volatilité, Sharpe, **Sortino**,
  **Calmar**, **profit factor**, max drawdown, taux de jours positifs, et des
  **statistiques par trade** — nombre de trades, taux de trades gagnants,
  gain/perte moyens, ratio gain/perte) — et applique exactement les mêmes
  stops et coupe-circuits que le live. Le signal calculé à la clôture du jour
  J s'exécute à l'**ouverture du jour suivant** (J+1), plus réaliste qu'une
  exécution immédiate à la clôture.
- **Recherche par grille (grid search)** et **walk-forward avec
  ré-optimisation** : teste plusieurs combinaisons de paramètres de
  stratégie via le même moteur event-driven (aucune approximation
  vectorisée), en parallèle sur plusieurs cœurs. Intégré au walk-forward
  pour choisir les paramètres sur chaque fenêtre d'ENTRAÎNEMENT puis les
  valider hors échantillon sur la fenêtre de TEST correspondante — jamais
  l'inverse. Voir [Optimisation de paramètres](#optimisation-de-paramètres)
  et [Walk-forward](#walk-forward).
- **Rapport de backtest HTML** autonome (courbe d'equity, drawdown,
  distribution du P&L par trade) — voir [Backtest](#backtest).
- **Paper trading** en continu via l'API Alpaca (compte de simulation gratuit),
  avec bascule facile vers un compte réel (à vos risques). L'état du bot
  (stops en cours, ordres stop natifs, coupe-circuits) est persisté sur disque
  entre deux redémarrages.
- Architecture modulaire : le code de stratégie/risque/allocation/stops/
  coupe-circuits est strictement identique entre backtest et live, pour
  éviter les écarts de comportement entre "ce qui est testé" et "ce qui est
  tradé".

## Avertissement

Ce projet est un point de départ technique, **pas un conseil en investissement**.
Le trading automatisé comporte un risque de perte en capital. Commence
systématiquement en paper trading, valide ta stratégie sur une longue période
avant d'envisager un compte réel, et ne trade jamais plus que ce que tu peux
te permettre de perdre.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Pour générer des rapports HTML de backtest (`--report`), installe en plus
l'extra dédié (matplotlib) :

```bash
pip install -e ".[report]"
```

Configure ensuite tes identifiants Alpaca (nécessaires uniquement pour le
paper/live trading, pas pour le backtest) :

```bash
cp .env.example .env
# édite .env et renseigne ALPACA_API_KEY / ALPACA_SECRET_KEY
# (clés "paper" gratuites sur https://app.alpaca.markets/paper/dashboard/overview)
```

## Configuration

Tout se configure dans [`config/config.yaml`](config/config.yaml) : symboles
suivis, stratégies actives et leurs paramètres/poids, règles de gestion du
risque, paramètres de backtest et du moteur live. Voir les commentaires dans
le fichier pour le détail de chaque option.

## Utilisation

### Backtest

```bash
python -m trading_bot backtest
# en sauvegardant la courbe d'equity en CSV :
python -m trading_bot backtest -o equity_curve.csv
# en générant un rapport HTML autonome (nécessite `pip install -e ".[report]"`) :
python -m trading_bot backtest --report rapport.html
```

Affiche les métriques de performance (rendement total/annualisé, volatilité,
Sharpe, Sortino, Calmar, profit factor, max drawdown, taux de jours positifs,
statistiques par trade), le nombre de sorties déclenchées par le stop
suiveur, et signale si un coupe-circuit s'est déclenché pendant la période
testée. `--report` génère un fichier HTML unique (courbe d'equity, courbe de
drawdown, histogramme du P&L par trade), consultable hors-ligne dans
n'importe quel navigateur.

### Walk-forward

```bash
python -m trading_bot walk-forward
# fenêtres personnalisées (en jours calendaires) :
python -m trading_bot walk-forward --train-days 365 --test-days 90
# avec ré-optimisation des paramètres sur chaque fenêtre d'entraînement
# (nécessite au moins une grille dans config.yaml -> optimization.grids) :
python -m trading_bot walk-forward --optimize
```

Découpe l'historique en fenêtres glissantes entraînement/test. Sans
`--optimize`, ré-exécute la même config (paramètres fixes) sur chaque
sous-période, pour vérifier que la performance tient dans le temps plutôt
que sur une seule période choisie. Avec `--optimize`, cherche en plus les
meilleurs paramètres sur chaque fenêtre d'ENTRAÎNEMENT (voir
[Optimisation de paramètres](#optimisation-de-paramètres)) puis les valide
hors échantillon sur la fenêtre de TEST correspondante — jamais l'inverse,
pour ne jamais laisser d'information du futur influencer le choix des
paramètres. Affiche le détail par fenêtre (y compris les paramètres retenus)
ainsi que les métriques cumulées sur toutes les fenêtres de test (hors
échantillon uniquement). Voir la docstring de
`trading_bot.backtest.walk_forward` pour le détail de ce que ça valide (et
ne valide pas — voir aussi [Limites connues](#limites-connues--pistes-damélioration)).

### Optimisation de paramètres

```bash
python -m trading_bot optimize
# métrique et parallélisme personnalisés :
python -m trading_bot optimize --metric calmar_ratio --max-workers 4 --top 5 -o resultats.csv
```

Teste toutes les combinaisons de paramètres définies dans `config.yaml ->
optimization.grids` (produit cartésien, à définir avant de lancer la
commande — vide par défaut), en lançant le **même moteur event-driven** que
le reste du bot pour chaque combinaison, en parallèle sur plusieurs
processus. Affiche les meilleures combinaisons triées par la métrique
choisie (`optimization.metric`, ex: `sharpe_ratio`, `calmar_ratio`,
`total_return_pct`...) ; `-o` sauvegarde tous les résultats en CSV.

⚠️ Le nombre de runs est le produit du nombre de valeurs de chaque paramètre
grillé, multiplié entre stratégies : commence avec de petites grilles avant
d'en tester des grandes. Cette recherche optimise sur UNE période fixe (pas
de garantie hors échantillon) — valide toujours la combinaison retenue via
`walk-forward --optimize` avant de t'y fier.

### Intraday (ex-scalping)

Le bot supporte des bougies intraday (`universe.timeframe: "5Min"`, `"15Min"`,
`"1Hour"`...) en plus du quotidien par défaut — même moteur event-driven,
mêmes commandes (`backtest`, `optimize`, `walk-forward`, `paper`), juste une
granularité différente. Un exemple complet est fourni :
[`config/config_intraday.example.yaml`](config/config_intraday.example.yaml)
(stratégie `momentum_breakout` reparamétrée en bougies 1h, source de données
Alpaca pour un historique intraday profond, rotation d'univers par momentum,
grille d'optimisation pré-remplie).

Cette config a d'abord visé du SCALPING (bougies 5 min) avec une stratégie de
retour à la moyenne (`bollinger_scalping`) : **validé par walk-forward sur 15
mois d'historique réel, ni le retour à la moyenne ni un breakout momentum au
même grain n'ont montré d'edge hors échantillon** (cumulé -64% et -66%
respectivement, 5Min *plus bruité et plus coûteux en trades* que 15Min qui a
fait pire encore). Reparamétré en bougies 1h, le breakout momentum valide
positivement (+11.5% cumulé hors échantillon, Sharpe 0.16, 15/24 fenêtres de
test positives) : le problème n'était pas la logique de stratégie, mais le
grain temporel — trop bruité/coûteux en trades à 5min et 15min sur cet
univers. Ce n'est donc plus vraiment du "scalping" (positions de quelques
heures à quelques jours, pas quelques minutes), d'où le renommage. Sharpe 0.16
reste modeste (à comparer au 1.29 du swing quotidien, voir plus haut) : à
traiter comme une piste à surveiller, pas une stratégie mûre pour du capital
réel.

```bash
cp config/config_intraday.example.yaml config/config_intraday.yaml

python -m trading_bot backtest --config config/config_intraday.yaml
python -m trading_bot optimize --config config/config_intraday.yaml
python -m trading_bot walk-forward --config config/config_intraday.yaml --train-days 90 --test-days 30 --optimize
```

⚠️ À lire avant d'aller plus loin (détaillé en commentaire dans le fichier
d'exemple) :
- `backtest.data_source: "alpaca"` : yfinance ne garantit qu'un historique
  intraday limité (~60 jours glissants en dessous de 1Hour, 7 jours en
  1Min) — bien trop peu pour un walk-forward avec assez de fenêtres hors
  échantillon. La source Alpaca (`trading_bot.data.market_data.
  fetch_historical_bars`) conserve plusieurs années d'historique intraday
  même sur un compte gratuit/paper, mais nécessite des identifiants Alpaca
  valides dans `.env`, y compris pour juste backtester.
- `backtest.commission_pct` (5 bps) sous-estime probablement les coûts
  réels : un test à commission nulle a fait passer le taux de trades
  gagnants de 64% à 71% et le profit factor de 0.84 à 1.18 sur la version
  5min — les coûts réels (spread bid-ask, slippage) pèsent significativement
  plus qu'un simple pourcentage de commission ne le capture.
- Contrainte réglementaire US : un compte sur marge de moins de 25 000 $
  qui fait 4 day trades ou plus en 5 jours ouvrés est classé *Pattern Day
  Trader* et se retrouve bloqué par le broker — moins critique qu'en
  scalping 5min (positions tenues plus longtemps), mais toujours à vérifier.
- Les autres stratégies (`sma_crossover`, `rsi_mean_reversion`) sont
  désactivées dans l'exemple : leurs fenêtres ont été calibrées en JOURS
  pour du swing, pas en bougies pour de l'intraday — les réactiver sans les
  retuner via `optimize` n'a pas de sens.

### Cotation de confiance par symbole (track record)

En plus des métriques techniques de `universe_rotation` (volatilité,
momentum, liquidité — voir [Intraday](#intraday-ex-scalping)), un candidat
peut être classé sur son **vécu réel avec ce bot** plutôt que sur un proxy
de prix : `metric: "track_record"`. Contrairement aux autres métriques
(recalculées à froid depuis l'historique de prix à chaque run), celle-ci
s'appuie sur une base persistante
([`trading_bot.portfolio.symbol_track_record`](src/trading_bot/portfolio/symbol_track_record.py))
qui accumule les trades réalisés d'un run à l'autre.

```bash
python -m trading_bot backtest --config config/config_intraday.yaml --update-track-record
```

`--update-track-record` enregistre les trades de ce backtest dans
`live.track_record_file` (`state/symbol_track_record.json` par défaut). Le
**paper/live trading alimente aussi cette même base automatiquement**, à
chaque cycle (`trading_bot.live.trade_realization.detect_realized_trades`,
voir `trading_bot.live.engine.run_once`) : contrairement au backtest, une
séance de trading réelle est TOUJOURS une période réellement nouvelle, donc
pas besoin de flag `--update-track-record` côté `paper`, ni de risque de
gonfler artificiellement les stats en rejouant la même période. Le prix
d'entrée utilisé est celui déjà calculé par le broker (`avg_entry_price`),
pas reconstruit à la main — voir les limites précises dans la docstring de
`trading_bot.live.trade_realization` (prix de sortie approximé par le
dernier prix connu, pas le fill exact du broker).

Le score utilisé pour classer les candidats est un gain moyen par trade
(`expectancy_score`), atténué vers 0 tant que l'échantillon est petit
(shrinkage bayésien simple) pour qu'un symbole avec 2 trades gagnants sur 2
ne paraisse pas déjà "excellent" ; un candidat encore inconnu de la base
n'est jamais sélectionné (confiance `NaN`, pas un score neutre par défaut).

⚠️ Deux limites importantes à connaître avant d'utiliser cette métrique :
- **L'accumulation n'apporte de l'information nouvelle que sur des
  périodes réellement nouvelles.** Rejouer le même backtest en boucle ne
  fait rien gagner (idempotent, voir la marque d'eau `last_exit_date` par
  symbole) ; l'enchaînement de fenêtres non recouvrantes (walk-forward, ou
  paper trading jour après jour) est ce qui fait vraiment grandir la base.
- Non encore validée par `walk-forward` sur cette métrique précise — comme
  toute nouvelle piste dans ce projet, à tester avant de lui faire
  remplacer les métriques techniques existantes.

### Paper trading

```bash
# Un seul cycle (utile pour tester la configuration)
python -m trading_bot paper --once --dry-run

# Boucle continue (respecte les horaires de marché, voir config.yaml -> live)
python -m trading_bot paper
```

`--dry-run` calcule les ordres sans les envoyer au broker — pratique pour
vérifier le comportement du bot avant de le laisser tourner pour de vrai (même
en paper trading).

⚠️ Pour trader en argent réel, passe `ALPACA_PAPER=false` dans `.env` **après**
avoir validé ta stratégie en paper trading pendant une durée significative.

### Mode manuel (PEA, ou tout courtier sans API — ex: Fortuneo)

Fortuneo, comme la quasi-totalité des courtiers PEA en France, n'expose aucune
API de passage d'ordres : le bot ne peut donc pas y trader automatiquement.
`live.broker: "manual"` (voir `config/config_pea_fortuneo.example.yaml` et
`trading_bot.execution.manual_broker`) permet quand même de l'utiliser en
semi-automatique : le bot calcule les ordres et les stops suiveurs comme
d'habitude, mais se contente de les **afficher** à exécuter toi-même sur le
site du courtier — jamais envoyés automatiquement. Dans ce mode, les cours
viennent de yfinance (Alpaca ne couvre pas les actions européennes), donc
aucune clé API n'est nécessaire.

```bash
# Avant le premier cycle : crée le fichier de compte avec le solde de cash
# et les positions réelles de ton PEA (voir live.manual.account_file dans la
# config), ex :
echo '{"cash": 5000.0, "positions": {}}' > state/manual_account_fortuneo.json

python -m trading_bot paper --config config/config_pea_fortuneo.example.yaml --once
```

Après chaque ordre affiché, le bot met à jour `account_file` en supposant
qu'il a été exécuté au prix affiché — corrige ce fichier à la main si le fill
réel diffère (prix, ou ordre pas encore passé) avant le prochain cycle.
Contrairement à Alpaca, les quantités sont arrondies à l'action entière (pas
de fractionnaire sur un PEA).

Univers PEA : SPY n'étant pas éligible, il est remplacé par **PSP5**
(`PSP5.PA`, Amundi PEA S&P 500 UCITS ETF, Euronext Paris), à la fois dans
l'univers tradé et comme référence du `market.regime_filter`. GLD et VIXY,
sans équivalent éligible PEA, sont exclus : pas de `defensive_rotation`, et
`market.volatility_filter` désactivé.

#### Notifications Pushover (ordres à passer sur ton téléphone)

En mode manuel, plutôt que de surveiller les logs, le bot peut t'envoyer via
[Pushover](https://pushover.net) **un seul push par cycle** récapitulant tous
les ordres et stops à passer sur ton courtier (titre du type « Trading Bot PEA
— 3 ordre(s) à passer », message tronqué proprement à 1024 caractères ; aucun
push si rien à faire), ainsi qu'une alerte quand un cycle crashe (première
erreur d'une série seulement) ou qu'un coupe-circuit se déclenche.

1. Crée un compte sur https://pushover.net, installe l'app Pushover sur ton
   téléphone et note ta **User Key** (tableau de bord).
2. Crée une application (« Create an Application/API Token ») et note son
   **API Token**.
3. Renseigne-les dans `.env` (jamais dans la config YAML ni dans le dépôt) :

   ```bash
   PUSHOVER_APP_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   PUSHOVER_USER_KEY=yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
   ```

4. Active `live.notifications.pushover.enabled: true` (déjà le cas dans
   `config/config_pea_fortuneo.example.yaml`) et vérifie :

   ```bash
   python -m trading_bot notify-test --config config/config_pea_fortuneo.example.yaml
   ```

Une panne de Pushover (réseau, identifiants invalides ou absents) ne fait
jamais échouer le cycle de trading : elle est simplement journalisée en
avertissement. Désactivé par défaut dans les autres configs.

#### PEA Fortuneo v2 : buy & hold PSP5 ou rotation mensuelle d'ETF

Constat de départ : la config active `config_pea_fortuneo.example.yaml`
(actions du CAC 40 + 3 stratégies journalières, boucle horaire) perd de
l'argent une fois les frais réels comptés — backtest 2018→09/2026 avec
2 306,59 €, barème Fortuneo et actions entières : **-19,8 %** (≈ 28 ordres/an,
480 € de frais, coupe-circuit de drawdown -20 % déclenché en route), contre **+237,5 %** pour un simple achat de PSP5. Deux modes
à faible rotation la remplacent :

| Mode | Config | Principe | Ordres à passer |
|---|---|---|---|
| (a) Buy & hold | `config/config_pea_buyhold.example.yaml` | 100 % PSP5 en actions entières, jamais revendu ; pas de stop, alertes d'information seulement (PSP5 sous sa SMA200, compte à -20 % de son plus haut) | 1 à l'entrée, puis ~0 |
| (b) Dual momentum | `config/config_pea_etf_momentum.example.yaml` | 1 fois par mois : l'ETF au meilleur rendement 12 mois parmi PSP5 / PUST / CAC / ETZ / PAEEM, s'il est positif ; sinon cash | ~6/an |

**Hypothèses de frais** (`backtest.commission_schedule`, voir
`trading_bot.portfolio.fees`, utilisées aussi en live) : barème standard
Fortuneo sur Euronext — ordre ≤ 500 € : 0,50 % (min 1,95 €) ; 500 à 2 000 € :
1,95 € ; > 2 000 € : 0,20 %. La promo « frais d'achat remboursés sur une
sélection d'ETF Amundi (ordres de 500 à 100 000 €) jusqu'au 31/12/2026 »
n'est **pas** modélisée (prudence). `backtest.whole_shares: true` : actions
entières, achats plafonnés au cash frais compris ; `min_order_value: 150`,
`max_fee_pct: 0.02` et `rebalance_tolerance_pct: 0.10` écartent les ordres
qui ne valent pas leurs frais (une clôture complète passe toujours).
Exécution à l'ouverture du lendemain du signal, comme le reste du backtest.

**Résultats** (2018-01-01 → 2026-09-24, 2 306,59 € de départ, reproduire
avec `python scripts/pea_compare.py`). (a) — choix du stop :

Chaque cellule : CAGR / Max DD / Sharpe / Calmar. Frais et ordres/an sur la
période complète.

| Variante | 2018→2026 | En échantillon 2018-2022 | Hors échantillon 2023→ | Ordres/an | Frais |
|---|---|---|---|---|---|
| **Buy & hold PSP5, sans stop** | +14,7 % / -33,6 % / **0,91** / 0,44 | +11,3 % / -33,6 % / **0,67** / 0,34 | +18,8 % / -23,1 % / 1,33 / 0,81 | 0,1 | 4,59 € |
| Stop suiveur ATR×4 (écart figé à l'entrée) | +8,5 % / -26,1 % / 0,69 / 0,33 | +3,8 % / -26,1 % / 0,33 / 0,15 | +16,2 % / -13,4 % / 1,41 / 1,21 | 7,6 | 382 € |
| Stop suiveur ATR×6 | +9,3 % / -23,8 % / 0,71 / 0,39 | +5,4 % / -23,8 % / 0,43 / 0,23 | +14,0 % / -18,6 % / 1,13 / 0,76 | 2,8 | 158 € |
| Stop suiveur ATR×8 | +9,9 % / -25,2 % / 0,72 / 0,39 | +5,8 % / -25,2 % / 0,43 / 0,23 | +17,1 % / -14,1 % / 1,42 / 1,21 | 1,2 | 58 € |
| Stop suiveur 15 % | +7,8 % / -30,2 % / 0,59 / 0,26 | +2,6 % / -30,2 % / 0,24 / 0,09 | +16,5 % / -15,9 % / 1,33 / 1,04 | 1,5 | 69 € |
| Stop suiveur 20 % | +11,9 % / -21,3 % / 0,84 / 0,56 | +9,5 % / -21,3 % / 0,64 / 0,44 | +14,7 % / -20,8 % / 1,17 / 0,71 | 0,6 | 35 € |
| Stop suiveur 25 % | +10,0 % / -27,1 % / 0,71 / 0,37 | +7,8 % / -27,1 % / 0,54 / 0,29 | +12,4 % / -26,4 % / 0,97 / 0,47 | 0,6 | 31 € |

Les stops sont modélisés comme un Stop Suiveur Fortuneo : seuil = plus haut
depuis l'entrée × (1 - écart), exécuté au seuil, ou à l'ouverture en cas de
gap. On ne rachète que lorsque PSP5 repasse au-dessus de sa SMA200 (sinon on
rachèterait le lendemain).

Lecture honnête :
- **Sur toute la période et en échantillon**, tous les stops perdent face au
  buy & hold en CAGR (-3 à -7 pts/an) et en Sharpe. Seul le stop 20 %
  améliore le Calmar (0,56 contre 0,44), et il le doit au seul krach de
  mars 2020.
- **Hors échantillon (2023→)**, ATR×8 (DD -14,1 %, Sharpe 1,42) et 15 %
  (DD -15,9 %, Sharpe 1,33) divisent presque par deux le drawdown du buy &
  hold (-23,1 %), à Sharpe égal ou meilleur, pour un rendement annuel plus
  faible de -1,7 et -2,3 pts. ATR×4 fait pareil (-13,4 %, 1,41), mais passe
  7,6 ordres/an et paie 382 € de frais. Le stop 20 % réduit un peu le
  drawdown (-20,8 %) mais perd 4 pts de CAGR, et son Sharpe est plus bas (1,17).
- Même hors échantillon, tous les stops restent sous le buy & hold en CAGR.
  Les classements s'inversent d'une sous-période à
  l'autre, signe d'une forte dépendance à quelques épisodes.

**Défaut retenu : pas de stop** (meilleur rendement et meilleur Sharpe sur
la période complète), avec des alertes d'information uniquement. Un stop
suiveur large reste un choix légitime si tu préfères limiter le drawdown :
ATR×8 ou 15 % ont été les plus convaincants hors échantillon, au prix d'un
rendement plus faible. Pour l'activer : `risk.stop_mode: trailing_pct`,
avec `trailing_stop_pct: 0.15`, ou `null` avec `atr_stop_multiple: 8`.

(b) — petite grille (lookback 6/12 mois × top 1/2), rien d'autre d'optimisé :

| Variante | 2018→2026 CAGR / DD / Sharpe | En échantillon 2018-2022 | Hors échantillon 2023→ | Ordres/an | Frais |
|---|---|---|---|---|---|
| Référence buy & hold PSP5 | +14,7 % / -33,6 % / 0,91 | +11,3 % / -33,6 % / 0,67 | +18,8 % / -23,1 % / 1,33 | 0,1 | 5 € |
| **12 mois, top 1** (retenu sur 2018-2022) | +13,0 % / -31,5 % / 0,74 | +8,7 % / -29,6 % / 0,52 | +19,1 % / -24,6 % / 1,08 | 5,8 | 370 € |
| 12 mois, top 2 | +12,3 % / -30,6 % / 0,76 | +7,6 % / -30,6 % / 0,50 | +19,1 % / -25,2 % / 1,19 | 6,0 | 142 € |
| 6 mois, top 1 | +11,1 % / -28,1 % / 0,66 | +5,4 % / -28,1 % / 0,37 | +18,5 % / -26,5 % / 1,04 | 8,0 | 473 € |
| 6 mois, top 2 | +6,9 % / -30,5 % / 0,50 | -0,0 % / -30,5 % / 0,08 | +16,6 % / -24,3 % / 1,06 | 9,5 | 166 € |

Honnêtement : **(b) ne bat pas le buy & hold PSP5**, ni sur toute la période
ni hors échantillon en ajusté du risque (drawdown à peine réduit, Sharpe plus
faible, ~6 ordres/an à passer à la main). Biais à garder en tête : l'univers
contient le Nasdaq-100, choisi en sachant qu'il a été le grand gagnant de la
période ; PAEEM n'a d'historique yfinance que depuis 04/2019. Pas d'actif
« refuge » PEA satisfaisant en risk-off : les ETF monétaires (CSH, C3M, XEON)
ne sont pas éligibles et OBLI.PA (Amundi PEA Euro Court Terme) a perdu ~22 %
en 2021-2022 — la case vide reste donc en cash.

**Recommandation : mode (a).** Ce sont des backtests (une seule trajectoire
historique, dominée par le marché US 2018-2026) : ils ne garantissent rien,
et un buy & hold à 100 % actions implique d'accepter des baisses de -30 % ou
plus sans rien faire.

**Lancer** (créer d'abord le fichier de compte avec le cash réellement
disponible — hors lignes que le bot ne gère pas) :

```bash
cp config/config_pea_buyhold.example.yaml config/config_pea_buyhold.yaml
echo '{"cash": 2306.59, "positions": {}}' > state/manual_account_pea_buyhold.json
python -m trading_bot backtest --config config/config_pea_buyhold.yaml
python -m trading_bot paper --config config/config_pea_buyhold.yaml --once   # un cycle, pour voir
python -m trading_bot paper --config config/config_pea_buyhold.yaml          # 1 cycle / jour de bourse à 18h30
```

`live.daily_run_after: "18:30"` : un seul cycle par jour de bourse, après la
clôture d'Euronext (signal sur la clôture du jour, ordre à passer le
lendemain matin — même convention que le backtest). Exemple de premier push
(cours du 24/09/2026) :

```
PEA buy & hold — 1 ordre(s) à passer
ORDRE : ACHETER 38 PSP5 — ordre au marché (ou à cours limité 59,72 €) · ≈ 2 257,96 € au cours de 59,42 €, frais ≈ 4,52 €
INFO : Pas de stop à poser sur PSP5 : risk.stop_mode vaut none dans cette config (justification dans ses commentaires et dans le README). Alertes d'information actives (SMA / drawdown).
```

**Stop Suiveur natif** (`live.manual.native_trailing_stop: true` avec
`risk.stop_mode: trailing_pct`) : l'écart est figé en % à l'entrée
(`risk.trailing_stop_pct`, ou `atr_stop_multiple` × ATR / cours), le bot
demande de poser l'ordre **une seule fois** — « Une fois l'achat exécuté, poser un STOP SUIVEUR : vendre
38 PSP5, écart 20,0 % (≈ 11,88 €), seuil de départ 47,54 € » — puis ne
notifie plus rien tant que la position ne change pas (Fortuneo remonte le
seuil lui-même). Il rejoue les plus hauts/bas yfinance pour détecter un
déclenchement probable, comptabilise alors la vente et envoie une alerte « à
vérifier » ; avant toute vente décidée par la stratégie, il demande d'annuler
le stop. Pas d'ordre Duo/Trio : ces stratégies n'ont pas d'objectif de gain.

#### PEA cœur-satellite : plan passif 55 / 20 / 25, sans signal ni stop

> **Config réellement utilisée : `config/config_pea_fortuneo_80_20.yaml`** — même moteur, mais **80 % DCAM / 20 % PSP5, sans levier** (décision du 2026-09-25 : la poche CL2 n'ajoute qu'environ +0,3 pt/an sur 1990→2026 pour ~10 pts de drawdown en plus, cf. tableaux ci-dessous). Compte manuel : `state/manual_account_fortuneo.json` (cash seul, AXA détenues à côté non gérées). La variante 55/20/25 reste documentée ci-dessous et dans `config_pea_core_satellite.example.yaml`.

Config : `config/config_pea_core_satellite.example.yaml` (stratégie
`core_satellite`, logique dans `trading_bot.portfolio.core_satellite`,
partagée à l'identique par le backtest et le live).

| Poche | ETF (yfinance) | Cible | Rôle |
|---|---|---|---|
| Cœur mondial | `DCAM.PA` — Amundi PEA Monde (MSCI World) UCITS ETF | 55 % | diversification mondiale |
| Cœur US | `PSP5.PA` — Amundi PEA S&P 500 UCITS ETF Acc | 20 % | S&P 500 |
| Satellite | `CL2.PA` — Amundi MSCI USA Daily (2x) Leveraged UCITS ETF Acc | 25 % | levier x2 quotidien sur les actions US |

**Choix du MSCI World** (vérifié sur yfinance le 24/09/2026) : `DCAM.PA`
(Amundi, « PEA Monde » dans son nom, ~6,2 € la part, ~960 000 parts/jour
échangées). Écartés : `CW8.PA` (Amundi MSCI World Swap, ~698 € la part :
actions entières trop grossières sur 2 000 €), `EWLD.PA` (~41 €, mais
historique yfinance depuis 03/2024 seulement et ~16 000 parts/jour),
`WPEA.PA` (iShares, pas Amundi). Limite : DCAM n'a d'historique yfinance
que depuis le **04/03/2025**. L'éligibilité PEA se déduit du nom des fonds :
vérifie-la sur la fiche Fortuneo avant d'acheter.

**Règles** (paramètres sous `strategies[0].params`) :
- **Rien à faire** (aucun push) tant qu'aucune poche ne dérive de plus de
  `drift_threshold_pts` (5 points) de sa cible. Poids = valeur de la poche /
  valeur totale, cash compris.
- **Rééquilibrage annuel** au premier jour de bourse de
  `annual_rebalance_month` (janvier ; rattrapé si le bot était arrêté ce
  jour-là). On ne vend rien si l'écart restant est sous
  `calendar_min_drift_pts` (1 point).
- **Apports d'abord** : le cash disponible (au-delà d'un coussin de
  max(10 €, 1 % du cash)) va d'abord aux poches en retard, sans rien vendre.
  On ne vend les poches en excès que si la dérive dépasse encore le seuil
  après ces achats.
- Actions entières, frais Fortuneo, `min_order_value: 150` et
  `max_fee_pct: 0.02`. Une poche n'est **jamais achetée au-delà de son écart
  à la cible** (+ 1 action d'arrondi), ni si elle est déjà à sa cible. Parmi
  toutes les combinaisons d'achats finançables qui respectent ces règles, le
  bot retient celle qui laisse la plus petite dérive max, puis le plus petit
  écart total. Un plan qui ne réduit pas la dérive n'est jamais proposé.
  Si aucun achat valable n'existe (poches en retard de moins de 150 €), le
  cash attend : un push « cash en attente » l'annonce, une fois par montant.
  Même logique pour les ventes : sur ~2 300 €, une dérive corrigeable
  seulement par une vente de moins de 150 € attend le prochain apport.
- Pas de stop (`risk.stop_mode: none`, imposé dans ce mode). Pas de
  coupe-circuit.

**Déclarer un apport** : après le virement sur le PEA, ajoute le montant au
`"cash"` de `state/manual_account_pea_core_satellite.json`. Le bot détecte
l'écart au cycle suivant (« Apport détecté : +… € ») et pousse les achats
quand le cash non investi atteint max(`contribution_min_eur` 200 €,
`contribution_min_pct` 1 % du portefeuille). Sous ce seuil, le cash attend.
La performance est suivie **par parts** : un apport achète des parts, il ne
compte pas comme du rendement. C'est la base des alertes de drawdown.

**Lancer** :

```bash
cp config/config_pea_core_satellite.example.yaml config/config_pea_core_satellite.yaml
echo '{"cash": 2306.59, "positions": {}}' > state/manual_account_pea_core_satellite.json
python -m trading_bot paper --config config/config_pea_core_satellite.yaml --once   # un cycle, pour voir
python -m trading_bot paper --config config/config_pea_core_satellite.yaml          # 1 cycle / jour de bourse à 18h30
python scripts/pea_core_satellite_compare.py --cache /tmp/cs.pkl                    # tableaux ci-dessous
```

Premier push réel (`paper --once`, cours du 24/09/2026, 2 306,59 € de cash) :

```
PEA cœur-satellite — 3 ordre(s) à passer
INFO : Plan cœur-satellite : apport — 2 283,52 € de cash à investir (seuil 200,00 €), vers les poches en retard, par les achats seulement, aucune vente.
INFO : Poids : DCAM 0,0 % → 54,2 % (cible 55,0 %) · PSP5 0,0 % → 20,6 % (cible 20,0 %) · CL2 0,0 % → 24,0 % (cible 25,0 %) · cash 2 306,59 € → 27,76 €
ORDRE : ACHETER 200 DCAM — ordre au marché (ou à cours limité 6,26 €) · ≈ 1 246,60 € au cours de 6,23 €, frais ≈ 1,95 €
ORDRE : ACHETER 8 PSP5 — ordre au marché (ou à cours limité 59,61 €) · ≈ 474,48 € au cours de 59,31 €, frais ≈ 2,37 €
ORDRE : ACHETER 17 CL2 — ordre au marché (ou à cours limité 32,60 €) · ≈ 551,48 € au cours de 32,44 €, frais ≈ 1,95 €
INFO : Pas de stop à poser (DCAM, PSP5, CL2) : risk.stop_mode vaut none dans cette config. Des alertes d'information préviennent en cas de forte baisse, sans jamais vendre.
```

Dérive max après achat : 1,0 pt. Une part de PSP5 vaut 2,6 % du
portefeuille, d'où l'arrondi. Avec 20 000 € : 1 748 DCAM, 66 PSP5, 152 CL2
(54,6 / 19,6 / 24,7 %), 39,48 € de frais, 220 € laissés en coussin. Le cycle
suivant, sans changement, n'envoie rien.

Apport de 300 € le lendemain (ajouté au cash du fichier de compte) :

```
PEA cœur-satellite — 1 ordre(s) à passer
INFO : Apport détecté : +300,00 €. Plan cœur-satellite : apport — 317,76 € de cash à investir (seuil 200,00 €), vers les poches en retard, par les achats seulement, aucune vente.
INFO : Poids : DCAM 47,9 % → 54,9 % (cible 55,0 %) · PSP5 18,2 % → 18,3 % (cible 20,0 %) · CL2 21,2 % → 21,2 % (cible 25,0 %) · cash 327,76 € → 145,05 €
ORDRE : ACHETER 29 DCAM — ordre au marché (ou à cours limité 6,26 €) · ≈ 180,76 € au cours de 6,23 €, frais ≈ 1,95 €
```

PSP5 et CL2 sont en retard de moins de 150 € : un ordre n'y vaudrait pas ses
frais. Les 145 € restants attendent l'apport suivant (dérive max 3,8 pts,
dans la bande).

**Alertes d'information** (`live.alerts`, jamais d'ordre, une seule fois par
palier, réarmées quand la baisse repasse sous la moitié du palier) :
- portefeuille à -20 / -35 / -50 % de son plus haut (NAV par part, hors
  apports) : « Ne vends pas, c'est prévu dans le plan », avec le rappel du
  plan (poches, seuil de dérive, rééquilibrage de janvier) ;
- CL2 à -30 / -50 % de son plus haut : rappel que le levier x2 amplifie les
  baisses et que le plan le rachète si son poids passe sous sa cible.

**Résultats** — `python scripts/pea_core_satellite_compare.py`, frais
Fortuneo, actions entières, ordre exécuté à l'ouverture du lendemain.
CAGR = rendement par parts (hors effet des apports) ; TRI = rendement
pondéré par l'argent ; Sharpe sans taux sans risque ; « récup. » = durée
entre le plus haut et son retour après le pire creux.

*(a) ETF réels, 20/05/2014 → 24/09/2026, 2 306,59 € sans apport.* La poche
World est CW8.PA remis à l'échelle du cours de DCAM avant le 04/03/2025
(proxy : même indice, Amundi, EUR). 9 cotations aberrantes de CL2.PA (x300
un jour sans volume) ont été retirées. Pire creux : février-mars 2020.

| Variante | CAGR | Max DD | Sharpe | Pire 12 mois | Récup. | Ordres/an | Frais |
|---|---|---|---|---|---|---|---|
| **Plan 55/20/25** | +16,6 % | -40,8 % | 0,88 | -20,9 % | 0,9 an | 1,9 | 46 € |
| 0 % levier (73/27) | +13,2 % | -33,6 % | 0,88 | -17,3 % | 0,9 an | 0,5 | 12 € |
| 50 % levier (37/13/50) | +20,0 % | -47,6 % | 0,87 | -25,3 % | 0,9 an | 2,8 | 69 € |
| 100 % PSP5 | +15,4 % | -33,7 % | 0,93 | -14,8 % | 0,9 an | 0,1 | 5 € |
| 100 % World | +12,4 % | -33,6 % | 0,84 | -18,2 % | 0,9 an | 0,1 | 5 € |

*(a) mêmes ETF, 20 000 € + 100 €/mois (34 800 € versés).*

| Variante | CAGR | TRI | Versé → final | Max DD | Ordres/an | Frais |
|---|---|---|---|---|---|---|
| **Plan 55/20/25** | +16,8 % | +16,8 % | 34 800 → 180 213 € | -40,7 % | 5,2 | 227 € |
| 0 % levier | +13,1 % | +13,1 % | 34 800 → 126 424 € | -33,5 % | 3,2 | 114 € |
| 50 % levier | +20,1 % | +20,0 % | 34 800 → 246 181 € | -47,8 % | 5,1 | 325 € |
| 100 % PSP5 | +15,3 % | +15,2 % | 34 800 → 155 470 € | -33,5 % | 2,4 | 97 € |
| 100 % World | +12,3 % | +12,4 % | 34 800 → 117 644 € | -33,5 % | 2,8 | 106 € |

Avec 2 306,59 € + 100 €/mois : plan 17 107 → 58 593 € (TRI +16,4 %),
100 % PSP5 52 140 €, 100 % World 42 937 €, ~5-7 ordres/an. Sur DCAM réel
seul (03/2025 → 09/2026, 1,5 an, trop court pour conclure) : plan +17,8 %/an,
DD -20,5 % ; PSP5 +15,2 %, DD -17,0 %.

*(b) Stress test synthétique, 2 306,59 € + 100 €/mois, 1990 → 2026
(46 307 € versés).* Pire creux : oct. 2007 → mars 2009 pour toutes les
variantes.

| Variante | CAGR | TRI | Versé → final | Max DD | Sharpe | Pire 12 mois | Récup. | Ordres/an | Frais |
|---|---|---|---|---|---|---|---|---|---|
| **Plan 55/20/25** | +10,9 % | +11,4 % | 46 307 → 689 600 € | -65,1 % | 0,63 | -57,3 % | 5,4 ans | 5,1 | 762 € |
| 0 % levier | +8,9 % | +9,4 % | 46 307 → 409 971 € | -56,9 % | 0,63 | -49,6 % | 5,4 ans | 3,5 | 262 € |
| 50 % levier | +12,6 % | +13,0 % | 46 307 → 1 065 519 € | -72,8 % | 0,60 | -65,1 % | 5,6 ans | 5,6 | 1 450 € |
| 100 % PSP5 | +10,6 % | +10,7 % | 46 307 → 582 903 € | -55,1 % | 0,65 | -47,3 % | 4,9 ans | 3,0 | 224 € |
| 100 % World | +8,2 % | +8,9 % | 46 307 → 360 398 € | -57,7 % | 0,58 | -50,4 % | 5,5 ans | 3,0 | 221 € |

*(b) Sous-périodes, 20 000 € sans apport* (CAGR / max DD / pire 12 mois) :

| Variante | 2000 → 2012 | 2007 → 2009 |
|---|---|---|
| **Plan 55/20/25** | +0,8 % / -65,6 % / -57,9 % — creux non récupéré fin 2012 | -8,9 % / -65,6 % / -57,9 % |
| 0 % levier | +1,6 % / -57,0 % / -49,7 % | -5,8 % / -57,1 % / -49,7 % |
| 50 % levier | -0,5 % / -73,3 % / -65,7 % | -12,3 % / -72,8 % / -65,1 % |
| 100 % PSP5 | +1,6 % / -55,2 % / -47,5 % — récupéré en 4,9 ans | -5,8 % / -55,2 % / -47,5 % |
| 100 % World | +1,6 % / -57,9 % / -50,5 % | -5,9 % / -57,8 % / -50,5 % |

Construction synthétique : S&P 500 = `^SP500TR` - 0,15 %/an de frais ;
levier x2 = 2 × r(`^SP500TR`) - (`^IRX`/100 + 0,6 %)/252 par jour ; World =
`^990100-USD-STRD` (indice MSCI World **prix**, depuis 1985) + 2,3 %/an de
dividendes - 0,20 %/an de frais. Les 2,3 % sont l'écart mesuré entre URTH
(dividendes réinvestis) et cet indice sur 2012-2026. Chaque série démarre à
un cours réaliste (6 / 60 / 30) pour garder la granularité des actions
entières. Calendrier NYSE, **USD, change ignoré**.

**Lecture honnête** :
- Sur 2014-2026, le levier a payé (+1,2 pt/an contre 100 % PSP5), mais pas
  gratuitement : drawdown -41 % contre -34 %, Sharpe plus bas (0,88 contre
  0,93). C'est une seule trajectoire, celle d'un marché US exceptionnel, et
  on le sait après coup : la poche à 25 % est un pari sur sa poursuite.
- Sur 36 ans synthétiques, le plan ne bat 100 % PSP5 que de ~0,3 pt/an. Il
  prend en échange 10 points de drawdown en plus (-65 % contre -55 %) et un
  pire 12 mois à -57 %. Le Sharpe est identique à la variante sans levier :
  CL2 ajoute du risque au moins autant que du rendement. En cause, la
  réinitialisation quotidienne (perte à la volatilité) et le coût d'emprunt
  (taux court + 0,6 %).
- Sur 2000-2012, **le plan est le pire des variantes sans levier** (+0,8 %/an
  contre +1,6 %) et son creux de 2007-2009 n'est toujours pas récupéré fin
  2012. À 50 % de levier, on perd de l'argent sur 13 ans.
- En 2007-2009, attends-toi à voir le portefeuille divisé par ~3 (-65 %) et
  CL2 par ~6 (-84 % en synthétique). Le plan ne vend pas : il rachète CL2 au rééquilibrage.
  C'est ce qui a permis de récupérer en ~5,5 ans sur 1990-2026, mais il faut
  tenir.
- Le World a fait moins bien que le S&P sur toutes les périodes (Japon des
  années 90, domination US depuis). La poche de 55 % sert à diversifier,
  pas à gagner le backtest.
- Si -60 % ne te paraît pas tenable sans vendre, une poche CL2 à 0-15 % a
  historiquement coûté peu en rendement ajusté du risque.

**Limites** : backtests (biais rétrospectif, une seule histoire) ; levier
synthétique (écart de suivi, frais et coût d'emprunt réels de CL2 non
modélisés au-delà de 0,6 %/an) ; change EUR/USD ignoré dans le stress test
(les ETF réels en EUR l'incluent) ; dividendes du World approximés par une
constante ; World réel = proxy CW8 avant 03/2025 ; promo Fortuneo sur les
ETF Amundi non modélisée ; fiscalité ignorée (PEA) ; ordres exécutés à
l'ouverture du lendemain au cours d'ouverture (le vrai cours du matin peut
différer, d'où le coussin de cash).

#### Aperçu du matin (push silencieux avant l'ouverture)

`live.morning_brief` (activé dans `config_pea_fortuneo_80_20.yaml`, désactivé
par défaut ailleurs) : le même process `paper` envoie, en plus du cycle de
18h30, **un push à 08:30** (heure de Paris, jours de bourse XPAR, priorité
Pushover -1 = sans son). Lecture seule : aucun ordre, fichier de compte et
état du plan jamais modifiés, alertes jamais « consommées ».

```yaml
live:
  morning_brief:
    enabled: true
    at: "08:30"
    only_trading_days: true   # false : envoyé aussi week-end/fériés (« Bourse fermée »)
    priority: -1
    catch_up_until: "12:00"   # bot redémarré après 08:30 sans aperçu : envoyé jusqu'à midi
    # title: "…"              # défaut : titre des notifications + « — aperçu du matin »
```

Contenu (< 1 024 caractères) : valeur à la dernière clôture et variation sur
la veille / le mois / l'année (à positions actuelles, cash compris) ; poids
contre cibles et dérive max contre le seuil de 5 pts ; **rappel des ordres
poussés la veille au soir** (mémorisés dans `state.last_cycle_orders`) ou
« Aucun ordre à passer aujourd'hui. » ; cash non investi et seuil d'apport ;
alerte de drawdown active le cas échéant ; une ligne de marché (S&P 500,
MSCI World via URTH, futures S&P, EUR/USD). Une donnée indisponible est omise.
Un aperçu déjà envoyé n'est jamais renvoyé le même jour (`state.last_morning_brief`),
même après un redémarrage. Seul le cycle planifié (`paper` sans `--once`)
mémorise ses ordres pour l'aperçu du lendemain.

```bash
python -m trading_bot morning-brief --config config/config_pea_fortuneo_80_20.yaml --print  # affiche, n'envoie rien
python -m trading_bot morning-brief --config config/config_pea_fortuneo_80_20.yaml          # envoie un aperçu maintenant
```

Exemple réel (`--print` le 25/09/2026 à 7h, 292 DCAM + 7 PSP5 + 56,41 € de
cash, un ordre fictif mémorisé la veille ; yfinance n'avait pas encore
publié la clôture du 24/09, d'où la ligne « pas encore publiée ») :

```
PEA Fortuneo 80/20 — aperçu du matin
Valeur à la clôture du 23/09 : 2 302,56 € (-3,63 €, -0,2 % sur la veille).
Perf. à positions actuelles : sept. +1,9 % · 2026 +15,2 %.
Clôture du 24/09 pas encore publiée par yfinance.
Poids : DCAM 79,4 % (cible 80,0 %) · PSP5 18,1 % (cible 20,0 %). Dérive max 1,9 pts (seuil 5) : dans la bande.
Ordres à passer aujourd'hui (cycle du 24/09) :
- ACHETER 2 PSP5 — ordre au marché (ou à cours limité 59,88 €)
Cash non investi : 56,41 € — sous le seuil d'apport (200,00 €) : il attend le prochain versement.
Marchés : S&P 500 +0,0 % · MSCI World -0,1 % · futures S&P -0,1 % · EUR/USD 1,1374 (-0,1 %).
```

### Reprise après coupe-circuit de drawdown

Si le coupe-circuit de drawdown (`risk.max_drawdown_pct`) se déclenche, le bot
liquide toutes les positions et refuse de retrader — y compris après un
redémarrage, car cet état est persisté dans `live.state_file`
(`state/live_state.json` par défaut). C'est volontaire : un drawdown important
mérite une revue humaine avant de relancer le capital.

Pour reprendre le trading après avoir analysé la situation :

```bash
# Option 1 : repartir d'un état neutre (perd l'historique des stops en cours,
# ce qui est normal puisque tout a été flatten par le coupe-circuit)
rm state/live_state.json

# Option 2 : éditer manuellement le fichier et repasser
# "drawdown_halted" à false dans risk_state, si tu veux conserver le reste de l'état.
```

### Tests

```bash
pytest
```

Les tests unitaires utilisent des données synthétiques (aucun accès réseau
requis) et couvrent les indicateurs, les stratégies, l'allocateur
multi-stratégies, la gestion du risque, le stop suiveur, les coupe-circuits,
le calendrier de marché, la persistance d'état, le moteur de backtest et le
mode PEA cœur-satellite (dérive, apports, rééquilibrage annuel, alertes).

## Architecture

```
src/trading_bot/
  config.py           # chargement de config.yaml + .env
  indicators.py        # SMA, RSI, ATR, rolling max/min
  market_calendar.py    # calendrier de marché (NYSE) : jours fériés, horaires, fermetures anticipées
  state.py               # persistance JSON de l'état live (stops en cours, coupe-circuits, positions connues)
  data/
    historical.py       # données historiques (yfinance) pour le backtest
    market_data.py       # données récentes (Alpaca) pour le live
  strategies/
    base.py              # classe abstraite Strategy
    sma_crossover.py, rsi_mean_reversion.py, momentum_breakout.py
    relative_strength.py  # rotation sectorielle / force relative (cross-sectionnelle)
    buy_and_hold.py        # toujours investi (mode PEA buy & hold)
    dual_momentum.py        # rotation mensuelle momentum relatif + absolu (mode PEA ETF)
    defensive_rotation.py # rotation vers un actif défensif (cross-sectionnelle)
    bollinger_scalping.py  # retour à la moyenne intraday (scalping)
    registry.py           # fabrique de stratégies à partir de la config
  portfolio/
    allocator.py          # combine les signaux de plusieurs stratégies
    risk.py                # dimensionnement des positions + caps de risque
    stops.py                # stop-loss suiveur ATR / en % "Stop Suiveur natif" (logique pure)
    fees.py                  # barème de frais par paliers, actions entières, garde-fous d'ordres
    circuit_breaker.py        # coupe-circuits perte journalière / drawdown
    regime.py                  # filtre de régime de marché (SMA du benchmark)
    universe_rotation.py         # rotation périodique de l'univers tradé, par stratégie
    symbol_track_record.py         # base persistante de performance réelle par symbole
  execution/
    broker_base.py          # interface abstraite de broker
    alpaca_broker.py          # implémentation Alpaca
    manual_broker.py           # implémentation "manuelle" (PEA sans API, ex: Fortuneo)
    rebalancer.py              # calcule et envoie les ordres nécessaires
  backtest/
    engine.py                   # boucle de backtest jour par jour
    trades.py                     # suivi des trades individuels (ouverture/réduction/clôture)
    metrics.py                     # métriques de performance (globales + par trade)
    walk_forward.py                 # validation par fenêtres glissantes entraînement/test
    optimizer.py                     # recherche par grille (grid search), parallélisée
    report.py                         # rapport HTML autonome (matplotlib)
  live/
    engine.py                      # boucle live (paper/réel)
    trade_realization.py             # détection des trades réalisés (comparaison de positions)
  cli.py                             # point d'entrée `python -m trading_bot`
```

## Ajouter une stratégie

1. Crée une classe héritant de `trading_bot.strategies.base.Strategy` et
   implémente `generate_signals(df) -> pd.Series` (valeurs dans `[-1, 1]`,
   vectorisé sur tout l'historique). C'est le cas classique : la stratégie
   juge chaque symbole indépendamment, à partir de son seul historique.
   - Si ta stratégie a besoin de comparer les symboles entre eux (rotation
     sectorielle...) ou de réagir au prix d'un *autre* symbole (rotation
     défensive pilotée par un benchmark...), surcharge plutôt
     `generate_universe_signals(data_by_symbol) -> dict[str, pd.Series]`
     (voir `relative_strength.py` et `defensive_rotation.py` comme exemples) :
     l'allocateur l'appellera une seule fois avec tout l'univers au lieu
     d'appeler `generate_signals` symbole par symbole.
2. Enregistre-la dans `trading_bot/strategies/registry.py` via
   `register_strategy(MaStrategie)`.
3. Ajoute une entrée dans `config/config.yaml` sous `strategies:` avec son
   `name`, un `weight`, et ses `params`.

Aucune autre modification n'est nécessaire : elle sera automatiquement prise
en compte en backtest comme en live.

**Diversifier plutôt qu'empiler** : ajouter une énième stratégie qui réagit au
même type de signal (tendance/momentum sur prix) que les stratégies
existantes n'apporte souvent pas de vraie diversification — elle vote juste
une fois de plus dans le même sens la plupart du temps. Vérifie sa
contribution *isolée* (l'activer seule dans `config.yaml` et comparer) avant
de l'ajouter au mélange à poids plein ; une stratégie d'appoint (couverture,
filtre défensif...) mérite souvent un poids réduit plutôt qu'un poids égal
aux autres (voir le commentaire sur `defensive_rotation` dans
`config/config.yaml`).

## Limites connues / pistes d'amélioration

- Le filtre de régime de marché est un simple seuil (clôture vs SMA) : il peut
  "whipsaw" (bascules répétées) si le prix oscille autour de sa moyenne
  mobile ; pas de zone morte ni de confirmation multi-jours pour l'instant.
- En live, après un ordre de rebalancement, le bot relit immédiatement les
  positions chez le broker pour poser le stop natif sur la quantité réelle ;
  rien ne garantit que l'ordre ait déjà fillé à cet instant précis (pas
  d'attente bloquante). Le cycle suivant corrige la situation si besoin.
- Pas de dimensionnement basé sur la corrélation entre positions (deux
  actions très corrélées peuvent chacune passer les caps de risque
  individuels tout en concentrant le risque réel du portefeuille).
- Pas de ciblage de volatilité au niveau du portefeuille global (le risque
  par trade est individuel, pas agrégé en une cible de volatilité globale).
- Le slippage est approximé par un pourcentage de commission fixe dans le
  backtest ; pas de modélisation de l'impact de marché.
- Pas de crypto (BTC/USD, ETH/USD...) : Alpaca les trade sous un format de
  symbole différent, avec un client de données distinct, et un marché ouvert
  24/7 alors que tout le reste du bot (calendrier NYSE, coupe-circuit
  journalier, buffer de clôture) suppose des séances avec horaires fixes. À
  traiter comme un chantier à part plutôt qu'un simple ajout de symboles.
- Le filtre de sentiment de news ne s'applique qu'au trading live : aucune
  donnée de news historique alignée sur les dates de prix n'est encore
  branchée au backtest (voir `trading_bot.data.news_sentiment`).
- Le walk-forward avec `--optimize` choisit les paramètres sur l'ENTRAÎNEMENT
  et valide sur le TEST correspondant (pas de fuite d'information du futur),
  mais reste soumis à l'overfitting sur le CHOIX de la grille elle-même
  (bornes et valeurs testées) : si les paramètres retenus varient beaucoup
  d'une fenêtre à l'autre (affiché dans le résumé), c'est un signal
  d'alerte à prendre au sérieux plutôt qu'à ignorer.
- `num_trades` (et les stats par trade associées) compte aussi les
  réductions PARTIELLES de position comme des trades réalisés, pas
  seulement les clôtures totales (voir `trading_bot.backtest.trades`) : sur
  un portefeuille rebalancé quotidiennement, le nombre de trades peut donc
  être élevé sans que ce soit une anomalie.
- Les statistiques par trade du résumé **cumulé** d'un walk-forward
  (`combined_out_of_sample_metrics`) valent 0 par construction : l'equity
  combinée entre fenêtres de test est recalée (voir `_chain_equity_curves`),
  ce qui rendrait le P&L en $ des trades individuels incohérent avec cette
  courbe. Les stats par trade restent disponibles fenêtre par fenêtre
  (`fold.test_metrics`).
- Pas de pré-filtrage rapide (vectorisé/approximatif) avant la recherche par
  grille complète : chaque combinaison relance le moteur event-driven en
  entier. Envisageable si la taille des grilles devient un problème en
  pratique, mais volontairement pas implémenté pour l'instant (voir
  `trading_bot.backtest.optimizer`) pour ne jamais risquer un écart entre le
  résultat affiché et ce qui tournerait réellement en live.
- L'intraday (voir [Intraday (ex-scalping)](#intraday-ex-scalping)) n'a
  qu'une seule combinaison validée pour l'instant (`momentum_breakout` en
  bougies 1h) ; les autres stratégies restent calibrées pour du swing
  quotidien, et `bollinger_scalping` (retour à la moyenne) n'a montré aucun
  edge démontrable sur l'univers testé, à aucun grain (5Min/15Min).
  yfinance limite en outre l'historique disponible aux granularités fines
  (~60 jours en dessous de 1Hour, 7 jours pour du 1 min) : `data_source:
  "alpaca"` contourne cette limite pour le backtest (voir
  `trading_bot.data.market_data.fetch_historical_bars`), mais reste soumis
  à la profondeur d'historique réellement disponible côté Alpaca pour
  chaque titre (ex: date d'introduction en bourse).
- Le compte de commission (5 bps) et le modèle de fill (prix d'ouverture de
  la bougie suivante, voir `trading_bot.backtest.engine`) n'ont pas été
  révisés spécifiquement pour l'intraday : un test à commission nulle a
  amélioré le taux de trades gagnants de 64% à 71% sur l'exemple intraday,
  signe qu'ils sous-estiment probablement les coûts réels (spread bid-ask,
  slippage), qui pèsent proportionnellement plus sur des gains visés petits
  et fréquents qu'en swing quotidien.
- La cotation de confiance par track record (voir [Cotation de confiance
  par symbole](#cotation-de-confiance-par-symbole-track-record)) s'appuie,
  côté live, sur `avg_entry_price` tel que renvoyé par le broker plutôt que
  sur un fill exact (le prix de sortie utilisé est le dernier prix connu au
  moment de la détection, pas le prix de fill réel — voir les limites
  précises dans `trading_bot.live.trade_realization`). La même base
  (`live.track_record_file`) est en outre partagée entre toutes les
  stratégies qui l'utilisent : les stats d'un symbole tradé différemment par
  deux stratégies se mélangeraient dans un seul score, pas de séparation par
  stratégie pour l'instant.

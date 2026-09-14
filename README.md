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
    pensée pour de l'**intraday** (bougies 5 min...) plutôt que du swing
    quotidien — voir [Scalping / intraday](#scalping--intraday).
  - Facile d'en ajouter de nouvelles (voir [Ajouter une stratégie](#ajouter-une-stratégie)).
- **Granularité configurable** (`universe.timeframe`) : quotidien par défaut
  (`1Day`), ou intraday (`1Hour`, `30Min`, `15Min`, `5Min`, `1Min`) pour du
  scalping — le même moteur event-driven tourne à l'identique quelle que
  soit la granularité (voir [Scalping / intraday](#scalping--intraday)).
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

### Scalping / intraday

Le bot supporte des bougies intraday (`universe.timeframe: "5Min"`, `"15Min"`,
`"1Hour"`...) en plus du quotidien par défaut — même moteur event-driven,
mêmes commandes (`backtest`, `optimize`, `walk-forward`, `paper`), juste une
granularité différente. Un exemple complet est fourni :
[`config/config_scalping.example.yaml`](config/config_scalping.example.yaml)
(stratégie `bollinger_scalping`, bougies 5 min, stops resserrés, grille
d'optimisation pré-remplie).

```bash
cp config/config_scalping.example.yaml config/config_scalping.yaml
# adapter backtest.start_date : yfinance ne garantit qu'un historique limité
# à cette granularité (~60 jours glissants pour du 5 min, 7 jours pour du 1 min)

python -m trading_bot backtest --config config/config_scalping.yaml
python -m trading_bot optimize --config config/config_scalping.yaml
python -m trading_bot walk-forward --config config/config_scalping.yaml --train-days 15 --test-days 5 --optimize
```

⚠️ À lire avant d'aller plus loin (détaillé en commentaire dans le fichier
d'exemple) :
- `backtest.commission_pct` (5 bps, calibré pour du swing quotidien)
  sous-estime probablement les coûts réels du scalping (spread bid-ask,
  slippage à haute fréquence) — les résultats de backtest sont optimistes
  tant que ce paramètre n'est pas révisé à la hausse.
- Contrainte réglementaire US : un compte sur marge de moins de 25 000 $
  qui fait 4 day trades ou plus en 5 jours ouvrés est classé *Pattern Day
  Trader* et se retrouve bloqué par le broker — le scalping en fait un
  usage intensif par construction.
- Les autres stratégies (`sma_crossover`, `rsi_mean_reversion`,
  `momentum_breakout`) sont désactivées dans l'exemple : leurs fenêtres ont
  été calibrées en JOURS pour du swing, pas en bougies pour de l'intraday —
  les réactiver sans les retuner via `optimize` n'a pas de sens.

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
le calendrier de marché, la persistance d'état et le moteur de backtest.

## Architecture

```
src/trading_bot/
  config.py           # chargement de config.yaml + .env
  indicators.py        # SMA, RSI, ATR, rolling max/min
  market_calendar.py    # calendrier de marché (NYSE) : jours fériés, horaires, fermetures anticipées
  state.py               # persistance JSON de l'état live (stops en cours, coupe-circuits)
  data/
    historical.py       # données historiques (yfinance) pour le backtest
    market_data.py       # données récentes (Alpaca) pour le live
  strategies/
    base.py              # classe abstraite Strategy
    sma_crossover.py, rsi_mean_reversion.py, momentum_breakout.py
    relative_strength.py  # rotation sectorielle / force relative (cross-sectionnelle)
    defensive_rotation.py # rotation vers un actif défensif (cross-sectionnelle)
    bollinger_scalping.py  # retour à la moyenne intraday (scalping)
    registry.py           # fabrique de stratégies à partir de la config
  portfolio/
    allocator.py          # combine les signaux de plusieurs stratégies
    risk.py                # dimensionnement des positions + caps de risque
    stops.py                # stop-loss suiveur ATR (logique pure)
    circuit_breaker.py        # coupe-circuits perte journalière / drawdown
    regime.py                  # filtre de régime de marché (SMA du benchmark)
  execution/
    broker_base.py          # interface abstraite de broker
    alpaca_broker.py          # implémentation Alpaca
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
- Le scalping/intraday (voir [Scalping / intraday](#scalping--intraday))
  n'a qu'une seule stratégie dédiée (`bollinger_scalping`) pour l'instant ;
  les autres restent calibrées pour du swing quotidien. yfinance limite en
  outre l'historique disponible aux granularités fines (~60 jours pour du
  5 min, 7 jours pour du 1 min) : `fetch_historical_data` avertit si
  `backtest.start_date` dépasse cette limite connue, mais ne la fait pas
  respecter automatiquement (yfinance renvoie alors moins de données que
  demandé, silencieusement).
- Le compte de commission (5 bps) et le modèle de fill (prix d'ouverture de
  la bougie suivante, voir `trading_bot.backtest.engine`) n'ont pas été
  révisés spécifiquement pour l'intraday : ils sous-estiment probablement
  les coûts réels du scalping (spread bid-ask, slippage à haute fréquence),
  qui pèsent proportionnellement bien plus sur des gains visés petits et
  fréquents qu'en swing quotidien.

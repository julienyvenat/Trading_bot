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
  - `rsi_mean_reversion` : retour à la moyenne (RSI survente/surachat)
  - `momentum_breakout` : breakout de momentum (canal de Donchian)
  - `relative_strength` : rotation sectorielle / force relative — classe les
    symboles de l'univers *entre eux* (plutôt que dans l'absolu) et ne reste
    investi que sur les plus forts, pour une vraie diversification de
    mécanisme par rapport aux trois stratégies ci-dessus.
  - `defensive_rotation` : rotation défensive — passe sur un actif peu
    corrélé aux actions (or via GLD par défaut) quand le marché large est en
    tendance baissière, pour une diversification de classe d'actif plutôt que
    de signal.
  - Facile d'en ajouter de nouvelles (voir [Ajouter une stratégie](#ajouter-une-stratégie)).
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
  courbe d'equity et métriques (rendement, volatilité, Sharpe, max drawdown) —
  et applique exactement les mêmes stops et coupe-circuits que le live. Le
  signal calculé à la clôture du jour J s'exécute à l'**ouverture du jour
  suivant** (J+1), plus réaliste qu'une exécution immédiate à la clôture.
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
# ou en sauvegardant la courbe d'equity :
python -m trading_bot backtest -o equity_curve.csv
```

Affiche les métriques de performance (rendement total/annualisé, volatilité,
Sharpe, max drawdown, taux de jours positifs), le nombre de sorties
déclenchées par le stop suiveur, et signale si un coupe-circuit s'est
déclenché pendant la période testée.

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
    metrics.py                    # métriques de performance
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

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
  - Facile d'en ajouter de nouvelles (voir [Ajouter une stratégie](#ajouter-une-stratégie)).
- **Gestion du risque** : dimensionnement des positions basé sur l'ATR (risque
  max par trade), poids max par position, exposition brute max du
  portefeuille, nombre max de positions ouvertes.
- **Backtest** event-driven sur données historiques (via `yfinance`), avec
  courbe d'equity et métriques (rendement, volatilité, Sharpe, max drawdown).
- **Paper trading** en continu via l'API Alpaca (compte de simulation gratuit),
  avec bascule facile vers un compte réel (à vos risques).
- Architecture modulaire : le code de stratégie/risque/allocation est
  strictement identique entre backtest et live, pour éviter les écarts de
  comportement entre "ce qui est testé" et "ce qui est tradé".

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
Sharpe, max drawdown, taux de jours positifs).

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

### Tests

```bash
pytest
```

Les tests unitaires utilisent des données synthétiques (aucun accès réseau
requis) et couvrent les indicateurs, les stratégies, l'allocateur
multi-stratégies, la gestion du risque et le moteur de backtest.

## Architecture

```
src/trading_bot/
  config.py           # chargement de config.yaml + .env
  indicators.py        # SMA, RSI, ATR, rolling max/min
  data/
    historical.py       # données historiques (yfinance) pour le backtest
    market_data.py       # données récentes (Alpaca) pour le live
  strategies/
    base.py              # classe abstraite Strategy
    sma_crossover.py, rsi_mean_reversion.py, momentum_breakout.py
    registry.py           # fabrique de stratégies à partir de la config
  portfolio/
    allocator.py          # combine les signaux de plusieurs stratégies
    risk.py                # dimensionnement des positions + caps de risque
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
   vectorisé sur tout l'historique).
2. Enregistre-la dans `trading_bot/strategies/registry.py` via
   `register_strategy(MaStrategie)`.
3. Ajoute une entrée dans `config/config.yaml` sous `strategies:` avec son
   `name`, un `weight`, et ses `params`.

Aucune autre modification n'est nécessaire : elle sera automatiquement prise
en compte en backtest comme en live.

## Limites connues / pistes d'amélioration

- Le backtest exécute au prix de clôture du jour où le signal est calculé
  (approximation optimiste courante pour un prototype) ; envisager une
  exécution à l'ouverture du jour suivant pour plus de réalisme.
- Pas encore de gestion des jours fériés / calendrier de marché fin (repose
  sur les données disponibles).
- Pas de persistance d'état entre redémarrages du moteur live (l'état réel
  du portefeuille est toujours relu depuis Alpaca à chaque cycle, ce qui
  limite l'impact mais ne conserve pas d'historique de décisions).
- Le slippage est approximé par un pourcentage de commission fixe dans le
  backtest ; pas de modélisation de l'impact de marché.

from __future__ import annotations

from pathlib import Path

import yaml

from trading_bot.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

_MINIMAL_YAML = """
universe:
  symbols: [SPY]
  timeframe: "1Day"

strategies:
  - name: sma_crossover
    enabled: true
    weight: 1.0
    params:
      fast_window: 20
      slow_window: 50
  - name: bollinger_scalping
    enabled: true
    weight: 1.0
    params:
      window: 20
      num_std: 2.0
    universe_rotation:
      enabled: true
      candidates: [TSLA, AMD, SMCI]
      metric: volatility
      min_confidence: 0.6
      stability_window: 3
      lookback_window: 20
      rebalance_every: 78

risk:
  allow_short: false
  max_gross_exposure_pct: 0.9
  max_position_weight_pct: 0.25
  risk_per_trade_pct: 0.01
  atr_stop_multiple: 2.5
  atr_window: 14
  max_open_positions: 5
  max_daily_loss_pct: 0.03
  max_drawdown_pct: 0.20

market:
  calendar: "NYSE"
  close_buffer_minutes: 15

backtest:
  start_date: "2020-01-01"
  end_date: null
  initial_cash: 100000
  commission_pct: 0.0005

live:
  loop_interval_seconds: 3600
  trade_only_when_market_open: true
"""


def test_universe_rotation_defaults_to_disabled_when_absent(tmp_path):
    """Une stratégie qui ne déclare pas `universe_rotation` (cas de toutes
    les configs existantes avant cette fonctionnalité) doit rester inchangée :
    rotation désactivée, aucun candidat."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_YAML)
    config = load_config(config_path)

    sma = next(s for s in config.strategies if s.name == "sma_crossover")
    assert sma.universe_rotation.enabled is False
    assert sma.universe_rotation.candidates == []


def test_universe_rotation_parses_declared_block(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_YAML)
    config = load_config(config_path)

    scalping = next(s for s in config.strategies if s.name == "bollinger_scalping")
    rotation = scalping.universe_rotation
    assert rotation.enabled is True
    assert rotation.candidates == ["TSLA", "AMD", "SMCI"]
    assert rotation.metric == "volatility"
    assert rotation.min_confidence == 0.6
    assert rotation.stability_window == 3
    assert rotation.lookback_window == 20
    assert rotation.rebalance_every == 78


def test_shipped_configs_still_load():
    """Garde-fou : les deux configs livrées avec le repo doivent toujours se
    charger sans erreur après l'ajout de `universe_rotation` au schéma."""
    load_config(REPO_ROOT / "config" / "config.yaml")
    load_config(REPO_ROOT / "config" / "config_intraday.example.yaml")
    load_config(REPO_ROOT / "config" / "config_pea_fortuneo.example.yaml")


def test_data_source_defaults_to_yfinance_when_absent(tmp_path):
    """Rétrocompatibilité : une config qui ne déclare pas `data_source`
    (cas de toutes les configs existantes avant cette fonctionnalité) doit
    continuer à utiliser yfinance sans aucun changement de comportement."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_YAML)
    config = load_config(config_path)

    assert config.backtest.data_source == "yfinance"


def test_data_source_parses_alpaca(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_YAML.replace('commission_pct: 0.0005', 'commission_pct: 0.0005\n  data_source: "alpaca"'))
    config = load_config(config_path)

    assert config.backtest.data_source == "alpaca"


def test_live_broker_defaults_to_alpaca_when_absent(tmp_path):
    """Rétrocompatibilité : une config existante sans `live.broker` doit
    continuer à démarrer un `AlpacaBroker`, sans aucun changement de
    comportement."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_MINIMAL_YAML)
    config = load_config(config_path)

    assert config.live.broker == "alpaca"
    assert config.live.manual.account_file == "state/manual_account.json"


def test_live_broker_parses_manual_section(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        _MINIMAL_YAML.replace(
            "trade_only_when_market_open: true",
            'trade_only_when_market_open: true\n  broker: "manual"\n  manual:\n    account_file: "state/pea.json"',
        )
    )
    config = load_config(config_path)

    assert config.live.broker == "manual"
    assert config.live.manual.account_file == "state/pea.json"


def test_shipped_pea_example_config_loads():
    config = load_config(REPO_ROOT / "config" / "config_pea_fortuneo.example.yaml")

    assert config.live.broker == "manual"
    assert config.market.calendar == "XPAR"


def test_pea_example_config_uses_psp5_and_excludes_us_only_symbols():
    """Mode PEA : PSP5 (ETF S&P 500 éligible PEA) remplace SPY, et GLD/VIXY
    (sans équivalent éligible PEA) sont totalement exclus — ni tradés, ni
    utilisés comme référence par un filtre ou une stratégie."""
    config = load_config(REPO_ROOT / "config" / "config_pea_fortuneo.example.yaml")

    assert config.live.broker == "manual"
    assert config.risk.allow_short is False

    assert "PSP5.PA" in config.symbols
    assert config.market.regime_filter.enabled is True
    assert config.market.regime_filter.symbol == "PSP5.PA"
    assert config.market.regime_filter.exempt_symbols == []
    assert config.market.volatility_filter.enabled is False
    assert all(s.name != "defensive_rotation" for s in config.strategies)

    referenced = set(config.symbols) | set(config.market.regime_filter.exempt_symbols)
    referenced.add(config.market.regime_filter.symbol)
    for strategy in config.strategies:
        referenced |= set(strategy.universe_rotation.candidates)
        referenced |= {str(v) for v in strategy.params.values()}
    forbidden = {"SPY", "GLD", "VIXY"}
    assert not referenced & forbidden
    # Filet de sécurité supplémentaire : aucune mention de ces tickers US dans
    # les valeurs du YAML brut (hors commentaires), quel que soit l'endroit.
    raw = yaml.safe_load((REPO_ROOT / "config" / "config_pea_fortuneo.example.yaml").read_text())

    def _values(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from _values(v)
        elif isinstance(node, list):
            for v in node:
                yield from _values(v)
        else:
            yield node

    assert not {str(v) for v in _values(raw)} & forbidden

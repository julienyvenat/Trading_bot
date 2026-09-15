from __future__ import annotations

from pathlib import Path

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
    load_config(REPO_ROOT / "config" / "config_scalping.example.yaml")

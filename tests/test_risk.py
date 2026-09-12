from __future__ import annotations

from trading_bot.config import RiskConfig
from trading_bot.portfolio.risk import RiskManager


def make_risk_manager(**overrides) -> RiskManager:
    defaults = dict(
        allow_short=False,
        max_gross_exposure_pct=0.9,
        max_position_weight_pct=0.25,
        risk_per_trade_pct=0.01,
        atr_stop_multiple=2.5,
        atr_window=14,
        max_open_positions=8,
    )
    defaults.update(overrides)
    return RiskManager(RiskConfig(**defaults))


def test_size_single_respects_max_position_weight():
    rm = make_risk_manager(max_position_weight_pct=0.2)
    sizing = rm.size_single("AAPL", exposure=1.0, last_close=100.0, last_atr=None)
    assert abs(sizing.target_weight) <= 0.2


def test_size_single_risk_based_sizing_shrinks_with_wider_stop():
    rm = make_risk_manager(max_position_weight_pct=1.0, risk_per_trade_pct=0.01, atr_stop_multiple=2.0)
    tight_stop = rm.size_single("AAPL", exposure=1.0, last_close=100.0, last_atr=1.0)
    wide_stop = rm.size_single("AAPL", exposure=1.0, last_close=100.0, last_atr=5.0)
    assert wide_stop.target_weight < tight_stop.target_weight


def test_size_single_negative_exposure_gives_negative_weight():
    rm = make_risk_manager()
    sizing = rm.size_single("AAPL", exposure=-0.5, last_close=100.0, last_atr=None)
    assert sizing.target_weight < 0


def test_apply_portfolio_caps_limits_open_positions():
    rm = make_risk_manager(max_open_positions=1, max_gross_exposure_pct=10.0)
    sizings = {
        "A": rm.size_single("A", 1.0, 100.0, None),
        "B": rm.size_single("B", 0.5, 100.0, None),
    }
    capped = rm.apply_portfolio_caps(sizings)
    assert len(capped) == 1
    assert "A" in capped  # signal le plus fort conservé


def test_apply_portfolio_caps_scales_down_gross_exposure():
    rm = make_risk_manager(max_position_weight_pct=1.0, max_gross_exposure_pct=0.5, max_open_positions=10)
    sizings = {
        "A": rm.size_single("A", 1.0, 100.0, None),
        "B": rm.size_single("B", 1.0, 100.0, None),
    }
    capped = rm.apply_portfolio_caps(sizings)
    gross = sum(abs(s.target_weight) for s in capped.values())
    assert gross <= 0.5 + 1e-9

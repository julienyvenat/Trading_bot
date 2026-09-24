from __future__ import annotations

import pytest

from trading_bot.config import BacktestConfig
from trading_bot.execution.rebalancer import PlannedOrder, apply_execution_rules
from trading_bot.portfolio.fees import CommissionModel, ExecutionRules, decide_order_qty, max_affordable_qty

FORTUNEO = {
    "tiers": [
        {"up_to": 500, "pct": 0.005, "min": 1.95},
        {"up_to": 2000, "fixed": 1.95},
        {"up_to": None, "pct": 0.002},
    ]
}


@pytest.mark.parametrize(
    "notional,expected",
    [(100, 1.95), (390, 1.95), (450, 2.25), (500, 2.50), (500.01, 1.95), (2000, 1.95), (2500, 5.0), (0, 0.0)],
)
def test_fortuneo_schedule(notional, expected):
    assert CommissionModel.from_schedule(FORTUNEO).fee(notional) == pytest.approx(expected)


def test_fee_is_symmetric_on_sells():
    model = CommissionModel.from_schedule(FORTUNEO)
    assert model.fee(-2500) == pytest.approx(5.0)


def test_fallback_to_commission_pct_without_schedule():
    config = BacktestConfig(start_date="2020-01-01", end_date=None, initial_cash=1000, commission_pct=0.001)
    assert CommissionModel.from_backtest_config(config).fee(1000) == pytest.approx(1.0)


def test_global_minimum_applies_to_proportional_model():
    model = CommissionModel.from_schedule({"tiers": [{"up_to": None, "pct": 0.001}], "minimum": 2.0})
    assert model.fee(100) == pytest.approx(2.0)


def test_unsorted_tiers_are_rejected():
    with pytest.raises(ValueError):
        CommissionModel.from_schedule({"tiers": [{"up_to": None, "pct": 0.01}, {"up_to": 500, "pct": 0.02}]})


def test_max_affordable_qty_whole_shares_includes_fees():
    model = CommissionModel.from_schedule(FORTUNEO)
    # 38 x 59.42 = 2257.96 + 4.52 de frais <= 2306.59 ; 39 x 59.42 = 2317.38 > cash
    assert max_affordable_qty(2306.59, 59.42, model, whole_shares=True) == 38
    # Juste assez pour 10 actions sans frais, mais pas avec : 9.
    assert max_affordable_qty(1000.0, 100.0, model, whole_shares=True) == 9


def test_max_affordable_qty_fractional():
    model = CommissionModel(pct=0.01)
    qty = max_affordable_qty(1010.0, 100.0, model, whole_shares=False)
    total = qty * 100 * 1.01
    assert 1009.0 <= total <= 1010.0 + 1e-9


def _rules(**kwargs) -> ExecutionRules:
    defaults = dict(commission=CommissionModel.from_schedule(FORTUNEO), whole_shares=True)
    defaults.update(kwargs)
    return ExecutionRules(**defaults)


def test_decide_rounds_down_and_caps_to_cash():
    qty, reason = decide_order_qty(0, 40.7, 59.42, 2306.59, 2306.59, _rules())
    assert qty == 38 and reason is None


def test_decide_full_exit_always_passes_even_if_tiny():
    qty, reason = decide_order_qty(1, 0, 20.0, 5000, 0, _rules(min_order_value=150, max_fee_pct=0.01))
    assert qty == 0 and reason is None


def test_decide_skips_small_orders():
    qty, reason = decide_order_qty(10, 12, 50.0, 5000, 5000, _rules(min_order_value=150))
    assert qty == 10 and "min_order_value" in reason


def test_decide_skips_orders_with_excessive_fees():
    qty, reason = decide_order_qty(10, 12, 50.0, 5000, 5000, _rules(max_fee_pct=0.01))
    assert qty == 10 and "frais" in reason  # 1,95 € sur 100 € = 1,95 %


def test_decide_tolerance_only_for_adjustments_not_entries():
    rules = _rules(rebalance_tolerance_pct=0.10)
    qty, reason = decide_order_qty(20, 24, 50.0, 2000, 2000, rules)  # 200 € = 10 % -> passe
    assert qty == 24
    qty, reason = decide_order_qty(20, 23, 50.0, 2000, 2000, rules)  # 150 € = 7,5 % -> ignoré
    assert qty == 20 and "tolérance" in reason
    qty, reason = decide_order_qty(0, 3, 50.0, 2000, 2000, rules)  # entrée : pas de tolérance
    assert qty == 3


def test_decide_default_rules_are_a_no_op():
    qty, reason = decide_order_qty(1.5, 3.25, 10.0, 1000, None, ExecutionRules())
    assert qty == 3.25 and reason is None


def test_apply_execution_rules_sells_first_and_funds_buys_with_proceeds():
    orders = [
        PlannedOrder("B", "buy", 20.0, 2000.0),
        PlannedOrder("A", "sell", 20.0, 2000.0),
    ]
    result = apply_execution_rules(
        orders, {"A": 20.0}, equity=2100.0, last_prices={"A": 100.0, "B": 100.0}, cash=100.0, rules=_rules()
    )
    assert [(o.symbol, o.side) for o in result] == [("A", "sell"), ("B", "buy")]
    # 100 + 2000 - 4 (frais de vente 0,20 %) = 2096 € de cash : 20 x 100 + 4 € de frais passent.
    assert result[1].qty == 20


def test_apply_execution_rules_caps_buy_to_cash():
    orders = [PlannedOrder("B", "buy", 40.7, 2418.4)]
    result = apply_execution_rules(orders, {}, equity=2306.59, last_prices={"B": 59.42}, cash=2306.59, rules=_rules())
    assert result[0].qty == 38


def test_apply_execution_rules_drops_skipped_orders():
    orders = [PlannedOrder("B", "buy", 1.0, 50.0)]
    result = apply_execution_rules(
        orders, {"B": 10.0}, equity=5000, last_prices={"B": 50.0}, cash=5000, rules=_rules(min_order_value=150)
    )
    assert result == []

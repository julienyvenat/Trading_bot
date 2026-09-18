from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest.trades import Trade
from trading_bot.portfolio.symbol_track_record import (
    SymbolStats,
    SymbolTrackRecord,
    load_track_record,
    merge_trades,
    save_track_record,
)


def _trade(symbol: str, exit_date: str, pnl: float, entry_value: float = 1000.0) -> Trade:
    return Trade(
        symbol=symbol,
        direction=1,
        entry_date=pd.Timestamp(exit_date) - pd.Timedelta(days=1),
        exit_date=pd.Timestamp(exit_date),
        entry_price=100.0,
        exit_price=100.0 + pnl / 10,
        qty=10.0,
        entry_value=entry_value,
        pnl=pnl,
        exit_reason="rebalance",
    )


def test_merge_trades_accumulates_stats():
    record = SymbolTrackRecord()
    trades = [_trade("TSLA", "2024-01-01", pnl=50.0), _trade("TSLA", "2024-01-02", pnl=-20.0)]

    record = merge_trades(record, trades)

    stats = record.by_symbol["TSLA"]
    assert stats.trade_count == 2
    assert stats.win_count == 1
    assert stats.total_pnl_pct == pytest.approx(0.05 + (-0.02))
    assert stats.last_exit_date == "2024-01-02T00:00:00"


def test_merge_trades_is_idempotent_on_replay():
    """Rejouer le MÊME backtest (mêmes trades) ne doit pas gonfler le
    compteur — la marque d'eau `last_exit_date` doit filtrer les doublons."""
    trades = [_trade("TSLA", "2024-01-01", pnl=50.0), _trade("TSLA", "2024-01-02", pnl=-20.0)]
    record = merge_trades(SymbolTrackRecord(), trades)

    replayed = merge_trades(record, trades)

    assert replayed.by_symbol["TSLA"].trade_count == 2  # inchangé, pas 4


def test_merge_trades_same_exit_date_within_one_call_are_not_mutually_filtered():
    """Régression : deux trades du MÊME symbole partageant la même
    exit_date (ex: deux bougies très rapprochées) au sein d'un SEUL appel
    ne doivent jamais se filtrer l'un l'autre. Le premier traité (ordre de
    tri stable) ne doit pas faire avancer la marque d'eau avant que le
    second, à exit_date identique, ne soit lui aussi comparé."""
    trades = [_trade("TSLA", "2024-01-01T10:00:00", pnl=50.0), _trade("TSLA", "2024-01-01T10:00:00", pnl=-10.0)]

    record = merge_trades(SymbolTrackRecord(), trades)

    assert record.by_symbol["TSLA"].trade_count == 2


def test_merge_trades_appends_genuinely_new_trades_after_watermark():
    """Simule le passage du temps : une deuxième fenêtre, non recouvrante,
    doit bien s'ajouter par-dessus la première."""
    first_window = [_trade("TSLA", "2024-01-01", pnl=50.0)]
    record = merge_trades(SymbolTrackRecord(), first_window)

    second_window = [_trade("TSLA", "2024-02-01", pnl=30.0)]
    record = merge_trades(record, second_window)

    assert record.by_symbol["TSLA"].trade_count == 2
    assert record.by_symbol["TSLA"].last_exit_date == "2024-02-01T00:00:00"


def test_merge_trades_handles_mixed_naive_and_tz_aware_exit_dates():
    """Régression : une marque d'eau posée par un backtest (`exit_date`
    naïve) ne doit pas faire planter la comparaison avec un trade LIVE
    (`exit_date` tz-aware UTC, voir `trading_bot.live.trade_realization`),
    et inversement."""
    naive_first_window = [_trade("TSLA", "2024-01-01", pnl=50.0)]
    record = merge_trades(SymbolTrackRecord(), naive_first_window)

    live_trade = _trade("TSLA", "2024-02-01", pnl=30.0)
    live_trade.exit_date = live_trade.exit_date.tz_localize("UTC")
    record = merge_trades(record, [live_trade])

    assert record.by_symbol["TSLA"].trade_count == 2

    later_naive = [_trade("TSLA", "2024-03-01", pnl=-5.0)]
    record = merge_trades(record, later_naive)

    assert record.by_symbol["TSLA"].trade_count == 3


def test_merge_trades_keeps_symbols_independent():
    trades = [_trade("TSLA", "2024-01-01", pnl=50.0), _trade("AMD", "2024-01-01", pnl=-10.0)]
    record = merge_trades(SymbolTrackRecord(), trades)

    assert record.by_symbol["TSLA"].trade_count == 1
    assert record.by_symbol["AMD"].trade_count == 1
    assert record.by_symbol["AMD"].win_count == 0


def test_expectancy_score_shrinks_small_samples_toward_zero():
    small_sample = SymbolStats(trade_count=2, win_count=2, total_pnl_pct=0.5, total_pnl_pct_sq=0.125)
    large_sample = SymbolStats(trade_count=200, win_count=140, total_pnl_pct=50.0, total_pnl_pct_sq=12.5)

    # Même moyenne brute par trade (0.25) mais échantillon très différent :
    # le score doit rester bien plus proche de 0 pour le petit échantillon.
    assert small_sample.avg_pnl_pct == pytest.approx(large_sample.avg_pnl_pct)
    assert abs(small_sample.expectancy_score()) < abs(large_sample.expectancy_score())


def test_expectancy_score_zero_for_unknown_symbol():
    assert SymbolStats().expectancy_score() == 0.0


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "track_record.json"
    trades = [_trade("TSLA", "2024-01-01", pnl=50.0), _trade("AMD", "2024-01-01", pnl=-10.0)]
    record = merge_trades(SymbolTrackRecord(), trades)

    save_track_record(path, record)
    loaded = load_track_record(path)

    assert loaded.by_symbol["TSLA"].to_dict() == record.by_symbol["TSLA"].to_dict()
    assert loaded.by_symbol["AMD"].to_dict() == record.by_symbol["AMD"].to_dict()


def test_load_missing_file_returns_empty_record(tmp_path):
    record = load_track_record(tmp_path / "does_not_exist.json")
    assert record.by_symbol == {}

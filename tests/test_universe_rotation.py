from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.config import UniverseRotationConfig
from trading_bot.portfolio.universe_rotation import compute_confidence, compute_membership


def _ohlcv(closes: np.ndarray, volume: float = 1_000_000.0, start: str = "2020-01-01") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=len(closes), freq="B")
    close = pd.Series(closes, index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": volume,
        }
    )


def _base_config(**overrides) -> UniverseRotationConfig:
    # Avec 2 candidats, le classement percentile ne peut valoir que 1.0
    # (meilleur) ou 0.5 (pire) : un seuil de 0.75 sélectionne donc
    # exactement le meilleur des deux, jamais l'autre.
    defaults = dict(
        enabled=True,
        candidates=[],
        metric="volatility",
        min_confidence=0.75,
        stability_window=1,
        lookback_window=5,
        rebalance_every=5,
    )
    defaults.update(overrides)
    return UniverseRotationConfig(**defaults)


def test_unreachable_min_confidence_selects_nothing():
    data = {"A": _ohlcv(np.full(20, 100.0)), "B": _ohlcv(np.full(20, 50.0))}
    config = _base_config(min_confidence=1.5)

    membership = compute_membership(data, config)

    assert (membership["A"] == 0.0).all()
    assert (membership["B"] == 0.0).all()


def test_volatility_metric_selects_most_volatile_symbol():
    rng = np.random.default_rng(0)
    calm = 100 + np.cumsum(rng.normal(0, 0.05, size=20))
    choppy = 100 + np.cumsum(rng.normal(0, 5.0, size=20))
    data = {"CALM": _ohlcv(calm), "CHOPPY": _ohlcv(choppy)}
    config = _base_config(metric="volatility", lookback_window=5, rebalance_every=5)

    membership = compute_membership(data, config)

    # Avant la première réévaluation possible (index < lookback_window), rien n'est sélectionné.
    assert (membership["CHOPPY"].iloc[:4] == 0.0).all()
    # Une fois l'historique suffisant, le titre le plus volatil est retenu, jamais le calme.
    assert (membership["CHOPPY"].iloc[5:] == 1.0).all()
    assert (membership["CALM"] == 0.0).all()


def test_momentum_metric_selects_highest_trailing_return():
    up = 100 + np.arange(20) * 2.0
    down = 100 - np.arange(20) * 2.0
    data = {"UP": _ohlcv(up), "DOWN": _ohlcv(down)}
    config = _base_config(metric="momentum", lookback_window=5, rebalance_every=5)

    membership = compute_membership(data, config)

    assert (membership["UP"].iloc[5:] == 1.0).all()
    assert (membership["DOWN"] == 0.0).all()


def test_dollar_volume_metric_selects_most_liquid_symbol():
    flat_price = np.full(20, 100.0)
    data = {
        "LIQUID": _ohlcv(flat_price, volume=10_000_000.0),
        "THIN": _ohlcv(flat_price, volume=1_000.0),
    }
    config = _base_config(metric="dollar_volume", lookback_window=5, rebalance_every=5)

    membership = compute_membership(data, config)

    assert (membership["LIQUID"].iloc[5:] == 1.0).all()
    assert (membership["THIN"] == 0.0).all()


def test_unsupported_metric_raises():
    data = {"A": _ohlcv(np.full(10, 100.0))}
    config = _base_config(metric="not_a_real_metric", lookback_window=3, rebalance_every=3)

    with pytest.raises(ValueError):
        compute_membership(data, config)


def test_stability_window_requires_more_history_before_first_selection():
    """`stability_window` moyenne sur plusieurs réévaluations : il faut donc
    plus d'historique avant la toute première sélection qu'avec
    `stability_window=1` (une seule réévaluation suffit), même si le titre
    le plus intéressant ne change pas entre-temps."""
    rng = np.random.default_rng(0)
    calm = 100 + np.cumsum(rng.normal(0, 0.05, size=30))
    choppy = 100 + np.cumsum(rng.normal(0, 5.0, size=30))
    data = {"CALM": _ohlcv(calm), "CHOPPY": _ohlcv(choppy)}

    quick = _base_config(metric="volatility", stability_window=1, lookback_window=5, rebalance_every=5)
    patient = _base_config(metric="volatility", stability_window=3, lookback_window=5, rebalance_every=5)

    quick_membership = compute_membership(data, quick)["CHOPPY"]
    patient_membership = compute_membership(data, patient)["CHOPPY"]

    first_quick = quick_membership.idxmax()  # premier index où la sélection passe à 1.0
    first_patient = patient_membership.idxmax()
    assert first_patient > first_quick


def test_confidence_is_rolling_average_of_checkpoint_percentiles():
    """Vérifie la mécanique exacte de lissage : la confiance à une date N'EST
    RIEN D'AUTRE que la moyenne du rang percentile (`metric`, parmi les
    candidats) sur les `stability_window` dernières réévaluations."""
    rng = np.random.default_rng(9)
    a = 100 + np.cumsum(rng.normal(0, 2.0, size=30))
    b = 100 + np.cumsum(rng.normal(0, 1.0, size=30))
    data = {"A": _ohlcv(a), "B": _ohlcv(b)}
    config = _base_config(metric="volatility", stability_window=3, lookback_window=5, rebalance_every=5)

    confidence = compute_confidence(data, config)

    close_df = pd.DataFrame({"A": a, "B": b}, index=data["A"].index)
    metric_df = close_df.pct_change().rolling(5, min_periods=5).std(ddof=0)
    percentile_df = metric_df.rank(axis=1, pct=True, ascending=True)
    checkpoints = data["A"].index[::5]  # 0, 5, 10, ...
    expected = percentile_df.loc[checkpoints].iloc[-3:]["A"].mean()

    assert confidence["A"].iloc[-1] == pytest.approx(expected)


def test_selection_holds_between_rebalance_checkpoints():
    """La sélection ne doit changer qu'aux points de rebalancement (tous les
    `rebalance_every` bougies), jamais entre deux."""
    rng = np.random.default_rng(1)
    a = np.concatenate([100 + np.cumsum(rng.normal(0, 5.0, size=10)), np.full(10, 100.0)])
    b = np.concatenate([np.full(10, 50.0), 50 + np.cumsum(rng.normal(0, 5.0, size=10))])
    data = {"A": _ohlcv(a), "B": _ohlcv(b)}
    config = _base_config(metric="volatility", lookback_window=5, rebalance_every=10)

    membership = compute_membership(data, config)

    # Un seul segment [10:20) entre les checkpoints 10 et 20 (exclu, hors
    # de la série de 20 lignes) : la sélection décidée au checkpoint 10 doit
    # être constante sur tout le reste de la série.
    tail_a = membership["A"].iloc[10:]
    tail_b = membership["B"].iloc[10:]
    assert tail_a.nunique() == 1
    assert tail_b.nunique() == 1


def test_no_lookahead_bias():
    """La sélection décidée jusqu'à une date T ne doit pas changer selon que
    des données APRÈS T existent ou non dans l'historique fourni."""
    rng = np.random.default_rng(2)
    a_full = 100 + np.cumsum(rng.normal(0, 1.0, size=30))
    rng = np.random.default_rng(3)
    b_full = 100 + np.cumsum(rng.normal(0, 1.0, size=30))

    data_full = {"A": _ohlcv(a_full), "B": _ohlcv(b_full)}
    data_truncated = {"A": _ohlcv(a_full[:15]), "B": _ohlcv(b_full[:15])}
    config = _base_config(metric="volatility", lookback_window=5, rebalance_every=5)

    membership_full = compute_membership(data_full, config)
    membership_truncated = compute_membership(data_truncated, config)

    for symbol in ("A", "B"):
        pd.testing.assert_series_equal(
            membership_full[symbol].iloc[:15],
            membership_truncated[symbol],
            check_names=False,
        )


def test_membership_reindexed_to_each_symbols_own_index():
    """Un candidat au calendrier différent (ex: coté moins longtemps) reçoit
    une série alignée sur SON PROPRE index, sans planter."""
    data = {
        "LONG": _ohlcv(np.full(20, 100.0)),
        "SHORT": _ohlcv(np.full(10, 100.0), start="2020-01-15"),
    }
    config = _base_config(metric="volatility", lookback_window=5, rebalance_every=5)

    membership = compute_membership(data, config)

    assert list(membership["SHORT"].index) == list(data["SHORT"].index)
    assert list(membership["LONG"].index) == list(data["LONG"].index)

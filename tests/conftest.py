from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.logger import setup_logging


@pytest.fixture(autouse=True)
def _isolate_trading_bot_log(monkeypatch, tmp_path):
    """Empêche tout test qui invoque une commande CLI (`cmd_paper`,
    `cmd_backtest`, ...) — qui appelle `setup_logging()` sans argument — de
    faire écrire le logger partagé `"trading_bot"` dans `logs/trading_bot.log`,
    le fichier de PRODUCTION du bot live surveillé en continu (voir
    `trading_bot.logger.setup_logging`) : ça polluerait ce log de fausses
    lignes qui ressemblent à des coupe-circuits. On redirige le chemin par
    défaut vers un fichier temporaire propre à chaque test plutôt que de
    modifier chaque appelant individuellement."""
    tmp_log = tmp_path / "trading_bot_test.log"
    monkeypatch.setattr(setup_logging, "__defaults__", (logging.INFO, str(tmp_log)))


_LIVE_STATE_DIR = (Path(__file__).resolve().parents[1] / "state").resolve()


@pytest.fixture(autouse=True)
def _forbid_writing_live_track_record(monkeypatch):
    """Fait échouer tout test qui écrirait dans `state/` du dépôt (ex:
    `state/symbol_track_record.json`, la base de suivi par symbole RÉELLE du
    bot live, chemin par défaut de `LiveConfig.track_record_file`) au lieu
    d'un `tmp_path` : ça y injecterait de faux trades (ex: symbole "ORPHAN")."""
    import trading_bot.live.engine as engine_module
    import trading_bot.portfolio.symbol_track_record as track_record_module

    real_save = track_record_module.save_track_record

    def guarded_save(path, record):
        if _LIVE_STATE_DIR in Path(path).resolve().parents:
            raise AssertionError(f"Un test tente d'écrire dans l'état live du bot : {path}")
        return real_save(path, record)

    monkeypatch.setattr(track_record_module, "save_track_record", guarded_save)
    monkeypatch.setattr(engine_module, "save_track_record", guarded_save)


def make_ohlcv(prices: np.ndarray, start: str = "2020-01-01") -> pd.DataFrame:
    """Construit un DataFrame OHLCV simple à partir d'une série de clôtures."""
    index = pd.date_range(start=start, periods=len(prices), freq="B")
    close = pd.Series(prices, index=index)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1_000_000,
        }
    )
    return df


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """Tendance haussière nette : la SMA rapide doit finir au-dessus de la lente."""
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(loc=0.5, scale=0.5, size=250))
    return make_ohlcv(prices)


@pytest.fixture
def trending_down_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    prices = 200 + np.cumsum(rng.normal(loc=-0.5, scale=0.5, size=250))
    prices = np.clip(prices, 1, None)
    return make_ohlcv(prices)


@pytest.fixture
def flat_df() -> pd.DataFrame:
    prices = np.full(120, 50.0)
    return make_ohlcv(prices)


@pytest.fixture(autouse=True)
def _forbid_real_pushover_requests(monkeypatch):
    """Aucun test ne doit envoyer de vraie notification Pushover (réseau) :
    `pytest.fail` lève une `BaseException`, donc n'est PAS avalée par le
    `except Exception` de `PushoverNotifier.send` — un appel réseau
    accidentel fait bien échouer le test. Les tests qui simulent l'API
    remplacent ce garde-fou par leur propre faux `urlopen`. Les identifiants
    éventuellement présents dans l'environnement du développeur sont aussi
    retirés, pour que chaque test parte d'un état connu."""

    def _no_network(*args, **kwargs):
        pytest.fail("Un test tente d'envoyer une vraie notification Pushover (appel réseau).")

    monkeypatch.setattr("trading_bot.notify.pushover.urllib.request.urlopen", _no_network)
    monkeypatch.delenv("PUSHOVER_APP_TOKEN", raising=False)
    monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)

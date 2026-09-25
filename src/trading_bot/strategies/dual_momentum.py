"""Momentum dual (relatif + absolu) à rééquilibrage mensuel — faible rotation."""

from __future__ import annotations

import pandas as pd

from trading_bot.indicators import sma
from trading_bot.strategies.base import Strategy


class DualMomentumStrategy(Strategy):
    """Rotation mensuelle entre quelques ETF : à chaque début de mois, classe
    les symboles par rendement sur `lookback_window` bougies (momentum
    RELATIF) et en retient au plus `top_n` ; chacun n'est gardé que si son
    propre momentum est positif (momentum ABSOLU, `absolute_filter:
    "momentum"`) ou s'il clôture au-dessus de sa SMA `sma_window`
    (`absolute_filter: "sma"`). Chaque "case" non remplie reste en cash, ou
    va sur `risk_off_symbol` s'il est configuré (et présent dans l'univers).

    Différences avec `relative_strength` (non réutilisée telle quelle) :
    celle-ci reclasse CHAQUE jour (rotation quotidienne, beaucoup trop
    d'ordres pour un PEA à passer à la main) et n'a pas de filtre absolu (elle
    reste investie même quand tout baisse).

    Décision prise sur la clôture de la PREMIÈRE séance de chaque mois
    présente dans les données (pas de look-ahead : on n'a pas besoin de
    savoir qu'une séance est la dernière du mois), puis maintenue telle
    quelle jusqu'au mois suivant. Le signal émis est directement le poids
    cible (1/`top_n` par case) : à combiner avec `risk.max_position_weight_pct:
    1.0` et un `risk_per_trade_pct` élevé pour que le dimensionnement ATR ne
    le rogne pas (voir `config/config_pea_etf_momentum.example.yaml`).

    Un symbole sans historique suffisant (ETF plus récent) est simplement
    ignoré du classement tant que son momentum n'est pas calculable.
    """

    name = "dual_momentum"

    def __init__(
        self,
        lookback_window: int = 252,
        top_n: int = 1,
        absolute_filter: str = "momentum",
        sma_window: int = 200,
        risk_off_symbol: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            lookback_window=lookback_window,
            top_n=top_n,
            absolute_filter=absolute_filter,
            sma_window=sma_window,
            risk_off_symbol=risk_off_symbol,
            **kwargs,
        )
        if absolute_filter not in ("momentum", "sma", "none"):
            raise ValueError("`absolute_filter` doit valoir 'momentum', 'sma' ou 'none'.")
        if top_n < 1:
            raise ValueError("`top_n` doit être >= 1.")
        self.lookback_window = lookback_window
        self.top_n = top_n
        self.absolute_filter = absolute_filter
        self.sma_window = sma_window
        self.risk_off_symbol = risk_off_symbol

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Stratégie cross-sectionnelle : voir generate_universe_signals.
        return pd.Series(0.0, index=df.index)

    @staticmethod
    def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Première séance de chaque mois présente dans `index`."""
        if len(index) == 0:
            return index
        periods = index.to_period("M")
        is_first = pd.Series(periods, index=index) != pd.Series(periods, index=index).shift(1)
        return index[is_first.to_numpy()]

    def generate_universe_signals(self, data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        close_df = pd.DataFrame({sym: df["close"] for sym, df in data_by_symbol.items()}).sort_index()
        # Pas de ffill avant le premier cours d'un symbole (il n'existait pas).
        close_df = close_df.ffill()
        momentum = close_df / close_df.shift(self.lookback_window) - 1.0
        above_sma = pd.DataFrame(
            {sym: close_df[sym] > sma(close_df[sym], self.sma_window) for sym in close_df.columns}
        )

        risky = [s for s in close_df.columns if s != self.risk_off_symbol]
        slot_weight = 1.0 / self.top_n
        weights = pd.DataFrame(float("nan"), index=close_df.index, columns=close_df.columns)

        for dt in self.rebalance_dates(close_df.index):
            row = momentum.loc[dt, risky].dropna()
            chosen = list(row.sort_values(ascending=False).index[: self.top_n])
            if self.absolute_filter == "momentum":
                chosen = [s for s in chosen if row[s] > 0]
            elif self.absolute_filter == "sma":
                chosen = [s for s in chosen if bool(above_sma.loc[dt, s])]

            w = pd.Series(0.0, index=close_df.columns)
            for s in chosen:
                w[s] = slot_weight
            empty_slots = self.top_n - len(chosen)
            if (
                empty_slots > 0
                and self.risk_off_symbol in close_df.columns
                and pd.notna(close_df.loc[dt, self.risk_off_symbol])
            ):
                w[self.risk_off_symbol] = empty_slots * slot_weight
            weights.loc[dt] = w

        weights = weights.ffill().fillna(0.0)
        return {symbol: weights[symbol].reindex(df.index).fillna(0.0) for symbol, df in data_by_symbol.items()}

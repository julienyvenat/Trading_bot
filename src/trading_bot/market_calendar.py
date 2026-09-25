"""Calendrier de marché : jours de bourse, horaires d'ouverture, jours fériés
et fermetures anticipées, via `pandas_market_calendars`.

Remplace une logique naïve ("le marché est ouvert du lundi au vendredi de
9h30 à 16h") par le vrai calendrier de la bourse (NYSE par défaut), qui gère
correctement les jours fériés (Thanksgiving, etc.) et les fermetures
anticipées (veille de Noël, lendemain de Thanksgiving...). Fonctionne
entièrement hors-ligne (calendrier basé sur des règles, pas une API).
"""

from __future__ import annotations

import pandas as pd


class MarketCalendar:
    def __init__(self, name: str = "NYSE") -> None:
        import pandas_market_calendars as mcal

        self.name = name
        self._calendar = mcal.get_calendar(name)

    @property
    def timezone(self) -> str:
        """Fuseau horaire local de la place (ex: "Europe/Paris" pour XPAR)."""
        return str(self._calendar.tz)

    def trading_days(self, start, end) -> pd.DatetimeIndex:
        """Jours de bourse (sans les horaires) entre `start` et `end` inclus."""
        schedule = self._calendar.schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(schedule.index)

    def is_trading_day(self, day) -> bool:
        day = pd.Timestamp(day).normalize()
        schedule = self._calendar.schedule(start_date=day, end_date=day)
        return not schedule.empty

    def session_for(self, when: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Renvoie (ouverture, clôture) en UTC pour le jour de `when`, ou None
        si ce jour n'est pas un jour de bourse (week-end, jour férié)."""
        when = _ensure_utc(when)
        day = when.tz_localize(None).normalize()
        schedule = self._calendar.schedule(start_date=day, end_date=day)
        if schedule.empty:
            return None
        row = schedule.iloc[0]
        return row["market_open"], row["market_close"]

    def is_open_now(self, when: pd.Timestamp | None = None) -> bool:
        when = _now_utc() if when is None else _ensure_utc(when)
        session = self.session_for(when)
        if session is None:
            return False
        market_open, market_close = session
        return market_open <= when <= market_close

    def is_actionable(self, when: pd.Timestamp | None = None, close_buffer_minutes: int = 0) -> bool:
        """True si le marché est ouvert ET qu'on est à plus de
        `close_buffer_minutes` de la clôture (évite l'auction de clôture)."""
        when = _now_utc() if when is None else _ensure_utc(when)
        session = self.session_for(when)
        if session is None:
            return False
        market_open, market_close = session
        cutoff = market_close - pd.Timedelta(minutes=close_buffer_minutes)
        return market_open <= when <= cutoff

    def next_session_open(self, when: pd.Timestamp | None = None, search_days: int = 21) -> pd.Timestamp:
        """Prochaine ouverture de séance strictement après `when`."""
        when = _now_utc() if when is None else _ensure_utc(when)
        schedule = self._calendar.schedule(
            start_date=when.tz_localize(None).normalize(),
            end_date=when.tz_localize(None).normalize() + pd.Timedelta(days=search_days),
        )
        future_opens = schedule[schedule["market_open"] > when]
        if future_opens.empty:
            raise RuntimeError(f"Aucune séance de bourse trouvée dans les {search_days} prochains jours.")
        return future_opens.iloc[0]["market_open"]

    def seconds_until_actionable(self, when: pd.Timestamp | None = None, close_buffer_minutes: int = 0) -> float:
        """Secondes à attendre avant le prochain instant où le bot peut agir
        (marché ouvert et hors buffer de clôture). 0 si actionnable maintenant."""
        when = _now_utc() if when is None else _ensure_utc(when)
        if self.is_actionable(when, close_buffer_minutes):
            return 0.0

        session = self.session_for(when)
        if session is not None:
            market_open, market_close = session
            cutoff = market_close - pd.Timedelta(minutes=close_buffer_minutes)
            if when < market_open:
                return (market_open - when).total_seconds()
            # Après le buffer de clôture (ou après la clôture) : direction la prochaine séance.
            next_open = self.next_session_open(when)
            return (next_open - when).total_seconds()

        next_open = self.next_session_open(when)
        return (next_open - when).total_seconds()


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _ensure_utc(when: pd.Timestamp) -> pd.Timestamp:
    when = pd.Timestamp(when)
    return when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")

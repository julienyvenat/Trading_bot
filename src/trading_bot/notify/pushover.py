"""Notifications push via Pushover (https://pushover.net/api).

Pensé d'abord pour le mode semi-automatique `live.broker: "manual"` (voir
`trading_bot.execution.manual_broker`) : au lieu de devoir surveiller les
logs, un seul push par cycle récapitule sur le téléphone les ordres et stops
à passer à la main sur le courtier. Envoie aussi une alerte quand un cycle
crashe ou qu'un coupe-circuit se déclenche (voir `trading_bot.live.engine`).

Identifiants : lus UNIQUEMENT depuis l'environnement (`PUSHOVER_APP_TOKEN`,
`PUSHOVER_USER_KEY`, éventuellement via `.env` chargé par
`trading_bot.config.load_config`), jamais depuis le YAML ni le dépôt.

Règle d'or : une panne de Pushover (réseau, réponse 4xx/5xx, identifiants
absents) ne doit JAMAIS faire échouer le cycle de trading — `send` ne lève
jamais, il journalise un avertissement et renvoie False.

Dépendances : stdlib uniquement (`urllib`), pas de `requests`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from trading_bot.config import PushoverConfig
from trading_bot.logger import get_logger

logger = get_logger()

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"
APP_TOKEN_ENV = "PUSHOVER_APP_TOKEN"
USER_KEY_ENV = "PUSHOVER_USER_KEY"
# Limites documentées par l'API Pushover.
MAX_MESSAGE_CHARS = 1024
MAX_TITLE_CHARS = 250
HTTP_TIMEOUT_SECONDS = 10
# Paramètres obligatoires pour une priorité 2 ("emergency") : répétition
# toutes les `retry` secondes jusqu'à acquittement, au plus `expire` secondes.
EMERGENCY_RETRY_SECONDS = 60
EMERGENCY_EXPIRE_SECONDS = 3600


def truncate_lines(lines: list[str], max_chars: int = MAX_MESSAGE_CHARS) -> str:
    """Assemble `lines` (une par ligne) en restant sous `max_chars`. Coupe
    proprement entre deux lignes plutôt qu'au milieu d'un ordre (un ordre à
    moitié affiché serait trompeur), en signalant combien de lignes ont été
    omises. Une première ligne trop longue à elle seule est tronquée net."""
    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full
    # On garde le plus grand nombre de lignes entières compatible avec la
    # limite, suffixe d'omission compris.
    for kept in range(len(lines) - 1, 0, -1):
        text = "\n".join(lines[:kept] + [f"… (+{len(lines) - kept} ligne(s), voir les logs)"])
        if len(text) <= max_chars:
            return text
    # Première ligne déjà trop longue à elle seule : coupe net avec une ellipse.
    return full[: max_chars - 1] + "…"


class PushoverNotifier:
    """Envoie des notifications Pushover. Si désactivé dans la config, ou si
    les identifiants manquent, `send` ne fait rien (et renvoie False)."""

    def __init__(self, config: PushoverConfig, app_token: str | None = None, user_key: str | None = None) -> None:
        self._config = config
        self._app_token = app_token if app_token is not None else os.getenv(APP_TOKEN_ENV, "").strip()
        self._user_key = user_key if user_key is not None else os.getenv(USER_KEY_ENV, "").strip()

    @property
    def has_credentials(self) -> bool:
        return bool(self._app_token and self._user_key)

    @property
    def alert_priority(self) -> int:
        return self._config.alert_priority

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self.has_credentials

    def warn_if_misconfigured(self) -> None:
        """À appeler au démarrage : prévient clairement si les notifications
        sont activées dans la config mais inutilisables faute d'identifiants."""
        if self._config.enabled and not self.has_credentials:
            logger.warning(
                "Notifications Pushover activées (live.notifications.pushover.enabled) mais %s et/ou %s "
                "absent(s) de l'environnement : AUCUNE notification ne sera envoyée. Renseigne-les dans "
                ".env (voir README, section Pushover).",
                APP_TOKEN_ENV,
                USER_KEY_ENV,
            )

    def send(self, title: str, message: str, priority: int | None = None, force: bool = False) -> bool:
        """Envoie une notification. Ne lève jamais : renvoie True si Pushover
        a accepté le message, False sinon (désactivé, identifiants absents,
        erreur réseau/HTTP — journalisé en avertissement). `force=True`
        ignore `enabled` (utilisé par la commande `notify-test`)."""
        if not (self._config.enabled or force):
            return False
        if not self.has_credentials:
            logger.warning(
                "Notification Pushover non envoyée : %s et/ou %s absent(s) de l'environnement.",
                APP_TOKEN_ENV,
                USER_KEY_ENV,
            )
            return False

        priority = self._config.priority if priority is None else priority
        full_title = f"{self._config.title} — {title}" if self._config.title else title
        payload = {
            "token": self._app_token,
            "user": self._user_key,
            "title": full_title[:MAX_TITLE_CHARS],
            "message": message[:MAX_MESSAGE_CHARS] or "(vide)",
            "priority": str(priority),
        }
        if priority >= 2:
            payload["retry"] = str(EMERGENCY_RETRY_SECONDS)
            payload["expire"] = str(EMERGENCY_EXPIRE_SECONDS)

        data = urllib.parse.urlencode(payload).encode("utf-8")
        request = urllib.request.Request(PUSHOVER_API_URL, data=data, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            # On ne journalise jamais le payload (il contient le token).
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001 - détail purement informatif
                pass
            logger.warning("Notification Pushover refusée (HTTP %s) : %s", exc.code, detail)
            return False
        except Exception as exc:  # noqa: BLE001 - une panne de Pushover ne doit jamais casser le cycle
            logger.warning("Échec d'envoi de la notification Pushover : %s", exc)
            return False

        if body.get("status") != 1:
            logger.warning("Notification Pushover rejetée : %s", body.get("errors") or body)
            return False
        return True

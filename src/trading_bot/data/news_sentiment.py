"""Filtre de sentiment de news via l'API News officielle d'Alpaca.

Volontairement basé sur l'API News d'Alpaca (incluse dans l'abonnement de
données déjà utilisé pour les prix, voir `trading_bot.data.market_data`)
plutôt que sur du scraping de forums ou réseaux sociaux : source unique et
officielle, sans risque de violation des CGU d'un tiers, et sans le bruit
massif des sources non modérées — le "sentiment" extrait par scraping de
Reddit/Twitter est un signal statistiquement très faible en pratique dans
la littérature quantitative, souvent pire que du bruit blanc une fois les
coûts de transaction pris en compte.

Score simple à base de mots-clés (pas de modèle NLP) : volontairement
transparent et auditable plutôt qu'une boîte noire. L'objectif ici est
uniquement de filtrer les cas extrêmes (ex: annonce de fraude, faillite,
enquête réglementaire) avant d'ouvrir une NOUVELLE position, pas de générer
un signal de trading à part entière — voir `NewsSentimentConfig` et son
utilisation dans `trading_bot.live.engine`.

Limite connue : ce filtre ne s'applique qu'au trading live pour l'instant.
Le backtester n'a pas accès à un historique de news aligné sur les mêmes
dates que les prix ; l'ajouter proprement nécessiterait de valider qu'aucun
biais de "look-ahead" ne s'introduit (une news n'est exploitable qu'après sa
publication, pas avant), ce qui reste à faire avant de l'intégrer au
backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

from trading_bot.config import AlpacaCredentials

# Vocabulaire volontairement restreint et sans ambiguïté : on préfère rater
# des nuances (faux négatifs) que déclencher un blocage sur un mot ambigu
# (faux positif) qui empêcherait une entrée par ailleurs légitime.
_NEGATIVE_KEYWORDS = {
    "fraud",
    "bankruptcy",
    "bankrupt",
    "investigation",
    "lawsuit",
    "recall",
    "downgrade",
    "delisting",
    "resigns",
    "resignation",
    "scandal",
    "probe",
    "restatement",
    "default",
    "layoffs",
    "plunge",
    "plunges",
    "subpoena",
}
_POSITIVE_KEYWORDS = {
    "beats",
    "beat estimates",
    "upgrade",
    "record revenue",
    "record profit",
    "surges",
    "surge",
    "raises guidance",
    "buyback",
    "acquisition",
    "partnership",
}


@dataclass
class NewsSentimentResult:
    # Dans [-1, 1] : (articles positifs - négatifs) / articles avec signal.
    # 0.0 si aucun article n'a de mot-clé reconnu (neutre par défaut, jamais
    # bloquant), pas seulement si aucun article n'existe.
    score: float
    num_articles: int


def _score_headline(text: str) -> int:
    lowered = text.lower()
    positive_hits = sum(1 for kw in _POSITIVE_KEYWORDS if kw in lowered)
    negative_hits = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in lowered)
    return positive_hits - negative_hits


def score_headlines(headlines: list[str]) -> NewsSentimentResult:
    """Calcule le score de sentiment à partir de titres/résumés déjà récupérés.

    Séparé de `fetch_recent_sentiment` pour rester testable sans réseau ni
    clés API.
    """
    if not headlines:
        return NewsSentimentResult(score=0.0, num_articles=0)

    per_article_scores = [_score_headline(h) for h in headlines]
    with_signal = [s for s in per_article_scores if s != 0]
    if not with_signal:
        return NewsSentimentResult(score=0.0, num_articles=len(headlines))

    # Normalisé par le nombre d'articles AVEC signal (pas le total), pour ne
    # pas diluer artificiellement le score par des articles neutres.
    score = sum(with_signal) / len(with_signal)
    return NewsSentimentResult(score=max(-1.0, min(1.0, score)), num_articles=len(headlines))


def fetch_recent_sentiment(symbol: str, credentials: AlpacaCredentials, lookback_hours: int) -> NewsSentimentResult:
    """Récupère les news récentes d'un symbole via l'API Alpaca et calcule leur sentiment."""
    from datetime import datetime, timedelta, timezone

    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(credentials.api_key, credentials.secret_key)
    start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    request = NewsRequest(symbols=symbol, start=start, limit=50, exclude_contentless=True)
    news_set = client.get_news(request)

    articles = news_set.data.get("news", [])
    headlines = [f"{article.headline} {article.summary}" for article in articles]
    return score_headlines(headlines)

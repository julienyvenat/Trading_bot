from __future__ import annotations

from trading_bot.data.news_sentiment import score_headlines


def test_score_headlines_no_articles_is_neutral():
    result = score_headlines([])
    assert result.score == 0.0
    assert result.num_articles == 0


def test_score_headlines_all_neutral_is_neutral_but_counts_articles():
    result = score_headlines(["Company announces new office location", "Quarterly newsletter published"])
    assert result.score == 0.0
    assert result.num_articles == 2


def test_score_headlines_negative_keywords_produce_negative_score():
    result = score_headlines(
        [
            "Company under investigation for accounting fraud",
            "Shares plunge after bankruptcy filing",
        ]
    )
    assert result.score < 0
    assert result.num_articles == 2


def test_score_headlines_positive_keywords_produce_positive_score():
    result = score_headlines(
        [
            "Company beats estimates on record revenue",
            "Stock surges after major acquisition announced",
        ]
    )
    assert result.score > 0
    assert result.num_articles == 2


def test_score_headlines_mixed_signals_partially_offset():
    result = score_headlines(
        [
            "Company beats estimates this quarter",  # positif
            "Executive resignation amid scandal",  # négatif
        ]
    )
    assert -1.0 <= result.score <= 1.0
    assert result.num_articles == 2

from datetime import UTC, datetime

from ai_news.models import Article
from ai_news.pipeline import NewsStore, evaluate, verify_and_rank
from ai_news import reporting
from ai_news.reporting import render_report, translate_to_korean, write_web_report


def article(title: str, region: str = "global") -> Article:
    return Article(title, "https://example.com/story", "AI model research update", datetime.now(UTC), "Example", region, 0.9)


def test_verified_article_is_ranked_once(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    first_run = verify_and_rank([article("New AI model")], store)
    second_run = verify_and_rank([article("New AI model")], store)
    assert first_run[0].topic == "model"
    assert first_run[0].score > 0
    assert second_run == []


def test_ranking_reserves_domestic_articles(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    articles = [article(f"Global AI model {index}") for index in range(12)]
    articles.extend([article(f"국내 AI 모델 {index}", "korea") for index in range(4)])
    selected = verify_and_rank(articles, store)
    assert len(selected) == 12
    assert sum(item.region == "korea" for item in selected) == 4


def test_report_and_evaluation_include_quality_signals():
    articles = [article("AI safety update", "korea")]
    text, html = render_report(articles, "공식 출처 우선", "https://example.github.io/ai-news")
    evaluation = evaluate(articles, articles)
    assert "오늘의 핵심 AI 뉴스" in text
    assert "상세 요약" in html and "출처 열기" in html
    assert "https://example.github.io/ai-news/#detail-1" in html
    assert evaluation.valid_url_ratio == 1
    assert evaluation.source_diversity == 1


def test_translation_keeps_articles_when_providers_fail(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(reporting, "_translate_with_google", lambda payload: None)
    original = [article("AI model update")]
    assert translate_to_korean(original) == original


def test_translation_prefers_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-key")
    monkeypatch.setattr(
        reporting,
        "_translate_with_gemini",
        lambda payload, api_key: [{"index": 0, "title": "AI 모델 업데이트", "summary": "AI 모델 연구 업데이트"}],
    )
    monkeypatch.setattr(reporting, "_translate_with_openai", lambda payload, api_key: None)
    translated = translate_to_korean([article("AI model update")])
    assert translated[0].title == "AI 모델 업데이트"


def test_body_summarization_falls_back_to_rss_summary(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    source = article("AI model update")
    enriched = reporting.summarize_article_bodies([source])
    assert enriched[0].detail_summary == source.summary


def test_web_report_has_native_expandable_details(tmp_path):
    destination = tmp_path / "index.html"
    write_web_report([article("AI model update")], "공식 출처 우선", destination)
    content = destination.read_text(encoding="utf-8")
    assert "<details id='detail-1'>" in content
    assert "<summary>1. AI model update</summary>" in content
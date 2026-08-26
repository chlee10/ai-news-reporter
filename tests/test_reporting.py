from datetime import UTC, datetime

import pytest

from ai_news import pipeline, reporting
from ai_news.fetch import FetchError
from ai_news.models import Article, RunMetrics
from ai_news.reporting import render_report, summarize_article_bodies, translate_to_korean, write_web_report
from ai_news.sources import Source


def article(title: str, region: str = "global") -> Article:
    return Article(title, "https://example.com/story", "AI model research update", datetime.now(UTC), "Example", region, 0.9)


# ---------------------------------------------------------------- collection resilience


def test_unreachable_source_is_recorded_instead_of_silently_skipped(monkeypatch):
    def refuse(url, timeout=15.0, attempts=3):
        raise FetchError("connection timed out")

    monkeypatch.setattr(pipeline, "http_get", refuse)
    result = pipeline.collect((Source("Dead Feed", "https://dead.example/rss", "global", 0.5),))
    assert result.articles == []
    assert result.ok_sources == []
    assert "Dead Feed" in result.failures


def test_empty_feed_is_a_recorded_failure_not_a_quiet_zero(monkeypatch):
    class Response:
        content = b"<html>not a feed</html>"

    monkeypatch.setattr(pipeline, "http_get", lambda url, timeout=15.0, attempts=3: Response())
    result = pipeline.collect((Source("Broken", "https://broken.example/rss", "global", 0.5),))
    assert result.failures and not result.ok_sources


def test_healthy_feed_yields_articles_with_dedup_keys(monkeypatch):
    feed = b"""<?xml version="1.0"?><rss version="2.0"><channel>
    <item><title>OpenAI ships a new model</title><link>https://openai.com/a</link>
    <description>Details here</description></item></channel></rss>"""

    class Response:
        content = feed

    monkeypatch.setattr(pipeline, "http_get", lambda url, timeout=15.0, attempts=3: Response())
    result = pipeline.collect((Source("OpenAI", "https://openai.com/rss.xml", "global", 1.0),))
    assert len(result.articles) == 1
    assert result.articles[0].fingerprint and result.articles[0].signature
    assert result.ok_sources == ["OpenAI"]


# ---------------------------------------------------------------- translation


def test_translation_reports_the_engine_it_used(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        reporting,
        "_translate_with_gemini",
        lambda payload, api_key: [{"index": 0, "title": "AI 모델 업데이트", "summary": "AI 모델 연구 업데이트"}],
    )
    translated, engine = translate_to_korean([article("AI model update")])
    assert translated[0].title == "AI 모델 업데이트"
    assert engine == "gemini"


def test_translation_falls_back_to_google_and_says_so(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(reporting, "_translate_with_gemini", lambda payload, api_key: None)
    monkeypatch.setattr(
        reporting, "_translate_with_google", lambda payload: [{"index": 0, "title": "번역", "summary": "요약"}]
    )
    translated, engine = translate_to_korean([article("AI model update")])
    assert engine == "google" and translated[0].title == "번역"


def test_translation_keeps_articles_when_every_engine_fails(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(reporting, "_translate_with_google", lambda payload: None)
    original = [article("AI model update")]
    translated, engine = translate_to_korean(original)
    assert translated == original and engine == "none"


def test_out_of_range_index_from_a_model_does_not_crash_the_run(monkeypatch):
    """A malformed model response must degrade the report, never abort delivery."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        reporting,
        "_translate_with_gemini",
        lambda payload, api_key: [
            {"index": 99, "title": "잘못된 색인", "summary": "무시"},
            {"index": "x", "title": "타입 오류", "summary": "무시"},
            {"index": 0, "title": "정상 번역", "summary": "정상 요약"},
        ],
    )
    translated, engine = translate_to_korean([article("AI model update")])
    assert len(translated) == 1
    assert translated[0].title == "정상 번역"
    assert engine == "gemini"


def test_korean_only_selection_skips_translation(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    translated, engine = translate_to_korean([article("국내 AI 소식", "korea")])
    assert engine == "not-needed" and translated[0].title == "국내 AI 소식"


# ---------------------------------------------------------------- summarization


def test_summary_falls_back_to_rss_excerpt_and_reports_the_downgrade(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    source = article("AI model update")
    summarized, engine = summarize_article_bodies([source])
    assert summarized[0].detail_summary == source.summary
    assert engine == "fallback"


def test_summary_uses_gemini_when_the_response_is_usable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        reporting, "_call_gemini", lambda prompt, api_key, timeout, schema=None: '{"summaries":[{"index":0,"summary":"핵심 요약"}]}'
    )
    summarized, engine = summarize_article_bodies([article("AI model update")])
    assert summarized[0].detail_summary == "핵심 요약" and engine == "gemini"


def test_unusable_gemini_payload_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(reporting, "_call_gemini", lambda prompt, api_key, timeout, schema=None: "not json at all")
    summarized, engine = summarize_article_bodies([article("AI model update")])
    assert engine == "fallback" and summarized[0].detail_summary


# ---------------------------------------------------------------- rendering


def test_email_links_to_the_public_report_when_one_is_configured():
    text, html = render_report([article("AI safety update", "korea")], "공식 출처 우선", "https://example.github.io/ai-news")
    assert "오늘의 핵심 AI 뉴스" in text
    assert "주요 뉴스" in html and "출처 열기" in html
    assert "https://example.github.io/ai-news/#detail-1" in html


def test_email_inlines_details_when_no_public_report_exists():
    _, html = render_report([article("AI safety update")], "공식 출처 우선")
    assert "id='detail-1'" in html and "상세 요약 보기" not in html


def test_web_report_has_native_expandable_details_and_a_quality_panel(tmp_path):
    destination = tmp_path / "index.html"
    metrics = RunMetrics(
        run_at=datetime.now(UTC),
        sources_ok=7,
        sources_failed=1,
        articles_collected=90,
        stories_clustered=60,
        stories_new=14,
        articles_selected=12,
        source_diversity=6,
        body_fetch_ok=9,
        body_fetch_failed=3,
        translation_engine="gemini",
        summary_engine="gemini",
    )
    write_web_report([article("AI model update")], "공식 출처 우선", destination, metrics)
    content = destination.read_text(encoding="utf-8")
    assert "<details id='detail-1' class='item' open>" in content
    assert "AI model update" in content and "주요 뉴스" in content
    assert "모두 펼치기" in content and "모두 접기" in content
    assert "이번 실행 품질 지표" in content
    assert "본문 수집률 75%" in content


def test_empty_report_states_the_gap_plainly(tmp_path):
    text, html = render_report([], "공식 출처 우선")
    assert "검증을 통과한 신규 기사가 없습니다" in text and "검증을 통과한 신규 기사가 없습니다" in html


def test_send_gmail_rejects_an_empty_recipient_list(monkeypatch):
    monkeypatch.setenv("GMAIL_USERNAME", "a@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "xxxx xxxx")
    monkeypatch.setenv("REPORT_RECIPIENTS", " , ")
    with pytest.raises(ValueError):
        reporting.send_gmail("subject", "text", "<p>html</p>")


# ---------------------------------------------------------------- extra feeds


def test_extra_feeds_are_appended_and_deduplicated():
    from ai_news.sources import DEFAULT_SOURCES, load_sources

    sources = load_sources(
        "https://newsroom.example.kr/rss, Custom|https://lab.example.com/feed|global|0.85, "
        "https://openai.com/news/rss.xml"
    )
    added = {item.name: item for item in sources if item not in DEFAULT_SOURCES}
    assert len(sources) == len(DEFAULT_SOURCES) + 2
    assert added["newsroom.example.kr"].region == "korea"
    assert added["Custom"].trust == 0.85


def test_malformed_extra_feeds_are_ignored_not_fatal():
    from ai_news.sources import DEFAULT_SOURCES, load_sources

    assert load_sources("not-a-url, ftp://x/y, ") == DEFAULT_SOURCES
    assert load_sources("") == DEFAULT_SOURCES


def test_extra_feed_with_bad_trust_falls_back_to_a_default():
    from ai_news.sources import load_sources

    source = next(item for item in load_sources("X|https://x.example/rss|global|abc") if item.name == "X")
    assert source.trust == 0.6


# ---------------------------------------------------------------- malformed model output


def test_schema_is_sent_so_gemini_cannot_return_shapeless_json(monkeypatch):
    captured = {}

    def fake_post(url, params=None, json=None, timeout=None):
        captured["config"] = json["generationConfig"]

        class Response:
            status_code = 200

            def raise_for_status(self):
                pass

            @staticmethod
            def json():
                return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"summaries":[]}'}]}}]}

        return Response()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(reporting.requests, "post", fake_post)
    summarize_article_bodies([article("AI model update")])
    assert captured["config"]["responseSchema"] == reporting.SUMMARY_SCHEMA
    assert captured["config"]["responseMimeType"] == "application/json"


def test_json_wrapped_in_a_code_fence_is_recovered():
    recovered = reporting._loads_forgiving('```json\n{"summaries":[{"index":0,"summary":"요약"}]}\n```')
    assert recovered["summaries"][0]["summary"] == "요약"


def test_json_wrapped_in_prose_is_recovered():
    recovered = reporting._loads_forgiving('Here you go:\n{"translations": [{"index": 0}]}\nHope that helps.')
    assert recovered["translations"][0]["index"] == 0


def test_unrecoverable_output_returns_none_rather_than_raising():
    assert reporting._loads_forgiving("no json here at all") is None
    assert reporting._loads_forgiving('{"broken": ') is None


# ---------------------------------------------------------------- toggle behaviour


def test_every_item_is_a_native_details_toggle_with_only_the_first_open(tmp_path):
    """Toggling must survive JavaScript being blocked, so it rests on <details>, not a script."""
    destination = tmp_path / "index.html"
    write_web_report([article(f"AI update {index}") for index in range(3)], "원칙", destination)
    content = destination.read_text(encoding="utf-8")
    assert content.count("<details id='detail-") == 3
    assert content.count(" open>") == 1
    assert "detail-1' class='item' open" in content
    assert "detail-2' class='item'>" in content


def test_expand_all_controls_are_absent_when_there_is_nothing_to_expand(tmp_path):
    destination = tmp_path / "index.html"
    write_web_report([], "원칙", destination)
    content = destination.read_text(encoding="utf-8")
    assert "모두 펼치기" not in content
    assert "검증을 통과한 신규 기사가 없습니다" in content


def test_web_report_headline_and_detail_are_escaped(tmp_path):
    from dataclasses import replace as dc_replace

    destination = tmp_path / "index.html"
    hostile = dc_replace(article("<script>alert(1)</script>"), detail_summary="<img src=x onerror=1>")
    write_web_report([hostile], "원칙", destination)
    content = destination.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content
    assert "<img src=x" not in content

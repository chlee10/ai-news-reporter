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
    assert summarized[0].detail_points == ("AI model research update",)
    assert engine == "fallback"


def test_summary_uses_gemini_when_the_response_is_usable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        reporting,
        "_call_gemini",
        lambda prompt, api_key, timeout, schema=None: (
            '{"summaries":[{"index":0,"points":["매출 30% 증가함","연내 출시 계획","- 규제 심사 진행 중"]}]}'
        ),
    )
    summarized, engine = summarize_article_bodies([article("AI model update")])
    assert engine == "gemini"
    assert summarized[0].detail_points == ("매출 30% 증가함", "연내 출시 계획", "규제 심사 진행 중")


def test_unusable_gemini_payload_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(reporting, "_call_gemini", lambda prompt, api_key, timeout, schema=None: "not json at all")
    summarized, engine = summarize_article_bodies([article("AI model update")])
    assert engine == "fallback" and summarized[0].detail_summary


# ---------------------------------------------------------------- rendering


def test_email_always_inlines_the_outline_even_with_a_public_report():
    """Gmail cannot expand anything, so a reader must never have to click to read the summary."""
    from dataclasses import replace as dc_replace

    item = dc_replace(article("AI safety update", "korea"), detail_points=("첫째 논점임", "둘째 논점임", "셋째 논점임"))
    text, html = render_report([item], "공식 출처 우선", "https://example.github.io/ai-news")
    assert "AI Daily Brief" in text and "총 1건 (국내 1 / 해외 0)" in text
    assert "출처 열기" in html
    assert "상세 요약 보기" not in html
    assert all(point in html for point in ("첫째 논점임", "둘째 논점임", "셋째 논점임"))
    assert "  - 첫째 논점임" in text
    assert "웹에서 보기" in html


def test_email_renders_the_outline_as_a_list_without_a_public_report():
    from dataclasses import replace as dc_replace

    item = dc_replace(article("AI safety update"), detail_points=("논점 하나", "논점 둘"))
    _, html = render_report([item], "공식 출처 우선")
    assert "논점 하나" in html and "논점 둘" in html
    assert "웹에서 보기" not in html
    assert "<style" not in html and "class=" not in html  # inline styles only, for Gmail


def test_outline_falls_back_through_every_available_text():
    from dataclasses import replace as dc_replace
    from ai_news.reporting import outline_points

    assert outline_points(article("t")) == ("AI model research update",)
    empty = dc_replace(article("t"), summary="", detail_summary="")
    assert outline_points(empty) == ("원문 본문을 가져오지 못했습니다.",)
    joined = dc_replace(article("t"), summary="", detail_summary="첫 문장. 둘째 문장. 셋째 문장.")
    assert len(outline_points(joined)) == 3


def test_web_report_has_popup_details_and_a_quality_panel(tmp_path):
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
    assert "id='detail-1'" in content and "class='modal'" in content
    assert "href='#detail-1'" in content
    assert "AI model update" in content and "주요 뉴스" in content
    assert "팝업으로 열립니다" in content
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


def test_each_item_opens_its_own_popup_without_relying_on_javascript(tmp_path):
    """The popup rests on the CSS :target selector, so it works from a downloaded file too."""
    destination = tmp_path / "index.html"
    write_web_report([article(f"AI update {index}") for index in range(3)], "원칙", destination)
    content = destination.read_text(encoding="utf-8")
    assert content.count("class='modal'") == 3
    assert all(f"href='#detail-{n}'" in content for n in (1, 2, 3))
    assert ".modal:target{display:flex}" in content
    assert content.count("href='#close'") == 9  # backdrop, x and 닫기 per item


def test_an_empty_report_has_no_popups(tmp_path):
    destination = tmp_path / "index.html"
    write_web_report([], "원칙", destination)
    content = destination.read_text(encoding="utf-8")
    assert "class='modal'" not in content
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


# ---------------------------------------------------------------- attachment


class _FakeSMTP:
    sent = []

    def __init__(self, host, port):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def login(self, username, password):
        pass

    def send_message(self, message):
        type(self).sent.append(message)


def _prepare_smtp(monkeypatch):
    monkeypatch.setenv("GMAIL_USERNAME", "a@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "xxxx xxxx")
    monkeypatch.setenv("REPORT_RECIPIENTS", "b@example.com")
    _FakeSMTP.sent = []
    monkeypatch.setattr(reporting.smtplib, "SMTP_SSL", _FakeSMTP)


def test_interactive_report_rides_along_as_an_attachment(monkeypatch, tmp_path):
    """Gmail strips the popup markup, so the working copy travels as a file the reader can open."""
    _prepare_smtp(monkeypatch)
    report = tmp_path / "index.html"
    write_web_report([article("AI model update")], "원칙", report)
    reporting.send_gmail("subject", "text", "<p>html</p>", attachment=report)

    message = _FakeSMTP.sent[0]
    attachments = [part for part in message.iter_attachments()]
    assert len(attachments) == 1
    assert attachments[0].get_filename().endswith(".html")
    assert "class='modal'" in attachments[0].get_content()
    assert message.get_body(preferencelist=("html",)).get_content().strip() == "<p>html</p>"


def test_a_missing_attachment_never_blocks_delivery(monkeypatch, tmp_path):
    _prepare_smtp(monkeypatch)
    reporting.send_gmail("subject", "text", "<p>html</p>", attachment=tmp_path / "absent.html")
    assert len(_FakeSMTP.sent) == 1
    assert list(_FakeSMTP.sent[0].iter_attachments()) == []


def test_sending_without_an_attachment_still_works(monkeypatch):
    _prepare_smtp(monkeypatch)
    reporting.send_gmail("subject", "text", "<p>html</p>")
    assert len(_FakeSMTP.sent) == 1


# ---------------------------------------------------------------- email design


def _sample(count: int = 3):
    from dataclasses import replace as dc_replace

    topics = ("model", "policy", "industry")
    return [
        dc_replace(
            article(f"기사 {index}", "korea" if index == 0 else "global"),
            topic=topics[index % len(topics)],
            detail_points=(f"논점 {index}-1", f"논점 {index}-2", f"논점 {index}-3"),
        )
        for index in range(count)
    ]


def test_header_summarises_the_edition_instead_of_concatenating_headlines():
    """The old overview glued three long headlines into one unreadable sentence."""
    text, html = render_report(_sample(3), "원칙")
    assert "총 3건 · 국내 1 · 해외 2" in html
    assert "모델 1" in html and "정책 1" in html
    assert "주요 이슈는" not in html


def test_topics_are_shown_in_korean_not_raw_slugs():
    from ai_news.reporting import topic_label

    _, html = render_report(_sample(1), "원칙")
    assert ">모델<" in html
    assert "model" not in html
    assert topic_label("safety") == "안전" and topic_label("unknown") == "unknown"


def test_operational_note_moves_out_of_the_headline_area_into_the_footer():
    _, html = render_report(_sample(2), "국내 비중 100% → 해외 동향 비중을 회복합니다.")
    header_end = html.index("AI DAILY BRIEF")
    note_at = html.index("국내 비중 100%")
    first_article = html.index("기사 0")
    assert header_end < first_article < note_at


def test_footer_carries_the_run_quality_when_metrics_are_supplied():
    metrics = RunMetrics(
        run_at=datetime.now(UTC),
        articles_collected=187,
        stories_new=81,
        articles_selected=12,
        source_diversity=5,
        body_fetch_ok=10,
        body_fetch_failed=2,
        translation_engine="gemini",
    )
    _, html = render_report(_sample(2), "원칙", "", metrics)
    assert "수집 187건 → 신규 81건 → 선정 12건" in html
    assert "본문 수집률 83%" in html


def test_preheader_is_hidden_but_present():
    _, html = render_report(_sample(1), "원칙")
    assert "display:none;max-height:0" in html
    assert html.index("display:none") < html.index("AI DAILY BRIEF")


def test_empty_edition_still_renders_a_valid_shell():
    text, html = render_report([], "원칙")
    assert "검증을 통과한 신규 기사가 없습니다" in text
    assert "검증을 통과한 신규 기사가 없습니다" in html
    assert html.count("<table") == html.count("</table>")


def test_email_column_is_the_widened_size_and_still_fluid_on_mobile():
    _, html = render_report(_sample(1), "원칙")
    assert "max-width:960px" in html
    assert "width:100%;max-width:960px" in html  # shrinks on narrow screens
    assert "max-width:640px" not in html


def test_a_short_edition_tells_the_reader_why():
    metrics = RunMetrics(run_at=datetime.now(UTC), articles_selected=3, target_size=12, stale_dropped=40)
    text, html = render_report(_sample(3), "원칙", "", metrics)
    assert "신선한 신규 기사 기준 3/12건" in html
    assert "신선한 신규 기사 기준 3/12건" in text


def test_a_full_edition_shows_no_shortfall_line():
    metrics = RunMetrics(run_at=datetime.now(UTC), articles_selected=3, target_size=3)
    _, html = render_report(_sample(3), "원칙", "", metrics)
    assert "신선한 신규 기사 기준" not in html


def test_a_repeated_story_is_labelled_so_the_reader_is_never_misled():
    from dataclasses import replace as dc_replace

    items = _sample(2)
    items[1] = dc_replace(items[1], times_sent=2)
    text, html = render_report(items, "원칙")
    assert html.count("재등장") == 1
    assert "(재등장) 기사 1" in text

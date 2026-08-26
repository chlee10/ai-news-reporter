import json
import os
import smtplib
from dataclasses import replace
from email.message import EmailMessage
from html import escape
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .fetch import FetchError
from .fetch import get as http_get
from .models import Article, RunMetrics
from .observability import get_logger

LOGGER = get_logger("reporting")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# responseMimeType alone does not guarantee parseable JSON; a schema makes Gemini's
# decoding enforce the shape. Observed in production: a translation batch came back
# with a malformed object and the whole run fell back to Google translation.
TRANSLATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "translations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "title": {"type": "STRING"},
                    "summary": {"type": "STRING"},
                },
                "required": ["index", "title", "summary"],
            },
        }
    },
    "required": ["translations"],
}
SUMMARY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summaries": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"index": {"type": "INTEGER"}, "summary": {"type": "STRING"}},
                "required": ["index", "summary"],
            },
        }
    },
    "required": ["summaries"],
}


# ---------------------------------------------------------------- source text


def enrich_with_article_bodies(articles: list[Article]) -> tuple[list[Article], dict[str, bool]]:
    """Retrieve readable source text. Returns per-source outcomes so unreadable outlets lose weight."""
    enriched: list[Article] = []
    outcomes: dict[str, bool] = {}
    for article in articles:
        body = _fetch_article_body(article.url)
        outcomes[article.source] = outcomes.get(article.source, False) or bool(body)
        enriched.append(replace(article, body=body))
    succeeded = sum(1 for article in enriched if article.body)
    LOGGER.info("fetched full text for %s/%s articles", succeeded, len(enriched))
    return enriched, outcomes


def _fetch_article_body(url: str) -> str:
    try:
        response = http_get(url, timeout=15.0, attempts=2)
    except FetchError as error:
        LOGGER.warning("body unavailable for %s: %s", url, error)
        return ""
    try:
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in soup.select("article p, main p")]
        if not paragraphs:
            paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in soup.find_all("p")]
        return " ".join(paragraphs)[:6000]
    except (ValueError, AttributeError) as error:
        LOGGER.warning("body unparseable for %s: %s", url, error)
        return ""


# ---------------------------------------------------------------- Korean synthesis


def translate_to_korean(articles: list[Article]) -> tuple[list[Article], str]:
    """Translate non-Korean text via Gemini, falling back to Google. Returns the engine that was used."""
    if not articles:
        return articles, "none"
    payload = [
        {"index": index, "title": article.title, "summary": article.summary[:500]}
        for index, article in enumerate(articles)
        if article.region != "korea"
    ]
    if not payload:
        return articles, "not-needed"

    engine = "none"
    translations = None
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        translations = _translate_with_gemini(payload, api_key)
        engine = "gemini" if translations else engine
    else:
        LOGGER.warning("GEMINI_API_KEY is unset; going straight to the Google translation fallback")
    if translations is None:
        translations = _translate_with_google(payload)
        engine = "google" if translations else "none"
    if translations is None:
        LOGGER.error("every translation engine failed; foreign articles ship untranslated")
        return articles, "none"

    translated = list(articles)
    applied = 0
    for item in translations:
        index = item.get("index") if isinstance(item, dict) else None
        if not isinstance(index, int) or not 0 <= index < len(translated):
            LOGGER.warning("translation engine returned an out-of-range index: %r", index)
            continue
        title = str(item.get("title") or translated[index].title)
        summary = str(item.get("summary") or translated[index].summary)
        translated[index] = replace(translated[index], title=title, summary=summary)
        applied += 1
    LOGGER.info("translated %s/%s foreign articles via %s", applied, len(payload), engine)
    return translated, engine


def summarize_article_bodies(articles: list[Article]) -> tuple[list[Article], str]:
    """Produce a 2-3 sentence Korean brief per article. Returns the engine so the run can grade itself."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not articles:
        if articles:
            LOGGER.warning("GEMINI_API_KEY is unset; detail summaries fall back to RSS excerpts")
        return _fallback_summaries(articles), "fallback" if articles else "none"
    payload = [
        {"index": index, "title": article.title, "body": (article.body or article.summary)[:3500]}
        for index, article in enumerate(articles)
    ]
    prompt = (
        "Summarize each news article in Korean in 2 or 3 concise sentences. "
        "Use only supplied text, preserve key entities and numbers, and return only JSON: "
        '{"summaries":[{"index":0,"summary":"..."}]}.\n' + json.dumps(payload, ensure_ascii=False)
    )
    content = _call_gemini(prompt, api_key, timeout=45.0, schema=SUMMARY_SCHEMA)
    parsed = _loads_forgiving(content) if content is not None else None
    if parsed is None:
        return _fallback_summaries(articles), "fallback"
    try:
        summaries = parsed["summaries"]
        by_index = {item["index"]: item["summary"] for item in summaries if isinstance(item, dict)}
    except (KeyError, TypeError, ValueError) as error:
        LOGGER.error("Gemini summary payload was unusable: %s", error)
        return _fallback_summaries(articles), "fallback"
    resolved = [
        replace(article, detail_summary=str(by_index.get(index, article.summary))[:900])
        for index, article in enumerate(articles)
    ]
    covered = sum(1 for index in range(len(articles)) if index in by_index)
    LOGGER.info("Gemini summarized %s/%s articles", covered, len(articles))
    return resolved, "gemini" if covered else "fallback"


def _fallback_summaries(articles: list[Article]) -> list[Article]:
    return [replace(article, detail_summary=article.summary[:550]) for article in articles]


def _call_gemini(prompt: str, api_key: str, timeout: float, schema: dict | None = None) -> str | None:
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    config: dict[str, object] = {"responseMimeType": "application/json"}
    if schema is not None:
        config["responseSchema"] = schema
    try:
        response = requests.post(
            GEMINI_ENDPOINT.format(model=model),
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config},
            timeout=timeout,
        )
        response.raise_for_status()
        candidate = response.json()["candidates"][0]
        reason = candidate.get("finishReason")
        if reason not in (None, "STOP"):
            LOGGER.warning("Gemini stopped early (finishReason=%s); output may be truncated", reason)
        return candidate["content"]["parts"][0]["text"]
    except requests.RequestException as error:
        LOGGER.error("Gemini request failed (model=%s): %s", model, error)
        return None
    except (KeyError, IndexError, TypeError, ValueError) as error:
        LOGGER.error("Gemini response had an unexpected shape (model=%s): %s", model, error)
        return None


def _loads_forgiving(content: str) -> dict | None:
    """Parse Gemini output, salvaging a JSON object wrapped in prose or a code fence."""
    try:
        return json.loads(content)
    except ValueError:
        pass
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1] if "```" in stripped[3:] else stripped[3:]
        stripped = stripped.removeprefix("json").strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        recovered = json.loads(stripped[start : end + 1])
    except ValueError:
        return None
    LOGGER.warning("recovered a usable JSON object from a malformed Gemini response")
    return recovered


def _translate_with_gemini(payload: list[dict[str, object]], api_key: str) -> list[dict[str, object]] | None:
    prompt = (
        "Translate each English AI-news title and summary into natural, concise Korean. "
        "Return only a JSON object with a translations array; each item must contain index, title, and summary. "
        "Do not add facts.\n" + json.dumps(payload, ensure_ascii=False)
    )
    content = _call_gemini(prompt, api_key, timeout=30.0, schema=TRANSLATION_SCHEMA)
    parsed = _loads_forgiving(content) if content is not None else None
    if parsed is None:
        return None
    try:
        translations = parsed["translations"]
    except (KeyError, TypeError) as error:
        LOGGER.error("Gemini translation payload was unusable: %s", error)
        return None
    if not isinstance(translations, list):
        LOGGER.error("Gemini returned a non-list translations field")
        return None
    return translations


def _translate_with_google(payload: list[dict[str, object]]) -> list[dict[str, object]] | None:
    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="auto", target="ko")
        titles = translator.translate_batch([str(item["title"]) for item in payload])
        summaries = translator.translate_batch([str(item["summary"]) for item in payload])
        if any(str(value).lstrip().lower().startswith("error ") for value in titles + summaries):
            LOGGER.error("Google translation returned an error marker")
            return None
        return [
            {"index": item["index"], "title": title, "summary": summary}
            for item, title, summary in zip(payload, titles, summaries, strict=True)
        ]
    except Exception as error:  # deep-translator raises a wide range of provider errors
        LOGGER.error("Google translation failed: %s", error)
        return None


# ---------------------------------------------------------------- rendering


def render_report(articles: list[Article], guidance: str, report_url: str = "") -> tuple[str, str]:
    headlines = ", ".join(article.title for article in articles[:3]) or "검증을 통과한 신규 기사가 없습니다"
    overview = f"오늘의 핵심 AI 뉴스 {len(articles)}건입니다. 주요 이슈는 {headlines}입니다. 국내외 정책, 모델, 산업 동향을 함께 선별했습니다."
    text = ["AI Daily Brief", overview, f"편집 원칙: {guidance}", ""]
    html = [
        "<div style='max-width:720px;margin:auto;font-family:Arial,sans-serif;color:#1f2937;line-height:1.65'>",
        "<h1 style='color:#0f766e'>AI Daily Brief</h1>",
        f"<p style='font-size:16px'><strong>오늘의 요약</strong><br>{escape(overview)}</p>",
        f"<p style='color:#4b5563'><strong>편집 원칙:</strong> {escape(guidance)}</p>",
        "<h2 style='border-bottom:2px solid #99f6e4'>주요 뉴스</h2>",
    ]
    for detail_number, article in enumerate(articles, start=1):
        detail = article.detail_summary or article.summary[:550].strip() or "원문 본문을 가져오지 못했습니다."
        corroboration = f" | 교차 보도 {article.related_domains}곳" if article.related_domains > 1 else ""
        text.extend([f"## {detail_number}. {article.title}", detail, f"{article.source}: {article.url}"])
        if report_url:
            detail_link = f"{report_url.rstrip('/')}/#detail-{detail_number}"
            html.append(
                f"<section style='padding:14px 0;border-bottom:1px solid #d1d5db'>"
                f"<h3>{detail_number}. {escape(article.title)}</h3>"
                f"<p><small>{escape(article.source)}{escape(corroboration)} | "
                f"<a href='{escape(detail_link, quote=True)}'>상세 요약 보기</a> | "
                f"원문: <a href='{escape(article.url, quote=True)}'>출처 열기</a></small></p></section>"
            )
            continue
        html.append(
            f"<section id='detail-{detail_number}' style='padding:14px 0;border-bottom:1px solid #d1d5db'>"
            f"<h3>{detail_number}. {escape(article.title)}</h3><p>{escape(detail)}</p>"
            f"<p><small>{escape(article.source)} | 원문: <a href='{escape(article.url, quote=True)}'>출처 열기</a> | "
            f"{escape(article.topic)}{escape(corroboration)}</small></p></section>"
        )
    if not articles:
        text.append("검증을 통과한 신규 기사가 없습니다.")
        html.append("<p>검증을 통과한 신규 기사가 없습니다.</p>")
    html.append("</div>")
    return "\n".join(text), "".join(html)


def write_web_report(
    articles: list[Article],
    guidance: str,
    path: Path = Path("reports/index.html"),
    metrics: RunMetrics | None = None,
) -> None:
    """Write a browser report with native expandable details and this run's quality panel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headlines = ", ".join(article.title for article in articles[:3]) or "검증을 통과한 신규 기사가 없습니다"
    overview = f"오늘의 핵심 AI 뉴스 {len(articles)}건입니다. 주요 이슈는 {headlines}입니다."
    sections = []
    for number, article in enumerate(articles, start=1):
        detail = article.detail_summary or article.summary[:550].strip() or "원문 본문을 가져오지 못했습니다."
        corroboration = f" · 교차 보도 {article.related_domains}곳" if article.related_domains > 1 else ""
        # The whole header row is the toggle; <details> keeps it working with JavaScript disabled.
        sections.append(
            f"<details id='detail-{number}' class='item'{' open' if number == 1 else ''}>"
            f"<summary><span class='num'>{number}</span>"
            f"<span class='headline'>{escape(article.title)}</span>"
            f"<span class='chev' aria-hidden='true'></span></summary>"
            f"<div class='body'><p class='detail'>{escape(detail)}</p>"
            f"<p class='meta'><span class='tag'>{escape(article.topic)}</span>{escape(article.source)}"
            f"{escape(corroboration)} · "
            f"<a href='{escape(article.url, quote=True)}' target='_blank' rel='noopener'>출처 열기 ↗</a></p></div>"
            "</details>"
        )
    body = "".join(sections) or "<p>검증을 통과한 신규 기사가 없습니다.</p>"
    controls = (
        "<div class='controls'><button type='button' data-open='1'>모두 펼치기</button>"
        "<button type='button' data-open='0'>모두 접기</button></div>"
        if articles
        else ""
    )
    page = f"""<!doctype html>
<html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>AI Daily Brief</title><style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f0fdfa;color:#172033;font-family:-apple-system,'Segoe UI','Malgun Gothic',Arial,sans-serif;line-height:1.7}}
main{{max-width:760px;margin:36px auto;padding:0 20px}}
h1{{color:#0f766e;margin:0 0 18px}}h2{{color:#134e4a;margin:32px 0 8px;padding-bottom:8px;border-bottom:2px solid #99f6e4}}
.intro{{background:#fff;padding:20px;border-left:4px solid #14b8a6;border-radius:0 6px 6px 0}}
.controls{{display:flex;gap:8px;margin:12px 0 4px}}
.controls button{{font:inherit;font-size:13px;padding:6px 12px;border:1px solid #99f6e4;background:#fff;color:#0f766e;border-radius:999px;cursor:pointer}}
.controls button:hover{{background:#ccfbf1}}
.item{{background:#fff;margin:10px 0;border:1px solid #ccfbf1;border-radius:8px;overflow:hidden}}
.item[open]{{border-color:#5eead4;box-shadow:0 1px 3px rgba(15,118,110,.12)}}
summary{{display:flex;align-items:center;gap:12px;cursor:pointer;padding:14px 18px;font-weight:700;color:#0f766e;list-style:none}}
summary::-webkit-details-marker{{display:none}}
summary:hover{{background:#f0fdfa}}
summary:focus-visible{{outline:2px solid #14b8a6;outline-offset:-2px}}
.num{{flex:none;width:26px;height:26px;border-radius:50%;background:#ccfbf1;color:#0f766e;font-size:13px;display:flex;align-items:center;justify-content:center}}
.headline{{flex:1}}
.chev{{flex:none;width:9px;height:9px;border-right:2px solid #14b8a6;border-bottom:2px solid #14b8a6;transform:rotate(45deg);transition:transform .18s;margin-right:4px}}
.item[open] .chev{{transform:rotate(-135deg)}}
.body{{padding:0 18px 16px 56px}}
.detail{{margin:0 0 10px}}
.meta{{color:#4b5563;font-size:13px;margin:0}}
.tag{{display:inline-block;background:#f0fdfa;border:1px solid #ccfbf1;border-radius:4px;padding:1px 7px;margin-right:8px;font-size:12px;color:#0f766e}}
a{{color:#0f766e}}
.quality{{margin-top:28px;padding:16px;background:#fff;border:1px dashed #99f6e4;color:#4b5563;font-size:13px;border-radius:8px}}
.quality ul{{margin:8px 0 0;padding-left:18px}}
@media (max-width:520px){{.body{{padding-left:18px}}}}
</style></head><body><main>
<h1>AI Daily Brief</h1>
<div class='intro'><strong>오늘의 요약</strong><br>{escape(overview)}<br><small>편집 원칙: {escape(guidance)}</small></div>
<h2>주요 뉴스</h2>{controls}{body}{_quality_panel(metrics)}
</main><script>
document.querySelectorAll('.controls button').forEach(function (button) {{
  button.addEventListener('click', function () {{
    var open = button.dataset.open === '1';
    document.querySelectorAll('details.item').forEach(function (item) {{ item.open = open; }});
  }});
}});
</script></body></html>"""
    path.write_text(page, encoding="utf-8")
    LOGGER.info("web report written to %s", path)


def _quality_panel(metrics: RunMetrics | None) -> str:
    if metrics is None:
        return ""
    rows = [
        f"수집 {metrics.articles_collected}건 → 사건 {metrics.stories_clustered}건 → 신규 {metrics.stories_new}건 → 선정 {metrics.articles_selected}건",
        f"피드 정상 {metrics.sources_ok}곳 / 실패 {metrics.sources_failed}곳 · 출처 다양성 {metrics.source_diversity}곳",
        f"본문 수집률 {metrics.body_fetch_ratio:.0%} · 기사 중위 연령 {metrics.median_age_hours:.0f}시간",
        f"번역 엔진 {metrics.translation_engine} · 요약 엔진 {metrics.summary_engine}",
    ]
    items = "".join(f"<li>{escape(row)}</li>" for row in rows)
    return f"<div class='quality'><strong>이번 실행 품질 지표</strong><ul>{items}</ul></div>"


# ---------------------------------------------------------------- delivery


def send_gmail(subject: str, text: str, html: str) -> None:
    username = os.environ["GMAIL_USERNAME"]
    password = os.environ["GMAIL_APP_PASSWORD"].replace(" ", "")
    recipients = [address.strip() for address in os.environ["REPORT_RECIPIENTS"].split(",") if address.strip()]
    if not recipients:
        raise ValueError("REPORT_RECIPIENTS must contain at least one email address")
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = username
    message["To"] = ", ".join(recipients)
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(username, password)
        smtp.send_message(message)
    LOGGER.info("report delivered to %s recipient(s)", len(recipients))

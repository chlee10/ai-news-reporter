import json
import os
import re
import smtplib
from dataclasses import replace
from datetime import datetime
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
                "properties": {
                    "index": {"type": "INTEGER"},
                    "points": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["index", "points"],
            },
        }
    },
    "required": ["summaries"],
}
MIN_POINTS, MAX_POINTS = 3, 4


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
    """Produce a 3-4 point Korean outline per article. Returns the engine so the run can grade itself."""
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
        f"각 기사를 한국어 개조식으로 {MIN_POINTS}~{MAX_POINTS}개 항목으로 요약하라. "
        "각 항목은 한 문장으로 쓰고 '~함', '~임', '~할 계획' 같은 명사형·축약형으로 끝낼 것. "
        "항목마다 다른 논점을 다루고, 핵심 수치·기관명·제품명은 그대로 보존할 것. "
        "제공된 본문에 없는 사실은 절대 추가하지 말 것. 불릿 기호는 붙이지 말 것.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    content = _call_gemini(prompt, api_key, timeout=45.0, schema=SUMMARY_SCHEMA)
    parsed = _loads_forgiving(content) if content is not None else None
    if parsed is None:
        return _fallback_summaries(articles), "fallback"
    try:
        summaries = parsed["summaries"]
        by_index = {item["index"]: item["points"] for item in summaries if isinstance(item, dict)}
    except (KeyError, TypeError) as error:
        LOGGER.error("Gemini summary payload was unusable: %s", error)
        return _fallback_summaries(articles), "fallback"

    resolved = []
    covered = 0
    for index, article in enumerate(articles):
        points = _clean_points(by_index.get(index))
        if points:
            covered += 1
        else:
            points = _points_from_text(article.summary)
        resolved.append(replace(article, detail_points=points, detail_summary=" ".join(points)[:900]))
    LOGGER.info("Gemini summarized %s/%s articles into outline points", covered, len(articles))
    return resolved, "gemini" if covered else "fallback"


def _clean_points(raw: object) -> tuple[str, ...]:
    """Keep only usable outline points and strip any bullet glyph the model added anyway."""
    if not isinstance(raw, list):
        return ()
    points = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = item.strip().lstrip("-•*·–—").strip()
        if text:
            points.append(text[:300])
    return tuple(points[:MAX_POINTS])


def _points_from_text(text: str) -> tuple[str, ...]:
    """Split an RSS excerpt into outline points so the fallback still reads as an outline."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。])\s+|\n+", text or "") if part.strip()]
    if not sentences:
        return ()
    return tuple(sentence[:300] for sentence in sentences[:MAX_POINTS])


def _fallback_summaries(articles: list[Article]) -> list[Article]:
    return [
        replace(
            article,
            detail_points=_points_from_text(article.summary[:550]),
            detail_summary=article.summary[:550],
        )
        for article in articles
    ]


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


def outline_points(article: Article) -> tuple[str, ...]:
    """The outline shown to the reader, falling back through every source of text we have."""
    if article.detail_points:
        return article.detail_points
    fallback = article.detail_summary or article.summary[:550].strip()
    return _points_from_text(fallback) or ("원문 본문을 가져오지 못했습니다.",)


TOPIC_LABELS = {
    "model": "모델",
    "policy": "정책",
    "industry": "산업",
    "research": "연구",
    "safety": "안전",
    "other": "기타",
}

# Email clients are unreliable with <style> blocks — Gmail's mobile apps drop them entirely —
# so every rule here is inlined, and the layout leans on tables rather than flex or grid.
FONT = "-apple-system,'Segoe UI','Malgun Gothic','Apple SD Gothic Neo',Arial,sans-serif"
INK, BODY, MUTED, HAIRLINE = "#111827", "#374151", "#6b7280", "#e8edf2"
ACCENT, ACCENT_DARK, ACCENT_SOFT, CANVAS = "#0f766e", "#0b5c55", "#e6fffa", "#eef2f5"


def topic_label(topic: str) -> str:
    return TOPIC_LABELS.get(topic, topic)


def _pill(label: str) -> str:
    return (
        f"<span style=\"display:inline-block;background:{ACCENT_SOFT};color:{ACCENT};"
        f"font-size:11px;font-weight:700;letter-spacing:.02em;line-height:1;"
        f"padding:5px 9px;border-radius:4px\">{escape(label)}</span>"
    )


def render_report(
    articles: list[Article],
    guidance: str,
    report_url: str = "",
    metrics: RunMetrics | None = None,
) -> tuple[str, str]:
    """Render the email. The outline is always inline: Gmail readers should never need a click."""
    today = datetime.now().strftime("%Y년 %m월 %d일")
    domestic = sum(1 for article in articles if article.region == "korea")
    counts: dict[str, int] = {}
    for article in articles:
        counts[article.topic] = counts.get(article.topic, 0) + 1
    spread = " · ".join(f"{topic_label(topic)} {count}" for topic, count in
                        sorted(counts.items(), key=lambda pair: -pair[1]))
    # A short edition is a real signal, not a defect to hide: say why rather than padding it out.
    shortfall = ""
    if metrics is not None and metrics.target_size and len(articles) < metrics.target_size:
        shortfall = f"신선한 신규 기사 기준 {len(articles)}/{metrics.target_size}건"

    # ---- plain text ----
    text = [
        "AI Daily Brief",
        f"{today} · 총 {len(articles)}건 (국내 {domestic} / 해외 {len(articles) - domestic})",
        f"주제 분포: {spread}" if spread else "",
        shortfall,
        "",
    ]

    # ---- html ----
    parts = [
        f"<div style=\"display:none;max-height:0;overflow:hidden;opacity:0\">"
        f"{escape(today)} AI 뉴스 {len(articles)}건 · 기사별 핵심을 개조식으로 정리했습니다.</div>",
        f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0' "
        f"style=\"background:{CANVAS};padding:24px 12px;font-family:{FONT}\"><tr><td align='center'>",
        f"<table role='presentation' width='960' cellpadding='0' cellspacing='0' "
        f"style=\"width:100%;max-width:960px;background:#ffffff;border-radius:12px;overflow:hidden\">",
        # header band
        f"<tr><td style=\"background:{ACCENT};padding:28px 36px\">"
        f"<div style=\"color:#a7f3ea;font-size:11px;font-weight:700;letter-spacing:.14em\">AI DAILY BRIEF</div>"
        f"<div style=\"color:#ffffff;font-size:22px;font-weight:700;margin-top:6px\">{escape(today)}</div>"
        f"<div style=\"color:#c4f5ee;font-size:13px;margin-top:8px\">"
        f"총 {len(articles)}건 · 국내 {domestic} · 해외 {len(articles) - domestic}"
        + (f" · {escape(spread)}" if spread else "")
        + "</div>"
        + (
            f"<div style=\"color:#a7f3ea;font-size:12px;margin-top:6px\">{escape(shortfall)}</div>"
            if shortfall
            else ""
        )
        + "</td></tr>",
    ]

    if not articles:
        parts.append(
            f"<tr><td style=\"padding:36px 36px;color:{BODY};font-size:15px\">"
            "검증을 통과한 신규 기사가 없습니다.</td></tr>"
        )
        text.append("검증을 통과한 신규 기사가 없습니다.")

    for number, article in enumerate(articles, start=1):
        points = outline_points(article)
        label = topic_label(article.topic)
        corroboration = f" · 교차 보도 {article.related_domains}곳" if article.related_domains > 1 else ""

        text.append(f"[{number}] {article.title}")
        text.extend(f"  - {point}" for point in points)
        text.append(f"  {article.source} · {label}{corroboration}")
        text.append(f"  원문: {article.url}")
        text.append("")

        bullets = "".join(
            f"<tr>"
            f"<td valign='top' style=\"width:14px;color:{ACCENT};font-size:16px;line-height:1.8;"
            f"padding:0 0 8px\">·</td>"
            f"<td style=\"color:{BODY};font-size:15.5px;line-height:1.8;padding:0 0 9px\">"
            f"{escape(point)}</td></tr>"
            for point in points
        )
        divider = "" if number == 1 else f"border-top:1px solid {HAIRLINE};"
        parts.append(
            f"<tr><td style=\"{divider}padding:26px 36px 24px\">"
            f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0'><tr>"
            f"<td valign='top' style=\"width:30px;padding:2px 12px 0 0\">"
            f"<div style=\"width:26px;height:26px;background:{ACCENT_SOFT};color:{ACCENT};"
            f"border-radius:13px;font-size:12px;font-weight:700;text-align:center;line-height:26px\">"
            f"{number}</div></td>"
            f"<td valign='top'>"
            f"<div style=\"color:{INK};font-size:18px;font-weight:700;line-height:1.5;"
            f"margin:0 0 10px\">{escape(article.title)}</div>"
            f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0' "
            f"style=\"margin:0 0 12px\">{bullets}</table>"
            f"<div style=\"font-size:12.5px;color:{MUTED};line-height:1.6\">{_pill(label)}"
            f"<span style=\"padding-left:8px\">{escape(article.source)}{escape(corroboration)}</span>"
            f"<span style=\"padding:0 6px;color:#cbd5e1\">|</span>"
            f"<a href=\"{escape(article.url, quote=True)}\" "
            f"style=\"color:{ACCENT};font-weight:600;text-decoration:none\">출처 열기 &rarr;</a>"
            f"</div></td></tr></table></td></tr>"
        )

    footer = [f"<div style=\"margin:0 0 6px\"><strong style=\"color:{BODY}\">편집 원칙</strong> {escape(guidance)}</div>"]
    text.append(f"편집 원칙: {guidance}")
    if metrics is not None:
        footer.append(
            f"<div style=\"margin:0 0 6px\">수집 {metrics.articles_collected}건 → 신규 "
            f"{metrics.stories_new}건 → 선정 {metrics.articles_selected}건 · 출처 "
            f"{metrics.source_diversity}곳 · 본문 수집률 {metrics.body_fetch_ratio:.0%} · "
            f"번역 {escape(metrics.translation_engine)}</div>"
        )
    if report_url and articles:
        link = report_url.rstrip("/")
        footer.append(
            f"<div style=\"margin-top:10px\"><a href=\"{escape(link, quote=True)}\" "
            f"style=\"color:{ACCENT};font-weight:600;text-decoration:none\">웹에서 보기 &rarr;</a></div>"
        )
        text.extend(["", f"웹에서 보기: {link}"])
    parts.append(
        f"<tr><td style=\"border-top:1px solid {HAIRLINE};background:#fafbfc;padding:20px 36px;"
        f"color:{MUTED};font-size:12px;line-height:1.65\">{''.join(footer)}</td></tr>"
    )

    parts.append("</table></td></tr></table>")
    return "\n".join(line for line in text if line is not None), "".join(parts)


def write_web_report(
    articles: list[Article],
    guidance: str,
    path: Path = Path("reports/index.html"),
    metrics: RunMetrics | None = None,
) -> None:
    """Write a browser report whose items open as popups, and this run's quality panel.

    The popup is driven by the CSS :target selector rather than a script, so it works in
    every browser, from a downloaded attachment, and with JavaScript disabled. The small
    script only adds Escape-to-close on top of that.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    headlines = ", ".join(article.title for article in articles[:3]) or "검증을 통과한 신규 기사가 없습니다"
    overview = f"오늘의 핵심 AI 뉴스 {len(articles)}건입니다. 주요 이슈는 {headlines}입니다."

    rows = []
    modals = []
    for number, article in enumerate(articles, start=1):
        points = outline_points(article)
        teaser = points[0] if len(points[0]) <= 90 else points[0][:90].rstrip() + "…"
        corroboration = f" · 교차 보도 {article.related_domains}곳" if article.related_domains > 1 else ""
        meta = f"{article.source}{corroboration}"
        rows.append(
            f"<li class='row'><a class='open' href='#detail-{number}'>"
            f"<span class='num'>{number}</span>"
            f"<span class='text'><span class='headline'>{escape(article.title)}</span>"
            f"<span class='teaser'>{escape(teaser)}</span>"
            f"<span class='meta'><span class='tag'>{escape(topic_label(article.topic))}</span>{escape(meta)}</span></span>"
            f"<span class='cue' aria-hidden='true'>＋</span></a></li>"
        )
        modals.append(
            f"<div class='modal' id='detail-{number}' role='dialog' aria-modal='true' "
            f"aria-labelledby='title-{number}'>"
            f"<a class='backdrop' href='#close' aria-label='닫기'></a>"
            f"<div class='card'>"
            f"<a class='x' href='#close' aria-label='닫기'>×</a>"
            f"<p class='eyebrow'><span class='tag'>{escape(topic_label(article.topic))}</span>"
            f"{escape(meta)}</p>"
            f"<h3 id='title-{number}'>{number}. {escape(article.title)}</h3>"
            f"<ul class='detail'>{''.join(f'<li>{escape(point)}</li>' for point in points)}</ul>"
            f"<p class='actions'>"
            f"<a class='primary' href='{escape(article.url, quote=True)}' target='_blank' rel='noopener'>"
            f"원문 열기 ↗</a>"
            f"<a class='ghost' href='#close'>닫기</a></p>"
            f"</div></div>"
        )

    body = f"<ol class='list'>{''.join(rows)}</ol>" if rows else "<p>검증을 통과한 신규 기사가 없습니다.</p>"
    page = f"""<!doctype html>
<html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>AI Daily Brief</title><style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f0fdfa;color:#172033;font-family:-apple-system,'Segoe UI','Malgun Gothic',Arial,sans-serif;line-height:1.7}}
main{{max-width:760px;margin:36px auto;padding:0 20px}}
h1{{color:#0f766e;margin:0 0 18px}}
h2{{color:#134e4a;margin:32px 0 12px;padding-bottom:8px;border-bottom:2px solid #99f6e4}}
.intro{{background:#fff;padding:20px;border-left:4px solid #14b8a6;border-radius:0 6px 6px 0}}
.hint{{color:#4b5563;font-size:13px;margin:0 0 12px}}
.list{{list-style:none;margin:0;padding:0}}
.row{{margin:10px 0}}
.open{{display:flex;align-items:flex-start;gap:12px;padding:14px 16px;background:#fff;border:1px solid #ccfbf1;border-radius:8px;text-decoration:none;color:inherit}}
.open:hover{{border-color:#5eead4;box-shadow:0 1px 3px rgba(15,118,110,.14)}}
.open:focus-visible{{outline:2px solid #14b8a6;outline-offset:2px}}
.num{{flex:none;width:26px;height:26px;margin-top:2px;border-radius:50%;background:#ccfbf1;color:#0f766e;font-size:13px;display:flex;align-items:center;justify-content:center}}
.text{{flex:1;min-width:0}}
.headline{{display:block;font-weight:700;color:#0f766e}}
.teaser{{display:block;color:#334155;font-size:14px;margin-top:2px}}
.meta{{display:block;color:#4b5563;font-size:12.5px;margin-top:4px}}
.tag{{display:inline-block;background:#f0fdfa;border:1px solid #ccfbf1;border-radius:4px;padding:1px 7px;margin-right:8px;font-size:12px;color:#0f766e}}
.cue{{flex:none;color:#14b8a6;font-size:18px;line-height:1.4}}
.modal{{display:none;position:fixed;inset:0;z-index:50;align-items:center;justify-content:center;padding:20px}}
.modal:target{{display:flex}}
.backdrop{{position:absolute;inset:0;background:rgba(15,23,42,.55)}}
.card{{position:relative;background:#fff;border-radius:12px;max-width:640px;width:100%;max-height:85vh;overflow:auto;padding:26px 28px;box-shadow:0 18px 44px rgba(15,23,42,.28)}}
.x{{position:absolute;top:12px;right:16px;font-size:26px;line-height:1;color:#64748b;text-decoration:none}}
.x:hover{{color:#0f766e}}
.eyebrow{{margin:0 0 6px;color:#4b5563;font-size:13px}}
.card h3{{margin:0 0 14px;color:#0f766e;padding-right:28px}}
.card .detail{{margin:0 0 20px;padding-left:20px}}
.card .detail li{{margin:0 0 8px}}
.actions{{margin:0;display:flex;gap:10px;flex-wrap:wrap}}
.actions a{{font-size:14px;text-decoration:none;border-radius:999px;padding:8px 18px}}
.primary{{background:#0f766e;color:#fff}}
.primary:hover{{background:#115e59}}
.ghost{{border:1px solid #ccfbf1;color:#0f766e}}
.ghost:hover{{background:#f0fdfa}}
.quality{{margin-top:28px;padding:16px;background:#fff;border:1px dashed #99f6e4;color:#4b5563;font-size:13px;border-radius:8px}}
.quality ul{{margin:8px 0 0;padding-left:18px}}
a{{color:#0f766e}}
</style></head><body><main>
<h1>AI Daily Brief</h1>
<div class='intro'><strong>오늘의 요약</strong><br>{escape(overview)}<br><small>편집 원칙: {escape(guidance)}</small></div>
<h2>주요 뉴스</h2>
<p class='hint'>제목을 클릭하면 상세 내용이 팝업으로 열립니다.</p>
{body}{_quality_panel(metrics)}
</main>{''.join(modals)}<script>
document.addEventListener('keydown', function (event) {{
  if (event.key === 'Escape' && location.hash && location.hash !== '#close') {{
    location.hash = '#close';
  }}
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


def send_gmail(subject: str, text: str, html: str, attachment: Path | None = None) -> None:
    """Send the brief. Gmail strips scripts and <details>, so the interactive copy rides along
    as an attachment: opening it gives the reader the popup view without any hosting."""
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
    if attachment is not None:
        try:
            payload = attachment.read_text(encoding="utf-8")
        except OSError as error:
            LOGGER.warning("interactive report not attached (%s): %s", attachment, error)
        else:
            message.add_attachment(
                payload.encode("utf-8"),
                maintype="text",
                subtype="html",
                filename=f"AI-Daily-Brief-{datetime.now().strftime('%Y-%m-%d')}.html",
            )
            LOGGER.info("attached the interactive report (%s KB)", round(len(payload.encode()) / 1024))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(username, password)
        smtp.send_message(message)
    LOGGER.info("report delivered to %s recipient(s)", len(recipients))

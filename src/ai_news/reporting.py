import os
import smtplib
import json
from email.message import EmailMessage
from html import escape
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .models import Article


def enrich_with_article_bodies(articles: list[Article]) -> list[Article]:
    """Retrieve readable source text before creating the email; failures keep RSS summaries."""
    enriched = []
    for article in articles:
        body = _fetch_article_body(article.url)
        enriched.append(Article(**{**article.__dict__, "body": body}))
    return enriched


def _fetch_article_body(url: str) -> str:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AI-News-Reporter/1.0)"},
            timeout=15,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in soup.select("article p, main p")]
        if not paragraphs:
            paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in soup.find_all("p")]
        return " ".join(paragraphs)[:6000]
    except (requests.RequestException, ValueError):
        return ""


def translate_to_korean(articles: list[Article]) -> list[Article]:
    """Translate non-Korean article text, without blocking delivery on provider errors."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not articles:
        return articles
    payload = [
        {"index": index, "title": article.title, "summary": article.summary[:500]}
        for index, article in enumerate(articles)
        if article.region != "korea"
    ]
    if not payload:
        return articles
    translations = _translate_with_gemini(payload, gemini_api_key) if gemini_api_key else None
    if translations is None and openai_api_key:
        translations = _translate_with_openai(payload, openai_api_key)
    if translations is None:
        translations = _translate_with_google(payload)
    if translations is None:
        return articles
    translated = list(articles)
    for item in translations:
        index = item["index"]
        original = translated[index]
        translated[index] = Article(**{**original.__dict__, "title": item["title"], "summary": item["summary"]})
    return translated


def summarize_article_bodies(articles: list[Article]) -> list[Article]:
    """Produce a 2-3 sentence Korean brief from collected article text when Gemini is available."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not articles:
        return [Article(**{**article.__dict__, "detail_summary": article.summary[:550]}) for article in articles]
    payload = [
        {"index": index, "title": article.title, "body": (article.body or article.summary)[:3500]}
        for index, article in enumerate(articles)
    ]
    try:
        prompt = (
            "Summarize each news article in Korean in 2 or 3 concise sentences. "
            "Use only supplied text, preserve key entities and numbers, and return only JSON: "
            '{"summaries":[{"index":0,"summary":"..."}]}.\n' + json.dumps(payload, ensure_ascii=False)
        )
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            timeout=45,
        )
        response.raise_for_status()
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        summaries = json.loads(content)["summaries"]
        by_index = {item["index"]: item["summary"] for item in summaries}
        return [
            Article(**{**article.__dict__, "detail_summary": str(by_index.get(index, article.summary))[:900]})
            for index, article in enumerate(articles)
        ]
    except (KeyError, IndexError, TypeError, ValueError, requests.RequestException):
        return [Article(**{**article.__dict__, "detail_summary": article.summary[:550]}) for article in articles]


def _translate_with_gemini(payload: list[dict[str, object]], api_key: str) -> list[dict[str, object]] | None:
    try:
        prompt = (
            "Translate each English AI-news title and summary into natural, concise Korean. "
            "Return only a JSON object with a translations array; each item must contain index, title, and summary. "
            "Do not add facts.\n" + json.dumps(payload, ensure_ascii=False)
        )
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        translations = json.loads(content)["translations"]
        if not isinstance(translations, list):
            return None
        return translations
    except (KeyError, IndexError, TypeError, ValueError, requests.RequestException):
        return None


def _translate_with_openai(payload: list[dict[str, object]], api_key: str) -> list[dict[str, object]] | None:
    try:
        from openai import OpenAI

        prompt = (
            "Translate each English AI-news title and summary into natural, concise Korean. "
            "Return only a JSON object with a translations array; each item must contain index, title, and summary. "
            "Do not add facts.\n" + json.dumps(payload, ensure_ascii=False)
        )
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)["translations"]
    except Exception:
        return None


def _translate_with_google(payload: list[dict[str, object]]) -> list[dict[str, object]] | None:
    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="auto", target="ko")
        titles = translator.translate_batch([str(item["title"]) for item in payload])
        summaries = translator.translate_batch([str(item["summary"]) for item in payload])
        if any(value.lstrip().lower().startswith("error ") for value in titles + summaries):
            return None
        return [
            {"index": item["index"], "title": title, "summary": summary}
            for item, title, summary in zip(payload, titles, summaries, strict=True)
        ]
    except Exception:
        return None


def render_report(articles: list[Article], guidance: str, report_url: str = "") -> tuple[str, str]:
    headlines = ", ".join(article.title for article in articles[:3]) or "검증을 통과한 신규 기사가 없습니다"
    overview = f"오늘의 핵심 AI 뉴스 {len(articles)}건입니다. 주요 이슈는 {headlines}입니다. 국내외 정책, 모델, 산업 동향을 함께 선별했습니다."
    text = ["AI Daily Brief", overview, f"편집 원칙: {guidance}", ""]
    html = [
        "<div style='max-width:720px;margin:auto;font-family:Arial,sans-serif;color:#1f2937;line-height:1.65'>",
        "<h1 style='color:#0f766e'>AI Daily Brief</h1>",
        f"<p style='font-size:16px'><strong>오늘의 요약</strong><br>{escape(overview)}</p>",
        f"<p style='color:#4b5563'><strong>편집 원칙:</strong> {escape(guidance)}</p>",
    ]
    html.append("<h2 style='border-bottom:2px solid #99f6e4'>상세 요약</h2>")
    for detail_number, article in enumerate(articles, start=1):
        detail = article.detail_summary or article.summary[:550].strip() or "원문 본문을 가져오지 못했습니다."
        text.extend([f"## {detail_number}. {article.title}", detail, f"{article.source}: {article.url}"])
        if report_url:
            detail_link = f"{report_url.rstrip('/')}/#detail-{detail_number}"
            html.append(
                f"<section style='padding:14px 0;border-bottom:1px solid #d1d5db'>"
                f"<h3>{detail_number}. {escape(article.title)}</h3>"
                f"<p><small>{escape(article.source)} | <a href='{escape(detail_link, quote=True)}'>상세 요약 보기</a> | "
                f"원문: <a href='{escape(article.url, quote=True)}'>출처 열기</a></small></p></section>"
            )
            continue
        html.append(
            f"<section id='detail-{detail_number}' style='padding:14px 0;border-bottom:1px solid #d1d5db'>"
            f"<h3>{detail_number}. {escape(article.title)}</h3><p>{escape(detail)}</p>"
            f"<p><small>{escape(article.source)} | 원문: <a href='{escape(article.url, quote=True)}'>출처 열기</a> | "
            f"{escape(article.topic)}</small></p></section>"
        )
    if not articles:
        text.append("검증을 통과한 신규 기사가 없습니다.")
        html.append("<p>검증을 통과한 신규 기사가 없습니다.</p>")
    html.append("</div>")
    return "\n".join(text), "".join(html)


def write_web_report(articles: list[Article], guidance: str, path: Path = Path("reports/index.html")) -> None:
    """Write a browser report with native expandable details for GitHub Pages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headlines = ", ".join(article.title for article in articles[:3]) or "검증을 통과한 신규 기사가 없습니다"
    overview = f"오늘의 핵심 AI 뉴스 {len(articles)}건입니다. 주요 이슈는 {headlines}입니다."
    sections = []
    for number, article in enumerate(articles, start=1):
        detail = article.detail_summary or article.summary[:550].strip() or "원문 본문을 가져오지 못했습니다."
        sections.append(
            f"<details id='detail-{number}'><summary>{number}. {escape(article.title)}</summary>"
            f"<p>{escape(detail)}</p><p class='meta'>{escape(article.source)} | {escape(article.topic)} | "
            f"<a href='{escape(article.url, quote=True)}' target='_blank' rel='noopener'>출처 열기</a></p></details>"
        )
    page = f"""<!doctype html>
<html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>AI Daily Brief</title><style>
body{{margin:0;background:#f0fdfa;color:#172033;font-family:Arial,sans-serif;line-height:1.7}}
main{{max-width:760px;margin:36px auto;padding:0 20px}}h1{{color:#0f766e}}.intro{{background:#fff;padding:20px;border-left:4px solid #14b8a6}}
details{{background:#fff;margin:12px 0;padding:16px;border:1px solid #ccfbf1}}summary{{cursor:pointer;font-weight:700;color:#0f766e}}.meta{{color:#4b5563;font-size:14px}}a{{color:#0f766e}}
</style></head><body><main><h1>AI Daily Brief</h1><div class='intro'><strong>오늘의 요약</strong><br>{escape(overview)}<br><small>편집 원칙: {escape(guidance)}</small></div>
<h2>상세 요약</h2>{''.join(sections) or '<p>검증을 통과한 신규 기사가 없습니다.</p>'}</main></body></html>"""
    path.write_text(page, encoding="utf-8")


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
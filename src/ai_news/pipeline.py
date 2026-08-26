import hashlib
import html
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import feedparser

from .models import Article, Evaluation
from .sources import DEFAULT_SOURCES, Source

TOPICS = {
    "model": ("model", "llm", "gpt", "gemini", "claude", "모델", "생성형"),
    "policy": ("policy", "regulation", "law", "governance", "정책", "규제", "법"),
    "industry": ("funding", "startup", "enterprise", "chip", "투자", "기업", "반도체"),
    "research": ("research", "paper", "benchmark", "연구", "논문", "벤치마크"),
    "safety": ("safety", "security", "risk", "alignment", "안전", "보안", "위험"),
}


class NewsStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS seen (fingerprint TEXT PRIMARY KEY, seen_at TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS evaluations (run_at TEXT, collected INTEGER, selected INTEGER, valid_ratio REAL, diversity INTEGER, note TEXT)"
        )

    def is_new(self, article: Article) -> bool:
        fingerprint = self._fingerprint(article)
        result = self.connection.execute("SELECT 1 FROM seen WHERE fingerprint = ?", (fingerprint,)).fetchone()
        return result is None

    def mark_seen(self, articles: list[Article]) -> None:
        self.connection.executemany(
            "INSERT OR IGNORE INTO seen VALUES (?, ?)",
            [(self._fingerprint(article), datetime.now(UTC).isoformat()) for article in articles],
        )
        self.connection.commit()

    def save_evaluation(self, evaluation: Evaluation) -> None:
        self.connection.execute(
            "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?)",
            (evaluation.run_at.isoformat(), evaluation.articles_collected, evaluation.articles_selected,
             evaluation.valid_url_ratio, evaluation.source_diversity, evaluation.improvement_note),
        )
        self.connection.commit()

    def latest_guidance(self) -> str:
        row = self.connection.execute("SELECT note FROM evaluations ORDER BY run_at DESC LIMIT 1").fetchone()
        return row[0] if row else "초기 실행: 국내외 균형과 공식 출처 우선 원칙을 적용하세요."

    @staticmethod
    def _fingerprint(article: Article) -> str:
        return hashlib.sha256(f"{article.title.lower()}|{urlparse(article.url).netloc}".encode()).hexdigest()


def collect(sources: tuple[Source, ...] = DEFAULT_SOURCES) -> list[Article]:
    articles = []
    for source in sources:
        feed = feedparser.parse(source.url)
        for entry in feed.entries[:20]:
            url = entry.get("link", "").strip()
            title = html.unescape(re.sub(r"<[^>]+>", "", entry.get("title", "")).strip())
            summary = html.unescape(re.sub(r"<[^>]+>", " ", entry.get("summary", "")).strip())
            if not title or not url:
                continue
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            published_at = datetime(*published[:6], tzinfo=UTC) if published else datetime.now(UTC)
            articles.append(Article(title, url, summary, published_at, source.name, source.region, source.trust))
    return articles


def verify_and_rank(articles: list[Article], store: NewsStore, force: bool = False) -> list[Article]:
    domains = Counter(urlparse(article.url).netloc for article in articles)
    selected = []
    now = datetime.now(UTC)
    for article in articles:
        parsed = urlparse(article.url)
        if parsed.scheme != "https" or not parsed.netloc or (not force and not store.is_new(article)):
            continue
        text = f"{article.title} {article.summary}".lower()
        topic = next((name for name, terms in TOPICS.items() if any(term in text for term in terms)), "other")
        age_hours = max(0, (now - article.published_at).total_seconds() / 3600)
        freshness = max(0.1, 1 - age_hours / 168)
        corroboration = min(1.0, domains[parsed.netloc] / 3)
        relevance = 1.0 if topic != "other" else 0.55
        score = round(100 * (0.40 * article.trust + 0.25 * freshness + 0.20 * relevance + 0.15 * corroboration), 1)
        selected.append(Article(**{**article.__dict__, "topic": topic, "score": score}))
    ranked = sorted(selected, key=lambda article: article.score, reverse=True)
    domestic = [article for article in ranked if article.region == "korea"][:4]
    report = domestic + [article for article in ranked if article not in domestic][:12 - len(domestic)]
    report = sorted(report, key=lambda article: article.score, reverse=True)
    if not force:
        store.mark_seen(report)
    return report


def evaluate(collected: list[Article], selected: list[Article]) -> Evaluation:
    valid_ratio = sum(urlparse(article.url).scheme == "https" for article in selected) / max(1, len(selected))
    diversity = len({article.source for article in selected})
    note = (
        "다음 실행에서는 국내 소스를 우선 보강하세요."
        if not any(article.region == "korea" for article in selected)
        else "상위 기사는 공식 출처와 다수 매체 보도로 계속 교차 검증하세요."
    )
    return Evaluation(datetime.now(UTC), len(collected), len(selected), valid_ratio, diversity, note)
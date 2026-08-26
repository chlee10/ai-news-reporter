import hashlib
import html
import re
import statistics
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from urllib.parse import urlparse

import feedparser

from .fetch import FetchError
from .fetch import get as http_get
from .models import Article, EditorialPolicy, RunMetrics
from .observability import get_logger
from .sources import Source, load_sources
from .store import NewsStore

LOGGER = get_logger("pipeline")

TOPICS = {
    "model": ("model", "llm", "gpt", "gemini", "claude", "모델", "생성형"),
    "policy": ("policy", "regulation", "law", "governance", "정책", "규제", "법"),
    "industry": ("funding", "startup", "enterprise", "chip", "투자", "기업", "반도체"),
    "research": ("research", "paper", "benchmark", "연구", "논문", "벤치마크"),
    "safety": ("safety", "security", "risk", "alignment", "안전", "보안", "위험"),
}

STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "is", "are", "be", "as", "at",
        "by", "from", "its", "it", "this", "that", "new", "says", "said", "will", "has", "have", "how", "why",
        "ai", "인공지능", "관련", "대한", "위한", "통해", "지난", "오늘", "기업", "발표", "공개",
    }
)
SIMILARITY_THRESHOLD = 0.55
MAX_ENTRIES_PER_SOURCE = 20

# Several Korean and vendor feeds are general tech, not AI-only. Relevance scoring alone only
# downweights an off-topic item, so the domestic quota could still pull one in. This gate excludes it.
AI_TERMS = (
    "인공지능", "머신러닝", "딥러닝", "생성형", "챗봇", "에이아이", "언어모델", "초거대",
    "llm", "gpt", "chatgpt", "openai", "anthropic", "claude", "gemini", "copilot", "deepmind",
    "machine learning", "deep learning", "neural", "transformer", "chatbot", "agentic",
)
AI_TOKEN = re.compile(r"(?<![a-z])(ai|a\.i\.)(?![a-z])")


@dataclass
class CollectResult:
    articles: list[Article] = field(default_factory=list)
    ok_sources: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)


@dataclass
class Story:
    """One real-world event, possibly reported by several outlets."""

    representative: Article
    members: list[Article]

    @property
    def domains(self) -> set[str]:
        return {urlparse(article.url).netloc for article in self.members}

    @property
    def signature(self) -> frozenset[str]:
        return signature_of(self.representative.title)


# ---------------------------------------------------------------- collection


def collect(sources: tuple[Source, ...] | None = None) -> CollectResult:
    """Read every feed with a real timeout. A dead feed is recorded, never silently skipped."""
    sources = load_sources() if sources is None else sources
    result = CollectResult()
    for source in sources:
        try:
            response = http_get(source.url, timeout=15.0, attempts=3)
        except FetchError as error:
            LOGGER.error("source unreachable: %s (%s)", source.name, error)
            result.failures[source.name] = str(error)
            continue
        feed = feedparser.parse(response.content)
        if feed.bozo and not feed.entries:
            reason = str(getattr(feed, "bozo_exception", "unparseable feed"))
            LOGGER.error("source returned an unusable feed: %s (%s)", source.name, reason)
            result.failures[source.name] = reason
            continue
        parsed = list(_entries_to_articles(feed.entries[:MAX_ENTRIES_PER_SOURCE], source))
        if not parsed:
            LOGGER.warning("source produced no usable entries: %s", source.name)
            result.failures[source.name] = "no usable entries"
            continue
        result.articles.extend(parsed)
        result.ok_sources.append(source.name)
        LOGGER.info("collected %s entries from %s", len(parsed), source.name)
    LOGGER.info(
        "collection finished: %s articles from %s/%s sources",
        len(result.articles), len(result.ok_sources), len(sources),
    )
    return result


def _entries_to_articles(entries: list, source: Source):
    for entry in entries:
        url = entry.get("link", "").strip()
        title = html.unescape(re.sub(r"<[^>]+>", "", entry.get("title", ""))).strip()
        summary = html.unescape(re.sub(r"<[^>]+>", " ", entry.get("summary", ""))).strip()
        if not title or not url:
            continue
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        published_at = datetime(*published[:6], tzinfo=UTC) if published else datetime.now(UTC)
        yield Article(
            title=title,
            url=url,
            summary=summary,
            published_at=published_at,
            source=source.name,
            region=source.region,
            trust=source.trust,
            signature=" ".join(sorted(signature_of(title))),
            fingerprint=fingerprint_of(title, url),
        )


# ---------------------------------------------------------------- deduplication


def signature_of(title: str) -> frozenset[str]:
    """Content tokens that identify a story regardless of which outlet wrote the headline."""
    normalized = re.sub(r"[^0-9a-z가-힣\s]", " ", title.lower())
    return frozenset(token for token in normalized.split() if len(token) > 1 and token not in STOPWORDS)


def fingerprint_of(title: str, url: str) -> str:
    return hashlib.sha256(f"{title.strip().lower()}|{urlparse(url).netloc}".encode()).hexdigest()


def similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def cluster_stories(articles: list[Article]) -> list[Story]:
    """Group outlets covering the same event so corroboration counts distinct domains, not repeat posts."""
    stories: list[Story] = []
    for article in sorted(articles, key=lambda item: (-item.trust, item.published_at)):
        tokens = signature_of(article.title)
        match = next((story for story in stories if similarity(story.signature, tokens) >= SIMILARITY_THRESHOLD), None)
        if match is None:
            stories.append(Story(article, [article]))
            continue
        match.members.append(article)
        if article.trust > match.representative.trust:
            match.representative = article
    LOGGER.info("clustered %s articles into %s distinct stories", len(articles), len(stories))
    return stories


def filter_new(stories: list[Story], store: NewsStore, force: bool = False) -> list[Story]:
    """Drop stories already delivered, including the same event republished by another outlet."""
    if force:
        LOGGER.warning("--force in effect: dedup memory ignored for this run")
        return stories
    known = store.known_fingerprints()
    recent = store.recent_signatures()
    fresh = []
    for story in stories:
        if any(member.fingerprint in known for member in story.members):
            continue
        if any(similarity(story.signature, seen) >= SIMILARITY_THRESHOLD for _, seen in recent):
            LOGGER.debug("cross-outlet duplicate skipped: %s", story.representative.title)
            continue
        fresh.append(story)
    LOGGER.info("%s/%s stories are new since the last delivery", len(fresh), len(stories))
    return fresh


# ---------------------------------------------------------------- ranking


def is_ai_related(article: Article) -> bool:
    """Hard gate. A general-tech feed must not smuggle an off-topic item in through the quota."""
    text = f"{article.title} {article.summary}".lower()
    return bool(AI_TOKEN.search(text)) or any(term in text for term in AI_TERMS)


def rank(stories: list[Story], policy: EditorialPolicy) -> list[Article]:
    """Score every story under the current policy, whose weights come from the previous run."""
    now = datetime.now(UTC)
    ranked = []
    off_topic = 0
    stale = 0
    for story in stories:
        article = story.representative
        if urlparse(article.url).scheme != "https" or not urlparse(article.url).netloc:
            LOGGER.warning("dropped story with an unverifiable URL: %s", article.url)
            continue
        if not is_ai_related(article):
            off_topic += 1
            continue
        age_hours = max(0.0, (now - article.published_at).total_seconds() / 3600)
        if age_hours > policy.max_age_hours:
            stale += 1
            continue
        text = f"{article.title} {article.summary}".lower()
        topic = next((name for name, terms in TOPICS.items() if any(term in text for term in terms)), "other")
        freshness = max(0.1, 1 - age_hours / 168)
        corroboration = min(1.0, (len(story.domains) - 1) / 2)
        relevance = 1.0 if topic != "other" else 0.55
        trust = min(1.0, article.trust * policy.reliability(article.source))
        score = round(
            100
            * (
                policy.trust_weight * trust
                + policy.freshness_weight * freshness
                + policy.relevance_weight * relevance
                + policy.corroboration_weight * corroboration
            ),
            1,
        )
        ranked.append(replace(article, topic=topic, score=score, related_domains=len(story.domains)))
    if off_topic:
        LOGGER.info("filtered out %s stories with no AI signal", off_topic)
    if stale:
        LOGGER.info(
            "filtered out %s stories older than the %.0f-hour freshness floor", stale, policy.max_age_hours
        )
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def select(ranked: list[Article], policy: EditorialPolicy) -> list[Article]:
    """Fill the domestic quota first, then the rest, capping how much any single outlet can occupy."""
    if not ranked:
        return []
    chosen: list[Article] = []
    per_source: dict[str, int] = {}

    def take(candidates: list[Article], limit: int, cap: int) -> None:
        remaining = limit
        for article in candidates:
            if len(chosen) >= policy.report_size or remaining <= 0:
                return
            if per_source.get(article.source, 0) >= cap or article in chosen:
                continue
            chosen.append(article)
            per_source[article.source] = per_source.get(article.source, 0) + 1
            remaining -= 1

    # The domestic quota is a floor, so it outranks the per-source cap when few outlets cover Korea.
    domestic = [article for article in ranked if article.region == "korea"]
    take(domestic, policy.domestic_quota, max(policy.max_per_source, policy.domestic_quota))
    take(ranked, policy.report_size - len(chosen), policy.max_per_source)
    if len(chosen) < policy.report_size:
        LOGGER.warning(
            "only %s/%s slots filled under a %s-per-source cap; relaxing the cap for this run",
            len(chosen), policy.report_size, policy.max_per_source,
        )
        take(ranked, policy.report_size - len(chosen), policy.report_size)
    return sorted(chosen, key=lambda item: item.score, reverse=True)


# ---------------------------------------------------------------- evaluation and improvement


def measure(
    run_at: datetime,
    collected: CollectResult,
    stories: list[Story],
    fresh: list[Story],
    selected: list[Article],
    policy: EditorialPolicy | None = None,
) -> RunMetrics:
    """Record what actually happened. Nothing here is derivable from a filter applied upstream."""
    now = datetime.now(UTC)
    policy = policy or EditorialPolicy()
    ages = [max(0.0, (now - article.published_at).total_seconds() / 3600) for article in selected]
    stale = sum(
        1
        for story in fresh
        if (now - story.representative.published_at).total_seconds() / 3600 > policy.max_age_hours
    )
    return RunMetrics(
        run_at=run_at,
        sources_ok=len(collected.ok_sources),
        sources_failed=len(collected.failures),
        articles_collected=len(collected.articles),
        stories_clustered=len(stories),
        stories_new=len(fresh),
        articles_selected=len(selected),
        target_size=policy.report_size,
        stale_dropped=stale,
        domestic_selected=sum(article.region == "korea" for article in selected),
        source_diversity=len({article.source for article in selected}),
        median_age_hours=round(statistics.median(ages), 1) if ages else 0.0,
    )


def improve(metrics: RunMetrics, policy: EditorialPolicy) -> EditorialPolicy:
    """Turn measurements into the next run's policy. This is the step that makes it a harness."""
    notes: list[str] = []
    domestic_quota = policy.domestic_quota
    max_per_source = policy.max_per_source
    weights = {
        "trust_weight": policy.trust_weight,
        "freshness_weight": policy.freshness_weight,
        "relevance_weight": policy.relevance_weight,
        "corroboration_weight": policy.corroboration_weight,
    }

    if metrics.articles_selected and metrics.domestic_ratio < 0.25:
        domestic_quota = min(6, domestic_quota + 1)
        notes.append(f"국내 비중 {metrics.domestic_ratio:.0%} → 국내 최소 편성을 {domestic_quota}건으로 상향합니다.")
    elif metrics.domestic_ratio > 0.5:
        domestic_quota = max(2, domestic_quota - 1)
        notes.append(f"국내 비중 {metrics.domestic_ratio:.0%} → 해외 동향 비중을 회복합니다.")

    if metrics.articles_selected and metrics.source_diversity < 5:
        max_per_source = max(2, max_per_source - 1)
        notes.append(f"출처 {metrics.source_diversity}곳에 편중 → 매체당 상한을 {max_per_source}건으로 조입니다.")
    elif metrics.source_diversity >= 8 and max_per_source < 3:
        max_per_source += 1

    if metrics.median_age_hours > 48:
        weights["freshness_weight"] += 0.03
        weights["trust_weight"] = max(0.25, weights["trust_weight"] - 0.03)
        notes.append(f"기사 중위 연령 {metrics.median_age_hours:.0f}시간 → 최신성 가중치를 높입니다.")
    elif 0 < metrics.median_age_hours < 12 and weights["freshness_weight"] > 0.20:
        weights["freshness_weight"] -= 0.02
        weights["trust_weight"] += 0.02

    if metrics.target_size and metrics.articles_selected < metrics.target_size:
        notes.append(
            f"신선한 신규 기사가 부족해 {metrics.articles_selected}/{metrics.target_size}건만 편성했습니다"
            f"(신선도 하한 {policy.max_age_hours:.0f}시간 초과 {metrics.stale_dropped}건 제외)."
        )
    if metrics.articles_selected and metrics.body_fetch_ratio < 0.6:
        notes.append(f"본문 수집률 {metrics.body_fetch_ratio:.0%} → 원문 접근이 막힌 매체를 감점합니다.")
    if metrics.translation_engine == "google":
        notes.append("Gemini 번역이 실패해 Google 번역으로 대체되었습니다. GEMINI_API_KEY를 점검하세요.")
    elif metrics.translation_engine == "none" and metrics.articles_selected:
        notes.append("번역 엔진을 사용하지 못해 해외 기사가 원문 그대로 나갔습니다.")
    if metrics.summary_engine == "fallback":
        notes.append("상세 요약이 RSS 발췌로 대체되었습니다. Gemini 요약 경로를 확인하세요.")
    if metrics.sources_failed:
        notes.append(f"피드 {metrics.sources_failed}곳이 응답하지 않았습니다. 소스 상태를 확인하세요.")
    if not metrics.delivered and metrics.articles_selected:
        notes.append("발송에 실패해 기사를 미발송 상태로 유지합니다. 다음 실행에서 재시도합니다.")
    if not metrics.stories_new:
        notes.append("신규 사건이 없어 이번 회차는 발송을 건너뛰었습니다.")

    total = sum(weights.values())
    weights = {key: round(value / total, 4) for key, value in weights.items()}
    note = " ".join(notes) or "지표가 목표 범위 안에 있습니다. 공식 출처 우선과 교차 검증 기조를 유지합니다."
    next_policy = EditorialPolicy(
        **weights,
        domestic_quota=domestic_quota,
        report_size=policy.report_size,
        max_age_hours=policy.max_age_hours,
        max_per_source=max_per_source,
        source_reliability=dict(policy.source_reliability),
        note=note,
        revision=policy.revision + 1,
    )
    LOGGER.info("policy revision %s: %s", next_policy.revision, note)
    return next_policy


def adjust_reliability(policy: EditorialPolicy, outcomes: dict[str, bool]) -> EditorialPolicy:
    """Sources whose articles cannot be read lose weight; recovering sources earn it back."""
    reliability = dict(policy.source_reliability)
    for source, ok in outcomes.items():
        current = reliability.get(source, 1.0)
        reliability[source] = round(min(1.0, current * 1.05) if ok else max(0.6, current * 0.9), 3)
    return replace(policy, source_reliability=reliability)

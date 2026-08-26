import os
from dataclasses import dataclass
from urllib.parse import urlparse

from .observability import get_logger

LOGGER = get_logger("sources")


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    region: str
    trust: float


# Verified reachable on 2026-08-26. Feeds that are not AI-only carry lower trust,
# and the AI relevance gate in pipeline.rank keeps their off-topic items out.
DEFAULT_SOURCES = (
    Source("OpenAI", "https://openai.com/news/rss.xml", "global", 1.0),
    Source("Google AI", "https://blog.google/technology/ai/rss/", "global", 1.0),
    Source("Google DeepMind", "https://deepmind.google/blog/rss.xml", "global", 1.0),
    Source("Microsoft", "https://blogs.microsoft.com/feed/", "global", 0.85),
    Source("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "global", 0.8),
    Source("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed/", "global", 0.9),
    Source("VentureBeat AI", "https://venturebeat.com/category/ai/feed/", "global", 0.8),
    Source("AI타임스", "https://www.aitimes.com/rss/allArticle.xml", "korea", 0.75),
    Source("AI타임스케이", "https://www.aitimes.kr/rss/allArticle.xml", "korea", 0.65),
    Source("ZDNet Korea", "https://feeds.feedburner.com/zdkorea", "korea", 0.7),
    Source("전자신문 IT", "https://rss.etnews.com/Section902.xml", "korea", 0.7),
)


def load_sources(extra_feeds: str | None = None) -> tuple[Source, ...]:
    """Default feeds plus anything in EXTRA_FEEDS, so a feed can be added without code changes.

    Each entry is a URL, optionally `name|url|region|trust`. Region defaults to korea for .kr hosts.
    """
    raw = extra_feeds if extra_feeds is not None else os.getenv("EXTRA_FEEDS", "")
    known = {source.url for source in DEFAULT_SOURCES}
    extras: list[Source] = []
    for entry in (item.strip() for item in raw.split(",")):
        if not entry:
            continue
        source = _parse_entry(entry)
        if source is None or source.url in known:
            continue
        known.add(source.url)
        extras.append(source)
    if extras:
        LOGGER.info("added %s feed(s) from EXTRA_FEEDS: %s", len(extras), ", ".join(item.name for item in extras))
    return DEFAULT_SOURCES + tuple(extras)


def _parse_entry(entry: str) -> Source | None:
    parts = [part.strip() for part in entry.split("|")]
    if len(parts) == 1:
        name, url, region, trust = "", parts[0], "", ""
    elif len(parts) >= 2:
        name, url = parts[0], parts[1]
        region = parts[2] if len(parts) > 2 else ""
        trust = parts[3] if len(parts) > 3 else ""
    else:
        return None

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        LOGGER.warning("ignoring malformed EXTRA_FEEDS entry: %r", entry)
        return None
    try:
        trust_value = min(1.0, max(0.0, float(trust))) if trust else 0.6
    except ValueError:
        LOGGER.warning("ignoring non-numeric trust in EXTRA_FEEDS entry: %r", entry)
        trust_value = 0.6
    if region not in ("korea", "global"):
        region = "korea" if parsed.netloc.endswith(".kr") else "global"
    return Source(name or parsed.netloc, url, region, trust_value)

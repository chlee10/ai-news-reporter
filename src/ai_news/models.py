from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    summary: str
    published_at: datetime
    source: str
    region: str
    trust: float
    topic: str = "other"
    score: float = 0.0
    body: str = ""
    detail_summary: str = ""


@dataclass(frozen=True)
class Evaluation:
    run_at: datetime
    articles_collected: int
    articles_selected: int
    valid_url_ratio: float
    source_diversity: int
    improvement_note: str
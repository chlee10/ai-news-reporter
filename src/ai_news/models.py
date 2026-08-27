import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

__all__ = ["Article", "EditorialPolicy", "RunMetrics", "SourceHealth", "replace"]


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
    detail_points: tuple[str, ...] = ()
    signature: str = ""
    fingerprint: str = ""
    related_domains: int = 1
    times_sent: int = 0  # 0 means never delivered; higher means it has repeated before


@dataclass(frozen=True)
class SourceHealth:
    source: str
    ok: int = 0
    failed: int = 0
    last_error: str = ""
    last_seen_at: str = ""

    @property
    def failure_ratio(self) -> float:
        total = self.ok + self.failed
        return self.failed / total if total else 0.0


@dataclass(frozen=True)
class EditorialPolicy:
    """The tunable part of the harness. Each run measures itself and writes the next one."""

    trust_weight: float = 0.40
    freshness_weight: float = 0.25
    relevance_weight: float = 0.20
    corroboration_weight: float = 0.15
    # Regional balance is structural, not a nudge: each region gets its own slots.
    region_quota: dict[str, int] = field(default_factory=lambda: {"korea": 10, "global": 10})
    max_age_hours: float = 168.0  # a fortnight-old story must not fill the edition
    repeat_penalty: float = 0.55  # score multiplier applied per previous delivery
    max_per_source: int = 3
    source_reliability: dict[str, float] = field(default_factory=dict)
    note: str = "초기 실행: 국내외 균형과 공식 출처 우선 원칙을 적용합니다."
    revision: int = 0

    @property
    def report_size(self) -> int:
        return sum(self.region_quota.values())

    def quota(self, region: str) -> int:
        return self.region_quota.get(region, 0)

    def reliability(self, source: str) -> float:
        return self.source_reliability.get(source, 1.0)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "EditorialPolicy":
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        # Older policies stored an auto-tuned domestic floor. Regional quotas are configuration
        # rather than something the loop tunes, so a stale tuned value must not be inherited:
        # dropping the unknown key falls through to the configured default.
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclass(frozen=True)
class RunMetrics:
    """Everything the harness needs to judge its own run. No metric is tautological."""

    run_at: datetime
    sources_ok: int = 0
    sources_failed: int = 0
    articles_collected: int = 0
    stories_clustered: int = 0
    stories_new: int = 0
    articles_selected: int = 0
    target_size: int = 0
    repeats_included: int = 0
    stale_dropped: int = 0
    domestic_selected: int = 0
    source_diversity: int = 0
    median_age_hours: float = 0.0
    body_fetch_ok: int = 0
    body_fetch_failed: int = 0
    translation_engine: str = "none"
    translation_failures: int = 0
    summary_engine: str = "none"
    delivered: bool = False
    delivery_error: str = ""

    @property
    def domestic_ratio(self) -> float:
        return self.domestic_selected / self.articles_selected if self.articles_selected else 0.0

    @property
    def body_fetch_ratio(self) -> float:
        total = self.body_fetch_ok + self.body_fetch_failed
        return self.body_fetch_ok / total if total else 0.0

    @property
    def source_failure_ratio(self) -> float:
        total = self.sources_ok + self.sources_failed
        return self.sources_failed / total if total else 0.0

    def to_json(self) -> str:
        data = asdict(self)
        data["run_at"] = self.run_at.isoformat()
        return json.dumps(data, ensure_ascii=False, sort_keys=True)

import sqlite3
from datetime import UTC, datetime, timedelta

from ai_news.models import Article, EditorialPolicy, RunMetrics
from ai_news.pipeline import (
    CollectResult,
    Story,
    adjust_reliability,
    cluster_stories,
    fingerprint_of,
    filter_new,
    improve,
    measure,
    rank,
    select,
    signature_of,
)
from ai_news.store import NewsStore


def article(
    title: str,
    region: str = "global",
    source: str = "Example",
    domain: str = "example.com",
    trust: float = 0.9,
    age_hours: float = 1.0,
) -> Article:
    url = f"https://{domain}/{abs(hash(title)) % 10**8}"
    return Article(
        title=title,
        url=url,
        summary="AI model research update",
        published_at=datetime.now(UTC) - timedelta(hours=age_hours),
        source=source,
        region=region,
        trust=trust,
        signature=" ".join(sorted(signature_of(title))),
        fingerprint=fingerprint_of(title, url),
    )


def story(item: Article, *others: Article) -> Story:
    return Story(item, [item, *others])


# ---------------------------------------------------------------- deduplication


def test_same_event_from_two_outlets_becomes_one_story_with_corroboration():
    stories = cluster_stories(
        [
            article("OpenAI releases GPT-6 reasoning model", source="OpenAI", domain="openai.com", trust=1.0),
            article("OpenAI releases GPT-6 reasoning model today", source="TechCrunch", domain="techcrunch.com"),
            article("EU passes new AI liability directive", source="MIT", domain="technologyreview.com"),
        ]
    )
    assert len(stories) == 2
    merged = next(item for item in stories if "GPT-6" in item.representative.title)
    assert merged.representative.source == "OpenAI"
    assert len(merged.domains) == 2


def test_corroboration_counts_distinct_domains_not_repeat_posts(tmp_path):
    policy = EditorialPolicy()
    single = story(article("Anthropic ships Claude update", domain="a.com"))
    repeat = story(
        article("Nvidia unveils new chip", domain="b.com"),
        article("Nvidia unveils new chip again", domain="b.com"),
    )
    cross = story(
        article("Meta opens Llama weights", domain="c.com"),
        article("Meta opens Llama weights broadly", domain="d.com"),
    )
    scores = {item.title: item.score for item in rank([single, repeat, cross], policy)}
    assert scores["Nvidia unveils new chip"] == scores["Anthropic ships Claude update"]
    assert scores["Meta opens Llama weights"] > scores["Nvidia unveils new chip"]


def test_delivered_story_is_not_repeated_by_another_outlet(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    first = article("Google launches Gemini 3 Ultra", source="Google AI", domain="blog.google")
    store.commit_delivery([first])
    republished = story(article("Google launches Gemini 3 Ultra for developers", source="ZDNet", domain="zdnet.co.kr"))
    assert filter_new([republished], store) == []
    assert filter_new([republished], store, force=True) == [republished]


def test_undelivered_story_survives_for_the_next_run(tmp_path):
    """A failed send must never consume a story: nothing is marked until delivery succeeds."""
    store = NewsStore(tmp_path / "news.db")
    pending = story(article("Samsung expands HBM capacity", region="korea"))
    assert filter_new([pending], store) == [pending]
    assert filter_new([pending], store) == [pending]
    store.commit_delivery([pending.representative])
    assert filter_new([pending], store) == []


def test_legacy_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.execute("CREATE TABLE seen (fingerprint TEXT PRIMARY KEY, seen_at TEXT NOT NULL)")
    legacy.execute("INSERT INTO seen VALUES ('old-fingerprint', '2026-01-01T00:00:00+00:00')")
    legacy.commit()
    legacy.close()

    store = NewsStore(path)
    assert "old-fingerprint" in store.known_fingerprints()
    columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(seen)")}
    assert {"signature", "url", "source"} <= columns


# ---------------------------------------------------------------- ranking under policy


def test_policy_weights_actually_change_the_ranking():
    """The stored policy is not decoration: changing it changes which story wins."""
    fresh_but_weak = story(article("Minor AI model tweak", trust=0.7, age_hours=1))
    trusted_but_old = story(article("Official AI policy statement", trust=1.0, age_hours=140))
    freshness_first = EditorialPolicy(trust_weight=0.15, freshness_weight=0.60, relevance_weight=0.15, corroboration_weight=0.10)
    trust_first = EditorialPolicy(trust_weight=0.70, freshness_weight=0.10, relevance_weight=0.15, corroboration_weight=0.05)

    assert rank([fresh_but_weak, trusted_but_old], freshness_first)[0].title == "Minor AI model tweak"
    assert rank([fresh_but_weak, trusted_but_old], trust_first)[0].title == "Official AI policy statement"


def test_selection_honours_domestic_quota_and_per_source_cap():
    policy = EditorialPolicy(report_size=8, domestic_quota=3, max_per_source=2)
    ranked = rank(
        [story(article(f"Global AI model {index}", source="TechCrunch", domain="techcrunch.com")) for index in range(10)]
        + [story(article(f"국내 AI 모델 {index}", "korea", source="ZDNet Korea", domain="zdnet.co.kr")) for index in range(5)],
        policy,
    )
    selected = select(ranked, policy)
    assert len(selected) == 8
    assert sum(item.region == "korea" for item in selected) >= 3


def test_selection_relaxes_the_cap_rather_than_shipping_a_thin_report():
    policy = EditorialPolicy(report_size=5, domestic_quota=0, max_per_source=2)
    ranked = rank([story(article(f"AI model story {index}")) for index in range(6)], policy)
    assert len(select(ranked, policy)) == 5


def test_selection_of_nothing_is_empty():
    assert select([], EditorialPolicy()) == []


def test_insecure_urls_never_reach_the_report():
    insecure = Article("Leaked AI memo", "http://example.com/x", "", datetime.now(UTC), "Blog", "global", 0.5)
    assert rank([story(insecure)], EditorialPolicy()) == []


# ---------------------------------------------------------------- the improvement loop


def base_metrics(**overrides) -> RunMetrics:
    defaults = {
        "run_at": datetime.now(UTC),
        "sources_ok": 8,
        "articles_collected": 90,
        "stories_clustered": 60,
        "stories_new": 20,
        "articles_selected": 12,
        "domestic_selected": 4,
        "source_diversity": 6,
        "median_age_hours": 20.0,
        "body_fetch_ok": 10,
        "body_fetch_failed": 2,
        "translation_engine": "gemini",
        "summary_engine": "gemini",
        "delivered": True,
    }
    return RunMetrics(**{**defaults, **overrides})


def test_low_domestic_share_raises_the_quota_for_the_next_run():
    policy = EditorialPolicy(domestic_quota=4)
    improved = improve(base_metrics(domestic_selected=1), policy)
    assert improved.domestic_quota == 5
    assert improved.revision == 1
    assert "국내" in improved.note


def test_concentrated_sources_tighten_the_per_source_cap():
    improved = improve(base_metrics(source_diversity=3), EditorialPolicy(max_per_source=3))
    assert improved.max_per_source == 2


def test_stale_reports_shift_weight_towards_freshness():
    policy = EditorialPolicy()
    improved = improve(base_metrics(median_age_hours=96.0), policy)
    assert improved.freshness_weight > policy.freshness_weight
    assert improved.trust_weight < policy.trust_weight
    assert round(
        improved.trust_weight + improved.freshness_weight + improved.relevance_weight + improved.corroboration_weight, 2
    ) == 1.0


def test_degraded_engines_are_recorded_in_the_next_policy_note():
    improved = improve(base_metrics(translation_engine="google", summary_engine="fallback", sources_failed=2), EditorialPolicy())
    assert "Gemini" in improved.note
    assert "RSS 발췌" in improved.note
    assert "피드 2곳" in improved.note


def test_failed_delivery_is_visible_in_the_next_policy():
    improved = improve(base_metrics(delivered=False, delivery_error="smtp down"), EditorialPolicy())
    assert "발송에 실패" in improved.note


def test_unreadable_sources_lose_weight_and_recover():
    policy = adjust_reliability(EditorialPolicy(), {"Blocked": False, "Open": True})
    assert policy.reliability("Blocked") < 1.0
    assert policy.reliability("Open") == 1.0
    for _ in range(10):
        policy = adjust_reliability(policy, {"Blocked": False})
    assert policy.reliability("Blocked") >= 0.6


def test_policy_survives_a_round_trip_through_the_store(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    assert store.load_policy().revision == 0
    metrics = base_metrics(domestic_selected=1)
    store.save_run(metrics, improve(metrics, store.load_policy()))
    reloaded = store.load_policy()
    assert reloaded.revision == 1
    assert reloaded.domestic_quota == 5
    assert store.recent_runs(1)[0]["metrics"]["articles_selected"] == 12


# ---------------------------------------------------------------- measurement


def test_metrics_describe_the_whole_run_not_a_filter_that_already_ran():
    collected = CollectResult(
        articles=[article("a"), article("b")],
        ok_sources=["OpenAI", "ZDNet Korea"],
        failures={"Dead Feed": "timeout"},
    )
    selected = [article("a", "korea", source="ZDNet Korea", age_hours=4), article("b", source="OpenAI", age_hours=8)]
    metrics = measure(datetime.now(UTC), collected, [story(item) for item in selected], [], selected)
    assert metrics.sources_ok == 2 and metrics.sources_failed == 1
    assert metrics.domestic_ratio == 0.5
    assert metrics.source_diversity == 2
    assert 3 < metrics.median_age_hours < 9


def test_source_health_is_tracked_across_runs(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    store.record_source("Dead Feed", ok=False, error="timeout")
    store.record_source("Dead Feed", ok=False, error="timeout")
    store.record_source("OpenAI", ok=True)
    health = {item.source: item for item in store.source_health()}
    assert health["Dead Feed"].failed == 2
    assert health["Dead Feed"].failure_ratio == 1.0
    assert health["OpenAI"].ok == 1


# ---------------------------------------------------------------- AI relevance gate


def off_topic(title: str, region: str = "korea") -> Article:
    url = "https://etnews.example/1"
    return Article(
        title=title,
        url=url,
        summary="화장실 요금 논란이 커지고 있다",
        published_at=datetime.now(UTC),
        source="전자신문 IT",
        region=region,
        trust=0.7,
        signature=" ".join(sorted(signature_of(title))),
        fingerprint=fingerprint_of(title, url),
    )


def test_general_tech_feed_cannot_smuggle_an_off_topic_item_into_the_report():
    policy = EditorialPolicy(report_size=5, domestic_quota=3)
    ranked = rank([story(off_topic("화장실 요금 성차별 논란")), story(article("국내 AI 모델 공개", "korea"))], policy)
    assert [item.title for item in ranked] == ["국내 AI 모델 공개"]
    assert select(ranked, policy) == ranked


def test_ai_signal_is_matched_in_korean_and_english_without_false_positives():
    from ai_news.pipeline import is_ai_related

    assert is_ai_related(off_topic("AI가 바꾸는 일자리"))
    assert is_ai_related(off_topic("인공지능 반도체 투자 확대"))
    assert is_ai_related(off_topic("OpenAI launches a chatbot"))
    assert not is_ai_related(off_topic("비가 내리는 주말 교통 상황"))
    assert not is_ai_related(off_topic("Spain said the rain will remain"))

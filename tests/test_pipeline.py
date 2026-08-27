import sqlite3
from datetime import UTC, datetime, timedelta

from ai_news.models import Article, EditorialPolicy, RunMetrics
from ai_news.pipeline import (
    CollectResult,
    Story,
    adjust_reliability,
    cluster_stories,
    fingerprint_of,
    annotate_history,
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


def story(item: Article, *others: Article, times_sent: int = 0) -> Story:
    return Story(item, [item, *others], times_sent)


def fresh_only(stories):
    return [item for item in stories if not item.is_repeat]


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


def test_a_delivered_story_returning_from_another_outlet_counts_as_a_repeat(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    first = article("Google launches Gemini 3 Ultra", source="Google AI", domain="blog.google")
    store.commit_delivery([first])
    republished = story(article("Google launches Gemini 3 Ultra for developers", source="ZDNet", domain="zdnet.co.kr"))
    assert annotate_history([republished], store)[0].is_repeat
    assert annotate_history([republished], store, force=True)[0].times_sent == 0


def test_undelivered_story_stays_unsent_until_delivery_succeeds(tmp_path):
    """A failed send must never consume a story: nothing is counted until delivery succeeds."""
    store = NewsStore(tmp_path / "news.db")
    pending = story(article("Samsung expands HBM capacity", region="korea"))
    assert annotate_history([pending], store)[0].times_sent == 0
    assert annotate_history([pending], store)[0].times_sent == 0
    store.commit_delivery([pending.representative])
    assert annotate_history([pending], store)[0].times_sent == 1


def test_every_delivery_of_the_same_story_deepens_its_penalty(tmp_path):
    store = NewsStore(tmp_path / "news.db")
    item = article("Naver ships an AI model", region="korea")
    for expected in (1, 2, 3):
        store.commit_delivery([item])
        assert annotate_history([story(item)], store)[0].times_sent == expected

    policy = EditorialPolicy(repeat_penalty=0.5)
    scores = [rank([story(item, times_sent=n)], policy)[0].score for n in (0, 1, 2)]
    assert scores[0] > scores[1] > scores[2]
    assert round(scores[1] / scores[0], 2) == 0.5


def test_a_stale_tuned_quota_is_not_inherited_from_an_older_policy():
    """Regional quotas are configuration; an old auto-tuned floor must not silently override it."""
    old = '{"domestic_quota": 2, "report_size": 12, "revision": 6, "max_per_source": 2}'
    policy = EditorialPolicy.from_json(old)
    assert policy.region_quota == {"korea": 10, "global": 10}
    assert policy.revision == 6 and policy.max_per_source == 2


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


def test_each_region_gets_exactly_its_own_quota():
    policy = EditorialPolicy(region_quota={"korea": 3, "global": 5}, max_per_source=2)
    ranked = rank(
        [story(article(f"Global AI model {i}", source=f"Outlet {i}", domain=f"o{i}.com")) for i in range(10)]
        + [story(article(f"국내 AI 모델 {i}", "korea", source=f"매체 {i}", domain=f"k{i}.co.kr")) for i in range(5)],
        policy,
    )
    selected = select(ranked, policy)
    assert len(selected) == 8
    assert sum(item.region == "korea" for item in selected) == 3
    assert sum(item.region == "global" for item in selected) == 5


def test_a_regions_quota_outranks_the_per_source_cap_when_outlets_are_few():
    policy = EditorialPolicy(region_quota={"korea": 4, "global": 0}, max_per_source=2)
    ranked = rank(
        [story(article(f"국내 AI 모델 {i}", "korea", source="AI타임스", domain="aitimes.com")) for i in range(6)],
        policy,
    )
    assert len(select(ranked, policy)) == 4


def test_report_size_is_the_sum_of_the_region_quotas():
    assert EditorialPolicy().report_size == 20
    assert EditorialPolicy(region_quota={"korea": 3, "global": 4}).report_size == 7


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


def test_region_quotas_are_structural_and_survive_policy_revisions():
    """Balance is enforced by the quota, so no feedback nudge should move it."""
    policy = EditorialPolicy(region_quota={"korea": 10, "global": 10})
    for _ in range(5):
        policy = improve(base_metrics(domestic_selected=1), policy)
    assert policy.region_quota == {"korea": 10, "global": 10}
    assert policy.revision == 5


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
    assert reloaded.region_quota == {"korea": 10, "global": 10}
    assert store.recent_runs(1)[0]["metrics"]["articles_selected"] == 12


# ---------------------------------------------------------------- measurement


def test_metrics_describe_the_whole_run_not_a_filter_that_already_ran():
    collected = CollectResult(
        articles=[article("a"), article("b")],
        ok_sources=["OpenAI", "ZDNet Korea"],
        failures={"Dead Feed": "timeout"},
    )
    selected = [article("a", "korea", source="ZDNet Korea", age_hours=4), article("b", source="OpenAI", age_hours=8)]
    metrics = measure(datetime.now(UTC), collected, [story(item) for item in selected], selected)
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
    policy = EditorialPolicy(region_quota={"korea": 3, "global": 0})
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


# ---------------------------------------------------------------- freshness floor


def test_stories_past_the_freshness_floor_never_reach_the_report():
    """A fortnight-old story must not fill the edition just because the pool ran dry."""
    policy = EditorialPolicy(max_age_hours=168.0)
    recent = story(article("AI 모델 신규 공개", age_hours=10))
    stale = story(article("AI 모델 예전 소식", age_hours=360))
    ranked = rank([recent, stale], policy)
    assert [item.title for item in ranked] == ["AI 모델 신규 공개"]


def test_stale_stories_never_pad_the_edition_even_when_it_is_short():
    """Repeats may fill a gap; stories past the freshness floor may not."""
    policy = EditorialPolicy(region_quota={"korea": 0, "global": 12}, max_age_hours=168.0)
    stories = [story(article(f"AI 모델 최신 {i}", age_hours=5)) for i in range(3)]
    stories += [story(article(f"AI 모델 구식 {i}", age_hours=400)) for i in range(20)]
    assert len(select(rank(stories, policy), policy)) == 3


def test_a_looser_floor_lets_older_stories_back_in():
    old = story(article("AI 모델 소식", age_hours=300))
    assert rank([old], EditorialPolicy(max_age_hours=168.0)) == []
    assert len(rank([old], EditorialPolicy(max_age_hours=720.0))) == 1


def test_metrics_count_what_the_floor_removed_and_the_target_it_missed():
    policy = EditorialPolicy(region_quota={"korea": 0, "global": 12}, max_age_hours=168.0)
    stories = [story(article("AI 모델 최신", age_hours=5))]
    stories += [story(article(f"AI 모델 구식 {i}", age_hours=400)) for i in range(4)]
    selected = select(rank(stories, policy), policy)
    metrics = measure(datetime.now(UTC), CollectResult(), stories, selected, policy)
    assert metrics.stale_dropped == 4
    assert metrics.target_size == 12
    assert metrics.articles_selected == 1


def test_a_short_edition_is_explained_in_the_next_policy_note():
    improved = improve(base_metrics(articles_selected=5, target_size=12, stale_dropped=30), EditorialPolicy())
    assert "신선한 신규 기사가 부족해 5/12건만 편성했습니다" in improved.note
    assert "168시간 초과 30건 제외" in improved.note


def test_a_full_edition_says_nothing_about_shortfall():
    improved = improve(base_metrics(articles_selected=12, target_size=12), EditorialPolicy())
    assert "부족해" not in improved.note


def test_the_freshness_floor_survives_policy_revisions():
    policy = EditorialPolicy(max_age_hours=96.0)
    for _ in range(5):
        policy = improve(base_metrics(domestic_selected=1), policy)
    assert policy.max_age_hours == 96.0


# ---------------------------------------------------------------- always-full editions


def test_repeats_fill_the_quota_only_after_fresh_stories_run_out():
    """The edition is always full, but nothing repeats while something new is available."""
    policy = EditorialPolicy(region_quota={"korea": 0, "global": 5}, max_per_source=5)
    stories = [story(article(f"AI 신규 {i}", age_hours=3)) for i in range(2)]
    stories += [story(article(f"AI 기존 {i}", age_hours=3), times_sent=1) for i in range(8)]
    selected = select(rank(stories, policy), policy)

    assert len(selected) == 5
    assert sum(1 for item in selected if not item.times_sent) == 2
    assert sum(1 for item in selected if item.times_sent) == 3


def test_a_fresh_story_always_outranks_a_repeat_of_similar_quality():
    policy = EditorialPolicy(region_quota={"korea": 0, "global": 1}, max_per_source=5)
    weaker_fresh = story(article("AI 신규 소식", trust=0.7, age_hours=40))
    stronger_repeat = story(article("AI 기존 소식", trust=1.0, age_hours=2), times_sent=1)
    selected = select(rank([stronger_repeat, weaker_fresh], policy), policy)
    assert [item.title for item in selected] == ["AI 신규 소식"]


def test_the_least_repeated_story_is_reused_first():
    policy = EditorialPolicy(region_quota={"korea": 0, "global": 2}, max_per_source=5)
    stories = [story(article(f"AI 소식 {n}", trust=1.0, age_hours=3), times_sent=n) for n in (3, 1, 2)]
    selected = select(rank(stories, policy), policy)
    assert sorted(item.times_sent for item in selected) == [1, 2]


def test_metrics_and_policy_note_report_how_many_repeats_were_used():
    policy = EditorialPolicy(region_quota={"korea": 0, "global": 4}, max_per_source=5)
    stories = [story(article("AI 신규", age_hours=3))]
    stories += [story(article(f"AI 기존 {i}", age_hours=3), times_sent=1) for i in range(5)]
    selected = select(rank(stories, policy), policy)
    metrics = measure(datetime.now(UTC), CollectResult(), stories, selected, policy)

    assert metrics.articles_selected == 4
    assert metrics.repeats_included == 3
    assert metrics.stories_new == 1
    assert "이전 발송분 3건을 재편성했습니다" in improve(metrics, policy).note


def test_a_fully_fresh_edition_mentions_no_repeats():
    metrics = base_metrics(repeats_included=0, articles_selected=20, target_size=20)
    assert "재편성" not in improve(metrics, EditorialPolicy()).note

import argparse
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .observability import configure_logging, get_logger
from .pipeline import (
    adjust_reliability,
    cluster_stories,
    collect,
    annotate_history,
    improve,
    measure,
    rank,
    select,
)
from .reporting import (
    enrich_with_article_bodies,
    render_report,
    send_gmail,
    summarize_article_bodies,
    translate_to_korean,
    write_web_report,
)
from .store import NewsStore

LOGGER = get_logger("cli")
DEFAULT_DB = Path("data/ai_news.db")


def run(dry_run: bool = False, force: bool = False, verbose: bool = False) -> str:
    """One harness cycle: collect, dedup, rank under policy, deliver, measure, write the next policy."""
    configure_logging(verbose)
    load_dotenv()
    run_at = datetime.now(UTC)
    store = NewsStore(Path(os.getenv("AI_NEWS_DB", str(DEFAULT_DB))))
    try:
        policy = store.load_policy()

        collected = collect()
        for name in collected.ok_sources:
            store.record_source(name, ok=True)
        for name, error in collected.failures.items():
            store.record_source(name, ok=False, error=error)

        stories = annotate_history(cluster_stories(collected.articles), store, force)
        selected = select(rank(stories, policy), policy)

        selected, translation_engine = translate_to_korean(selected)
        selected, body_outcomes = enrich_with_article_bodies(selected)
        selected, summary_engine = summarize_article_bodies(selected)

        metrics = replace(
            measure(run_at, collected, stories, selected, policy),
            translation_engine=translation_engine,
            summary_engine=summary_engine,
            body_fetch_ok=sum(1 for article in selected if article.body),
            body_fetch_failed=sum(1 for article in selected if not article.body),
        )

        web_report = Path("reports/index.html")
        write_web_report(selected, policy.note, web_report, metrics)
        report_text, report_html = render_report(
            selected, policy.note, os.getenv("REPORT_PUBLIC_URL", ""), metrics
        )

        if dry_run:
            print(report_text)
            preview = web_report.with_name("email-preview.html")
            preview.write_text(report_html, encoding="utf-8")
            LOGGER.info("dry run: nothing sent, dedup memory and policy left untouched")
            LOGGER.info("email preview written to %s", preview)
            return report_text

        failure: Exception | None = None
        if not selected:
            LOGGER.warning("no new verified stories this cycle; skipping delivery")
        else:
            try:
                subject = f"[AI Daily Brief] {datetime.now().strftime('%Y-%m-%d')}"
                send_gmail(subject, report_text, report_html, attachment=web_report)
                metrics = replace(metrics, delivered=True)
            except Exception as error:  # smtplib raises several unrelated exception types
                failure = error
                metrics = replace(metrics, delivered=False, delivery_error=str(error))
                LOGGER.exception("delivery failed; stories stay unmarked and will be retried next cycle")

        if metrics.delivered:
            store.commit_delivery(selected)
            print(f"Sent {len(selected)} verified stories to Gmail.")

        next_policy = improve(metrics, adjust_reliability(policy, body_outcomes))
        store.save_run(metrics, next_policy)
        pruned = store.prune()
        if pruned:
            LOGGER.info("pruned %s expired dedup entries", pruned)

        if failure is not None:
            raise failure
        return report_text
    finally:
        store.close()


def status(limit: int = 5, verbose: bool = False) -> None:
    """Show what the harness has measured and how its policy has moved."""
    configure_logging(verbose)
    store = NewsStore(Path(os.getenv("AI_NEWS_DB", str(DEFAULT_DB))))
    try:
        policy = store.load_policy()
        print("현재 편집 정책")
        print(f"  revision        : {policy.revision}")
        print(f"  가중치          : 신뢰도 {policy.trust_weight} / 최신성 {policy.freshness_weight} / "
              f"관련성 {policy.relevance_weight} / 교차보도 {policy.corroboration_weight}")
        quota = " / ".join(f"{region} {count}건" for region, count in sorted(policy.region_quota.items()))
        print(f"  지역 정원       : {quota} (총 {policy.report_size}건)")
        print(f"  매체당 상한     : {policy.max_per_source}건 · 신선도 하한 {policy.max_age_hours:.0f}시간 "
              f"· 재발송 감점 {policy.repeat_penalty}")
        print(f"  개선 메모       : {policy.note}")
        downgraded = {name: value for name, value in policy.source_reliability.items() if value < 1.0}
        if downgraded:
            print(f"  감점된 매체     : {json.dumps(downgraded, ensure_ascii=False)}")

        runs = store.recent_runs(limit)
        print(f"\n최근 실행 {len(runs)}건")
        for entry in runs:
            metrics = entry["metrics"]
            flag = "OK " if metrics.get("delivered") else "MISS"
            print(
                f"  [{flag}] {entry['run_at'][:19]} 수집 {metrics.get('articles_collected')} "
                f"→ 사건 {metrics.get('stories_clustered')} → 신규 {metrics.get('stories_new')} "
                f"→ 선정 {metrics.get('articles_selected')} "
                f"(국내 {metrics.get('domestic_selected')}, 출처 {metrics.get('source_diversity')}곳, "
                f"번역 {metrics.get('translation_engine')}, 요약 {metrics.get('summary_engine')})"
            )
            if metrics.get("delivery_error"):
                print(f"         발송 오류: {metrics['delivery_error']}")

        health = store.source_health()
        if health:
            print("\n소스 상태")
            for item in health:
                print(f"  {item.source:<26} 성공 {item.ok:>4} 실패 {item.failed:>4} "
                      f"실패율 {item.failure_ratio:.0%} {item.last_error[:60]}")
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI news daily reporting harness")
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--dry-run", action="store_true", help="Render the report without sending or recording it")
    parser.add_argument("--force", action="store_true", help="Ignore dedup memory for a one-off resend")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--limit", type=int, default=5, help="How many past runs `status` should show")
    args = parser.parse_args()

    if args.command == "run":
        run(args.dry_run, args.force, args.verbose)
        return
    status(args.limit, args.verbose)


if __name__ == "__main__":
    main()

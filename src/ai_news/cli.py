import argparse
import os
import time
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

from .pipeline import NewsStore, collect, evaluate, verify_and_rank
from .reporting import enrich_with_article_bodies, render_report, send_gmail, summarize_article_bodies, translate_to_korean, write_web_report


def run(dry_run: bool = False, force: bool = False) -> str:
    load_dotenv()
    store = NewsStore(Path("data/ai_news.db"))
    collected = collect()
    selected = verify_and_rank(collected, store, force)
    selected = translate_to_korean(selected)
    selected = enrich_with_article_bodies(selected)
    selected = summarize_article_bodies(selected)
    guidance = store.latest_guidance()
    write_web_report(selected, guidance)
    report_text, report_html = render_report(selected, guidance, os.getenv("REPORT_PUBLIC_URL", ""))
    evaluation = evaluate(collected, selected)
    store.save_evaluation(evaluation)
    subject = f"[AI Daily Brief] {datetime.now().strftime('%Y-%m-%d')}"
    if dry_run:
        print(report_text)
    else:
        send_gmail(subject, report_text, report_html)
        print(f"Sent {len(selected)} verified articles to Gmail.")
    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(description="AI news daily reporting harness")
    parser.add_argument("command", choices=("run", "schedule"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Include previously reported articles for a one-off resend")
    args = parser.parse_args()
    if args.command == "run":
        run(args.dry_run, args.force)
        return
    scheduler = BlockingScheduler(timezone="Asia/Seoul")
    scheduler.add_job(run, "cron", hour="8,12,17", minute=0, id="ai-news-report", replace_existing=True)
    print("Scheduler started: daily at 08:00, 12:00, 17:00 Asia/Seoul")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    main()
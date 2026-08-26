import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .models import Article, EditorialPolicy, RunMetrics, SourceHealth
from .observability import get_logger

LOGGER = get_logger("store")
DEDUP_WINDOW_DAYS = 7


class NewsStore:
    """Durable memory for the harness: what was delivered, how each run scored, what to do next."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        cursor = self.connection
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS seen ("
            "fingerprint TEXT PRIMARY KEY, seen_at TEXT NOT NULL, signature TEXT NOT NULL DEFAULT '', "
            "url TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '')"
        )
        existing = {row["name"] for row in cursor.execute("PRAGMA table_info(seen)")}
        for column in ("signature", "url", "source"):
            if column not in existing:
                cursor.execute(f"ALTER TABLE seen ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
                LOGGER.info("migrated seen table: added column %s", column)
        cursor.execute("CREATE INDEX IF NOT EXISTS seen_at_idx ON seen(seen_at)")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "run_at TEXT PRIMARY KEY, metrics TEXT NOT NULL, policy TEXT NOT NULL, note TEXT NOT NULL DEFAULT '')"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS source_health ("
            "source TEXT PRIMARY KEY, ok INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0, "
            "last_error TEXT NOT NULL DEFAULT '', last_seen_at TEXT NOT NULL DEFAULT '')"
        )
        self.connection.commit()

    # ---------- delivery memory ----------

    def recent_signatures(self, days: int = DEDUP_WINDOW_DAYS) -> list[tuple[str, frozenset[str]]]:
        """Signatures delivered recently, used to catch the same story from a different outlet."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        rows = self.connection.execute(
            "SELECT fingerprint, signature FROM seen WHERE seen_at >= ? AND signature != ''", (cutoff,)
        ).fetchall()
        return [(row["fingerprint"], frozenset(row["signature"].split())) for row in rows]

    def known_fingerprints(self) -> set[str]:
        return {row["fingerprint"] for row in self.connection.execute("SELECT fingerprint FROM seen")}

    def commit_delivery(self, articles: list[Article]) -> None:
        """Called only after the report has actually been delivered, so a failed send never loses a story."""
        now = datetime.now(UTC).isoformat()
        self.connection.executemany(
            "INSERT OR IGNORE INTO seen (fingerprint, seen_at, signature, url, source) VALUES (?, ?, ?, ?, ?)",
            [(article.fingerprint, now, article.signature, article.url, article.source) for article in articles],
        )
        self.connection.commit()
        LOGGER.info("committed %s delivered stories to dedup memory", len(articles))

    def prune(self, days: int = 90) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        cursor = self.connection.execute("DELETE FROM seen WHERE seen_at < ?", (cutoff,))
        self.connection.commit()
        return cursor.rowcount

    # ---------- run history and policy ----------

    def save_run(self, metrics: RunMetrics, next_policy: EditorialPolicy) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO runs (run_at, metrics, policy, note) VALUES (?, ?, ?, ?)",
            (metrics.run_at.isoformat(), metrics.to_json(), next_policy.to_json(), next_policy.note),
        )
        self.connection.commit()

    def load_policy(self) -> EditorialPolicy:
        row = self.connection.execute("SELECT policy FROM runs ORDER BY run_at DESC LIMIT 1").fetchone()
        if row is None:
            LOGGER.info("no prior run found; starting from the default editorial policy")
            return EditorialPolicy()
        policy = EditorialPolicy.from_json(row["policy"])
        LOGGER.info("loaded editorial policy revision %s: %s", policy.revision, policy.note)
        return policy

    def recent_runs(self, limit: int = 10) -> list[dict]:
        rows = self.connection.execute(
            "SELECT run_at, metrics, note FROM runs ORDER BY run_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"run_at": row["run_at"], "metrics": json.loads(row["metrics"]), "note": row["note"]} for row in rows]

    # ---------- source health ----------

    def record_source(self, source: str, ok: bool, error: str = "") -> None:
        self.connection.execute(
            "INSERT INTO source_health (source, ok, failed, last_error, last_seen_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(source) DO UPDATE SET ok = ok + ?, failed = failed + ?, "
            "last_error = CASE WHEN ? = '' THEN last_error ELSE ? END, last_seen_at = ?",
            (
                source, int(ok), int(not ok), error, datetime.now(UTC).isoformat(),
                int(ok), int(not ok), error, error, datetime.now(UTC).isoformat(),
            ),
        )
        self.connection.commit()

    def source_health(self) -> list[SourceHealth]:
        rows = self.connection.execute("SELECT * FROM source_health ORDER BY failed DESC, source").fetchall()
        return [
            SourceHealth(row["source"], row["ok"], row["failed"], row["last_error"], row["last_seen_at"])
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()

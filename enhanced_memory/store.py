"""Enhanced Memory Store — two-tier SQLite backend with FTS5 full-text search.

Provides a durable, thread-safe storage layer for the Hermes Agent enhanced-memory
plugin.  Raw conversational facts are stored in ``raw_facts`` and periodically
condensed into higher-level summaries in ``condensed``.  Both tables are backed by
FTS5 virtual tables with automatic trigger-based synchronisation.

Architecture overview::

    raw_facts  ──(condenser)──►  condensed
        │                            │
        ▼                            ▼
    raw_facts_fts              condensed_fts
    (FTS5, auto-sync            (FTS5, auto-sync
     via triggers)               via triggers)

Thread safety:
    * Each thread gets its own ``sqlite3.Connection`` via ``threading.local``.
    * Writes are serialised through a ``threading.Lock``.
    * SQLite WAL mode allows concurrent readers alongside a single writer.

Usage::

    store = EnhancedMemoryStore()                       # default path
    store = EnhancedMemoryStore("/tmp/my_memory.db")    # custom path

    fact_id = store.add_raw_fact("User prefers dark mode", category="user_pref")
    results = store.search_raw("dark mode")
    stats   = store.stats()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid category labels for classifying stored facts.
# These are used for filtering, priority assignment, and topic grouping.
VALID_CATEGORIES: set[str] = {
    "user_pref",
    "project",
    "tool",
    "env",
    "decision",
    "security",
    "general",
}

# Default SQLite database filename placed inside the Hermes home directory.
_DEFAULT_DB_NAME = "memory_store.db"


def _default_db_path() -> str:
    """Return the default database path, preferring ``hermes_constants`` if available.

    Resolution order:
        1. ``hermes_constants.get_hermes_home()`` (respects ``$HERMES_HOME``,
           active profile, etc.)
        2. Falls back to ``~/.hermes/`` if the import fails.

    The parent directory is created automatically if it does not exist.

    Returns:
        str: Absolute path to the default SQLite database file.
    """
    try:
        from hermes_constants import get_hermes_home  # type: ignore[import-untyped]

        base = get_hermes_home()
    except Exception:
        base = os.path.join(Path.home(), ".hermes")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, _DEFAULT_DB_NAME)


# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

# ── Core table: raw_facts ─────────────────────────────────────────────────
# Stores individual facts as they arrive, before condensation.
# The ``condensed`` flag (0/1) tracks whether a fact has already been
# processed by the condenser pipeline.
_SCHEMA_RAW_FACTS = """\
CREATE TABLE IF NOT EXISTS raw_facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT    NOT NULL,
    category   TEXT    NOT NULL DEFAULT 'general',
    source     TEXT    NOT NULL DEFAULT '',
    session_id TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL,
    condensed  INTEGER NOT NULL DEFAULT 0
);
"""

# ── Core table: condensed ─────────────────────────────────────────────────
# Stores condensed summaries grouped by (topic, category).
# ``priority`` is clamped to 1-10 via a CHECK constraint.
# ``source_ids`` is a JSON array of raw_facts.id values that contributed.
# ``version`` increments on every update so callers can detect staleness.
_SCHEMA_CONDENSED = """\
CREATE TABLE IF NOT EXISTS condensed (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT    NOT NULL,
    summary    TEXT    NOT NULL,
    category   TEXT    NOT NULL DEFAULT 'general',
    priority   INTEGER NOT NULL DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    source_ids TEXT    NOT NULL DEFAULT '[]',
    fact_count INTEGER NOT NULL DEFAULT 0,
    version    INTEGER NOT NULL DEFAULT 1,
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);
"""

# ── FTS5 virtual table for raw_facts ──────────────────────────────────────
# External-content FTS5 table: the content= and content_rowid= parameters
# tell FTS5 that the real data lives in ``raw_facts``.  This avoids
# duplicating text and lets FTS5 handle tokenisation/ranking internally.
_SCHEMA_FTS_RAW = """\
CREATE VIRTUAL TABLE IF NOT EXISTS raw_facts_fts
USING fts5(content, category, source, content=raw_facts, content_rowid=id);
"""

# ── FTS5 virtual table for condensed ──────────────────────────────────────
# Same external-content pattern; indexes topic, summary, and category.
_SCHEMA_FTS_CONDENSED = """\
CREATE VIRTUAL TABLE IF NOT EXISTS condensed_fts
USING fts5(topic, summary, category, content=condensed, content_rowid=id);
"""

# ── FTS5 synchronisation triggers ─────────────────────────────────────────
# Because we use external-content FTS5 tables, SQLite does NOT automatically
# keep the FTS index in sync with the content tables.  These triggers handle
# INSERT / DELETE / UPDATE events following the pattern from the official
# SQLite FTS5 documentation:
#
#   • AFTER INSERT  → insert the new row into the FTS index.
#   • AFTER DELETE  → issue a special "delete" command to remove the old row.
#   • AFTER UPDATE  → delete the old entry then insert the new one.
#
# The "delete" command uses the magic first-column syntax:
#     INSERT INTO fts_table(fts_table, rowid, ...) VALUES('delete', old.id, ...);
# which tells FTS5 to remove the entry without touching the content table.

_TRIGGERS_RAW = [
    """\
CREATE TRIGGER IF NOT EXISTS raw_facts_ai AFTER INSERT ON raw_facts BEGIN
    INSERT INTO raw_facts_fts(rowid, content, category, source)
    VALUES (new.id, new.content, new.category, new.source);
END;
""",
    """\
CREATE TRIGGER IF NOT EXISTS raw_facts_ad AFTER DELETE ON raw_facts BEGIN
    INSERT INTO raw_facts_fts(raw_facts_fts, rowid, content, category, source)
    VALUES ('delete', old.id, old.content, old.category, old.source);
END;
""",
    """\
CREATE TRIGGER IF NOT EXISTS raw_facts_au AFTER UPDATE ON raw_facts BEGIN
    INSERT INTO raw_facts_fts(raw_facts_fts, rowid, content, category, source)
    VALUES ('delete', old.id, old.content, old.category, old.source);
    INSERT INTO raw_facts_fts(rowid, content, category, source)
    VALUES (new.id, new.content, new.category, new.source);
END;
""",
]

_TRIGGERS_CONDENSED = [
    """\
CREATE TRIGGER IF NOT EXISTS condensed_ai AFTER INSERT ON condensed BEGIN
    INSERT INTO condensed_fts(rowid, topic, summary, category)
    VALUES (new.id, new.topic, new.summary, new.category);
END;
""",
    """\
CREATE TRIGGER IF NOT EXISTS condensed_ad AFTER DELETE ON condensed BEGIN
    INSERT INTO condensed_fts(condensed_fts, rowid, topic, summary, category)
    VALUES ('delete', old.id, old.topic, old.summary, old.category);
END;
""",
    """\
CREATE TRIGGER IF NOT EXISTS condensed_au AFTER UPDATE ON condensed BEGIN
    INSERT INTO condensed_fts(condensed_fts, rowid, topic, summary, category)
    VALUES ('delete', old.id, old.topic, old.summary, old.category);
    INSERT INTO condensed_fts(rowid, topic, summary, category)
    VALUES (new.id, new.topic, new.summary, new.category);
END;
""",
]

# ── B-tree indexes for fast filtering and lookups ─────────────────────────
# The unique index on (topic, category) enforces the upsert semantics used
# by the condenser: each topic+category pair maps to exactly one condensed row.
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_raw_facts_category ON raw_facts(category);",
    "CREATE INDEX IF NOT EXISTS idx_raw_facts_condensed ON raw_facts(condensed);",
    "CREATE INDEX IF NOT EXISTS idx_raw_facts_session ON raw_facts(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_raw_facts_created ON raw_facts(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_condensed_category ON condensed(category);",
    "CREATE INDEX IF NOT EXISTS idx_condensed_topic ON condensed(topic);",
    "CREATE INDEX IF NOT EXISTS idx_condensed_priority ON condensed(priority DESC);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_condensed_topic_cat ON condensed(topic, category);",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        str: Timestamp with seconds precision, e.g. ``'2025-01-15T12:30:45+00:00'``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_category(category: str) -> str:
    """Validate and normalise a category string.

    Args:
        category: The candidate category label.

    Returns:
        str: The original *category* if it is in :data:`VALID_CATEGORIES`,
            otherwise ``'general'`` with a warning logged.
    """
    if category in VALID_CATEGORIES:
        return category
    logger.warning("Invalid category %r, falling back to 'general'", category)
    return "general"


def _row_to_dict(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict[str, Any]:
    """Convert a ``sqlite3.Row`` to a plain :class:`dict`.

    Uses the cursor's ``description`` to map column positions to names.

    Args:
        cursor: The cursor whose ``description`` provides column names.
        row: A single result row from a ``fetchone`` / ``fetchall`` call.

    Returns:
        dict[str, Any]: Column-name → value mapping for the row.
    """
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class EnhancedMemoryStore:
    """Two-tier SQLite memory store with FTS5 full-text search.

    Provides CRUD operations on two tables:

    * **raw_facts** — individual factual statements extracted from conversations.
    * **condensed** — higher-level summaries produced by the :class:`FactCondenser`.

    Both tables have companion FTS5 virtual tables (``raw_facts_fts``,
    ``condensed_fts``) that are kept in sync via database triggers so that
    full-text search is always up to date.

    Attributes:
        _db_path (str): Absolute path to the SQLite database file.
        _lock (threading.Lock): Serialises write transactions across threads.
        _local (threading.local): Holds per-thread ``sqlite3.Connection``
            instances so that each thread uses its own connection.

    Args:
        db_path: Path to the SQLite database file.  When ``None`` the default
            path derived from ``hermes_constants.get_hermes_home()`` (or
            ``~/.hermes/memory_store.db``) is used.

    Example::

        store = EnhancedMemoryStore("/tmp/test.db")
        fid = store.add_raw_fact("User prefers vim", category="user_pref")
        results = store.search_raw("vim")
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialise the memory store and create the schema if needed.

        Args:
            db_path: Path to the SQLite database file.  When ``None`` the
                default path derived from ``hermes_constants.get_hermes_home()``
                (or ``~/.hermes/memory_store.db``) is used.
        """
        self._db_path: str = db_path or _default_db_path()
        # Threading lock serialises all write transactions (readers are lock-free
        # thanks to WAL mode).
        self._lock = threading.Lock()
        # Each thread lazily creates its own sqlite3.Connection stored here.
        self._local = threading.local()
        logger.info("EnhancedMemoryStore using db: %s", self._db_path)
        # Ensure schema exists on construction.
        self._init_schema()

    # -- connection helpers -------------------------------------------------

    def get_connection(self) -> sqlite3.Connection:
        """Return the per-thread SQLite connection.

        Other components (e.g. :class:`FactCondenser`, :class:`SemanticSearch`)
        call this method to share the same connection and benefit from WAL-mode
        concurrency.

        Returns:
            sqlite3.Connection: A lazily-created, thread-local connection.
        """
        return self._get_conn()

    @property
    def conn(self) -> sqlite3.Connection:
        """Property alias for :meth:`get_connection`.

        Provides attribute-style access (``store.conn``) used by the
        :mod:`embeddings` module when resolving fact content from vector
        search results.

        Returns:
            sqlite3.Connection: The per-thread connection.
        """
        return self._get_conn()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection, creating it lazily on first access.

        Connection PRAGMAs applied on creation:

        * **journal_mode=WAL** — Write-Ahead Logging enables concurrent
          readers without blocking writers and vice-versa.
        * **foreign_keys=ON** — Enforce FK constraints (good practice).
        * **busy_timeout=5000** — Wait up to 5 s for a database lock before
          raising ``OperationalError``, avoiding immediate failures under
          contention.

        Returns:
            sqlite3.Connection: A configured, thread-local connection.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            # check_same_thread=False is safe because we manage thread-safety
            # ourselves via self._lock and thread-local storage.
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")       # enable WAL for concurrent reads
            conn.execute("PRAGMA foreign_keys=ON;")        # enforce referential integrity
            conn.execute("PRAGMA busy_timeout=5000;")      # 5s wait before SQLITE_BUSY
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @contextmanager
    def _write_tx(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that serialises writes and wraps them in a transaction.

        Acquires ``self._lock`` so that only one thread can execute a write
        transaction at a time.  Uses ``BEGIN IMMEDIATE`` to acquire a
        reserved lock right away (avoids deadlocks when two connections
        try to upgrade from shared to reserved simultaneously).

        On success the transaction is committed; on any exception it is
        rolled back and the exception re-raised.

        Yields:
            sqlite3.Connection: The thread-local connection inside the
                active transaction.

        Raises:
            Exception: Any exception raised by the caller's block is
                propagated after a rollback.
        """
        conn = self._get_conn()
        with self._lock:  # serialise writers across threads
            try:
                conn.execute("BEGIN IMMEDIATE;")  # reserved lock from the start
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @contextmanager
    def _read_tx(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for read-only access (no lock required with WAL).

        WAL mode allows multiple concurrent readers, so we do **not**
        acquire ``self._lock`` here.  The connection is simply yielded
        for the caller to execute read queries.

        Yields:
            sqlite3.Connection: The thread-local connection.
        """
        conn = self._get_conn()
        yield conn

    # -- schema -------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create tables, FTS virtual tables, triggers, and indexes if missing.

        Idempotent — every DDL statement uses ``IF NOT EXISTS`` so calling
        this multiple times is safe.  Runs inside a single write transaction
        so schema creation is atomic.
        """
        with self._write_tx() as conn:
            # Core content tables
            conn.execute(_SCHEMA_RAW_FACTS)
            conn.execute(_SCHEMA_CONDENSED)
            # FTS5 virtual tables (external content)
            conn.execute(_SCHEMA_FTS_RAW)
            conn.execute(_SCHEMA_FTS_CONDENSED)
            # Triggers that keep FTS indexes in sync with content tables
            for trigger_sql in _TRIGGERS_RAW + _TRIGGERS_CONDENSED:
                conn.execute(trigger_sql)
            # B-tree indexes for efficient filtering
            for idx_sql in _INDEXES:
                conn.execute(idx_sql)
        logger.debug("Schema initialisation complete.")

    # -- raw_facts CRUD -----------------------------------------------------

    def add_raw_fact(
        self,
        content: str,
        category: str = "general",
        source: str = "",
        session_id: str = "",
    ) -> int:
        """Insert a single raw fact and return its ``rowid``.

        The fact is inserted inside a serialised write transaction.  The
        companion FTS5 trigger automatically updates ``raw_facts_fts``.

        Args:
            content: The fact text.
            category: One of :data:`VALID_CATEGORIES`.  Invalid values
                fall back to ``'general'``.
            source: Free-form provenance string (e.g. ``"dialog"``,
                ``"auto_extract"``).
            session_id: Identifier for the originating session.

        Returns:
            int: The ``rowid`` of the newly inserted fact.

        Example::

            fid = store.add_raw_fact("User prefers dark mode", category="user_pref")
        """
        category = _validate_category(category)
        now = _utcnow()
        with self._write_tx() as conn:
            cur = conn.execute(
                "INSERT INTO raw_facts (content, category, source, session_id, created_at) "
                "VALUES (?, ?, ?, ?, ?);",
                (content, category, source, session_id, now),
            )
            rowid = cur.lastrowid
        logger.debug("Added raw fact id=%s category=%s", rowid, category)
        return rowid  # type: ignore[return-value]

    def add_raw_facts_batch(self, facts: list[dict[str, Any]]) -> list[int]:
        """Insert multiple raw facts in a single write transaction.

        Each dict in *facts* may contain the keys: ``content`` (required),
        ``category``, ``source``, ``session_id``.

        Facts with empty or missing ``content`` are silently skipped (with a
        warning logged).

        Args:
            facts: List of fact dicts.  Each must contain at least ``content``.

        Returns:
            list[int]: The inserted ``rowid`` values, in insertion order
            (may be shorter than *facts* if any were skipped).

        Example::

            ids = store.add_raw_facts_batch([
                {"content": "fact one", "category": "env"},
                {"content": "fact two"},
            ])
        """
        if not facts:
            return []
        now = _utcnow()
        ids: list[int] = []
        with self._write_tx() as conn:
            for fact in facts:
                content = fact.get("content")
                if not content:
                    logger.warning("Skipping fact with empty content: %s", fact)
                    continue
                category = _validate_category(fact.get("category", "general"))
                source = fact.get("source", "")
                session_id = fact.get("session_id", "")
                cur = conn.execute(
                    "INSERT INTO raw_facts (content, category, source, session_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?);",
                    (content, category, source, session_id, now),
                )
                ids.append(cur.lastrowid)  # type: ignore[arg-type]
        logger.debug("Batch-inserted %d raw facts", len(ids))
        return ids

    def get_raw_by_id(self, fact_id: int) -> dict[str, Any] | None:
        """Retrieve a single raw fact by its primary key.

        Args:
            fact_id: The ``raw_facts.id`` to look up.

        Returns:
            dict[str, Any] | None: A dict with keys ``id``, ``content``,
            ``category``, ``source``, ``session_id``, ``created_at``,
            ``condensed``; or ``None`` if no matching row exists.
        """
        with self._read_tx() as conn:
            row = conn.execute(
                "SELECT id, content, category, source, session_id, created_at, condensed "
                "FROM raw_facts WHERE id = ?",
                (fact_id,),
            ).fetchone()
        if not row:
            return None
        return dict(zip(
            ("id", "content", "category", "source", "session_id", "created_at", "condensed"),
            row,
        ))

    def get_condensed_by_id(self, condensed_id: int) -> dict[str, Any] | None:
        """Retrieve a single condensed entry by its primary key.

        The ``source_ids`` field is automatically deserialised from its
        JSON string representation into a Python list.

        Args:
            condensed_id: The ``condensed.id`` to look up.

        Returns:
            dict[str, Any] | None: A dict with keys ``id``, ``topic``,
            ``summary``, ``category``, ``priority``, ``source_ids``,
            ``fact_count``, ``version``, ``created_at``, ``updated_at``;
            or ``None`` if no matching row exists.
        """
        with self._read_tx() as conn:
            row = conn.execute(
                "SELECT id, topic, summary, category, priority, source_ids, "
                "fact_count, version, created_at, updated_at "
                "FROM condensed WHERE id = ?",
                (condensed_id,),
            ).fetchone()
        if not row:
            return None
        d = dict(zip(
            ("id", "topic", "summary", "category", "priority", "source_ids",
             "fact_count", "version", "created_at", "updated_at"),
            row,
        ))
        try:
            d["source_ids"] = json.loads(d["source_ids"]) if d["source_ids"] else []
        except (json.JSONDecodeError, TypeError):
            d["source_ids"] = []
        return d

    def search_raw(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Full-text search over raw facts.

        Uses the ``raw_facts_fts`` FTS5 virtual table.  Results are ranked
        by the built-in BM25-style ``rank`` column (lower = better match).

        Args:
            query: FTS5 match expression (e.g. ``"python OR rust"``).
                Empty/blank queries return an empty list immediately.
            limit: Maximum number of results to return.

        Returns:
            list[dict[str, Any]]: Matching rows ordered by FTS5 rank
            (best match first).
        """
        if not query or not query.strip():
            return []
        with self._read_tx() as conn:
            cur = conn.execute(
                "SELECT rf.* FROM raw_facts rf "
                "JOIN raw_facts_fts fts ON rf.id = fts.rowid "
                "WHERE raw_facts_fts MATCH ? "
                "ORDER BY fts.rank "
                "LIMIT ?;",
                (query, limit),
            )
            return [_row_to_dict(cur, row) for row in cur.fetchall()]

    def list_uncondensed(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return raw facts that have not yet been condensed.

        Results are ordered oldest-first (``created_at ASC``) so the condenser processes them
        chronologically.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            list[dict[str, Any]]: Uncondensed facts in chronological order.
        """
        with self._read_tx() as conn:
            cur = conn.execute(
                "SELECT * FROM raw_facts WHERE condensed = 0 "
                "ORDER BY created_at ASC LIMIT ?;",
                (limit,),
            )
            return [_row_to_dict(cur, row) for row in cur.fetchall()]

    def mark_condensed(self, fact_ids: list[int]) -> None:
        """Mark the given raw fact IDs as condensed.

        Sets ``condensed = 1`` on each row so it will no longer appear in
        :meth:`list_uncondensed` results.

        Args:
            fact_ids: List of ``raw_facts.id`` values to mark.  An empty
                list is a no-op.
        """
        if not fact_ids:
            return
        # Build a dynamic IN (...) clause with one placeholder per ID.
        placeholders = ",".join("?" for _ in fact_ids)
        with self._write_tx() as conn:
            conn.execute(
                f"UPDATE raw_facts SET condensed = 1 WHERE id IN ({placeholders});",
                fact_ids,
            )
        logger.debug("Marked %d facts as condensed", len(fact_ids))

    # -- condensed CRUD -----------------------------------------------------

    def add_condensed(
        self,
        topic: str,
        summary: str,
        category: str = "general",
        priority: int = 5,
        source_ids: list[int] | None = None,
        fact_count: int = 0,
    ) -> int:
        """Insert a new condensed summary.

        A unique index on ``(topic, category)`` means only one condensed
        row exists per topic+category pair.  Use :meth:`update_condensed`
        to merge new information into an existing row.

        Args:
            topic: Short topic label (should be unique per category).
            summary: The condensed summary text.
            category: One of :data:`VALID_CATEGORIES`.
            priority: Importance ranking 1–10 (10 = highest).  Values
                outside this range are clamped.
            source_ids: Raw-fact IDs that contributed to this summary.
            fact_count: Number of raw facts that were condensed.

        Returns:
            int: The ``rowid`` of the inserted summary.

        Raises:
            sqlite3.IntegrityError: If a row with the same ``(topic, category)``
                already exists (use :meth:`update_condensed` instead).
        """
        category = _validate_category(category)
        priority = max(1, min(10, priority))  # clamp to valid range
        now = _utcnow()
        source_ids_json = json.dumps(source_ids or [])
        with self._write_tx() as conn:
            cur = conn.execute(
                "INSERT INTO condensed "
                "(topic, summary, category, priority, source_ids, fact_count, version, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?);",
                (topic, summary, category, priority, source_ids_json, fact_count, now, now),
            )
            rowid = cur.lastrowid
        logger.debug("Added condensed id=%s topic=%r", rowid, topic)
        return rowid  # type: ignore[return-value]

    def update_condensed(
        self,
        id: int,
        summary: str,
        source_ids: list[int] | None = None,
        fact_count: int | None = None,
        priority: int | None = None,
    ) -> None:
        """Update an existing condensed summary in-place.

        Only the provided non-``None`` fields are changed.  The ``version``
        counter is automatically incremented and ``updated_at`` refreshed.

        Args:
            id: Row id of the condensed record.
            summary: New summary text (always applied).
            source_ids: Updated list of contributing raw-fact IDs.
                ``None`` leaves the existing value unchanged.
            fact_count: Updated count.  ``None`` leaves unchanged.
            priority: Updated priority (clamped 1–10).  ``None`` leaves
                unchanged.
        """
        now = _utcnow()
        # Start with the fields that are always updated.
        sets: list[str] = ["summary = ?", "updated_at = ?", "version = version + 1"]
        params: list[Any] = [summary, now]

        # Conditionally append optional fields.
        if source_ids is not None:
            sets.append("source_ids = ?")
            params.append(json.dumps(source_ids))
        if fact_count is not None:
            sets.append("fact_count = ?")
            params.append(fact_count)
        if priority is not None:
            sets.append("priority = ?")
            params.append(max(1, min(10, priority)))

        params.append(id)
        sql = f"UPDATE condensed SET {', '.join(sets)} WHERE id = ?;"
        with self._write_tx() as conn:
            conn.execute(sql, params)
        logger.debug("Updated condensed id=%s", id)

    def get_condensed(self, topic: str, category: str = "general") -> dict[str, Any] | None:
        """Retrieve a single condensed record by topic and category.

        Leverages the unique index on ``(topic, category)`` so this is an
        O(1) lookup.

        Args:
            topic: The topic label to look up.
            category: Category to match (validated first).

        Returns:
            dict[str, Any] | None: The matching row with ``source_ids``
            deserialised to a list, or ``None`` if not found.
        """
        category = _validate_category(category)
        with self._read_tx() as conn:
            cur = conn.execute(
                "SELECT * FROM condensed WHERE topic = ? AND category = ? LIMIT 1;",
                (topic, category),
            )
            row = cur.fetchone()
            if row is None:
                return None
            result = _row_to_dict(cur, row)
            # Deserialise the JSON array stored in the source_ids column.
            result["source_ids"] = json.loads(result.get("source_ids", "[]"))
            return result

    def search_condensed(
        self,
        query: str | None = None,
        category: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search condensed summaries with optional FTS and/or category filter.

        When *query* is provided, the FTS5 ``condensed_fts`` index is used
        and results include the BM25 rank.  Otherwise all rows are returned,
        filtered only by *category* (if given).

        Args:
            query: FTS5 match expression (optional).  ``None`` or blank
                disables full-text filtering.
            category: Filter by category (optional).  Invalid values are
                normalised via :func:`_validate_category`.
            limit: Maximum number of results to return.

        Returns:
            list[dict[str, Any]]: Matching rows ordered by priority DESC,
            then by FTS rank (if query given) or ``updated_at DESC``.
        """
        conditions: list[str] = []
        params: list[Any] = []

        # Decide whether to use the FTS5 join or a plain table scan.
        use_fts = bool(query and query.strip())

        if use_fts:
            base = (
                "SELECT c.* FROM condensed c "
                "JOIN condensed_fts fts ON c.id = fts.rowid "
                "WHERE condensed_fts MATCH ?"
            )
            params.append(query)
        else:
            base = "SELECT * FROM condensed c WHERE 1=1"

        if category:
            category = _validate_category(category)
            conditions.append("c.category = ?")
            params.append(category)

        where_extra = (" AND " + " AND ".join(conditions)) if conditions else ""
        order = "ORDER BY c.priority DESC" + (", fts.rank" if use_fts else ", c.updated_at DESC")
        sql = f"{base}{where_extra} {order} LIMIT ?;"
        params.append(limit)

        with self._read_tx() as conn:
            cur = conn.execute(sql, params)
            results = []
            for row in cur.fetchall():
                d = _row_to_dict(cur, row)
                d["source_ids"] = json.loads(d.get("source_ids", "[]"))
                results.append(d)
            return results

    def list_condensed(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return condensed summaries sorted by priority (highest first).

        A convenience wrapper around :meth:`search_condensed` without any
        query or category filter.

        Args:
            limit: Maximum number of results to return.

        Returns:
            list[dict[str, Any]]: Condensed rows ordered by
            ``priority DESC, updated_at DESC``.
        """
        with self._read_tx() as conn:
            cur = conn.execute(
                "SELECT * FROM condensed ORDER BY priority DESC, updated_at DESC LIMIT ?;",
                (limit,),
            )
            results = []
            for row in cur.fetchall():
                d = _row_to_dict(cur, row)
                d["source_ids"] = json.loads(d.get("source_ids", "[]"))
                results.append(d)
            return results

    # -- statistics ---------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the memory store.

        Returns:
            dict[str, Any]: Keys: ``raw_total``, ``raw_uncondensed``, ``condensed_total``,
            ``categories`` (dict of category → count for raw facts),
            ``db_path``, ``db_size_bytes``.
        """
        with self._read_tx() as conn:
            raw_total = conn.execute("SELECT COUNT(*) FROM raw_facts;").fetchone()[0]
            raw_uncondensed = conn.execute(
                "SELECT COUNT(*) FROM raw_facts WHERE condensed = 0;"
            ).fetchone()[0]
            condensed_total = conn.execute("SELECT COUNT(*) FROM condensed;").fetchone()[0]

            cat_rows = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM raw_facts GROUP BY category;"
            ).fetchall()
            categories = {row["category"]: row["cnt"] for row in cat_rows}

        db_size: int = 0
        try:
            db_size = os.path.getsize(self._db_path)
        except OSError:
            pass

        return {
            "raw_total": raw_total,
            "raw_uncondensed": raw_uncondensed,
            "condensed_total": condensed_total,
            "categories": categories,
            "db_path": self._db_path,
            "db_size_bytes": db_size,
        }

    # -- housekeeping -------------------------------------------------------

    def close(self) -> None:
        """Close the current thread's database connection if open.

        Safe to call multiple times — subsequent calls are no-ops.
        Other threads' connections are unaffected.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def __repr__(self) -> str:
        """Return a developer-friendly representation including the database path."""
        return f"<EnhancedMemoryStore db={self._db_path!r}>"

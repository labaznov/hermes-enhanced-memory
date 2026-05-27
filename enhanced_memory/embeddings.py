"""Semantic vector search for the enhanced-memory plugin.

Provides KNN (K-Nearest Neighbours) vector search using the ``sqlite-vec``
extension.  Embedding generation is delegated to pluggable providers
(:class:`~embedding_providers.EmbeddingProvider` subclasses) — Gemini, OpenAI,
or local sentence-transformers.

Architecture::

    query text
        │
        ▼
    EmbeddingProvider.embed_single(query)
        │
        ▼
    sqlite-vec KNN: vec_memory WHERE embedding MATCH ?
        │
        ▼
    [fact_id, distance, similarity] results

ID mapping:
    Raw-fact IDs are stored directly.  Condensed-entry IDs are mapped to
    negative space via ``-(id + 10_000)`` so both tables can share a single
    ``vec_memory`` virtual table without primary-key collisions.

Configuration (``config.yaml``)::

    plugins:
      enhanced-memory:
        embedding_provider: gemini        # "gemini", "openai", "local", "none"
        embedding_model: gemini-embedding-001
        embedding_dims: 3072
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from datetime import datetime, timezone
from typing import Any

# try/except pattern: when loaded as part of the ``enhanced-memory`` package
# (e.g. by Hermes Agent), relative imports work.  When run standalone or in
# tests, fall back to absolute imports from the same directory.
try:
    from .embedding_providers import EmbeddingProvider, create_embedding_provider
except ImportError:
    from embedding_providers import EmbeddingProvider, create_embedding_provider

logger = logging.getLogger(__name__)

# Condensed-table IDs are mapped to negative space to avoid PK collisions
# with raw_facts IDs in the shared ``vec_memory`` virtual table.
# Formula: vec_id = -(original_id + _CONDENSED_ID_OFFSET)
# To recover: original_id = -(vec_id) - _CONDENSED_ID_OFFSET
_CONDENSED_ID_OFFSET = 10_000


def _serialize_f32(vec: list[float]) -> bytes:
    """Pack a list of floats into a compact little-endian binary blob.

    The ``sqlite-vec`` extension expects embedding vectors as raw bytes
    in little-endian IEEE 754 single-precision format.

    Args:
        vec: Embedding vector as a Python list of floats.

    Returns:
        bytes: Binary blob of ``len(vec) * 4`` bytes.
    """
    return struct.pack(f"<{len(vec)}f", *vec)


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load the ``sqlite-vec`` extension into *conn*.

    Uses the ``sqlite_vec`` Python package's :func:`load` helper which
    locates and loads the correct shared library for the current platform.

    Args:
        conn: An open SQLite connection.

    Returns:
        bool: ``True`` if the extension was loaded successfully, ``False``
        if ``sqlite_vec`` is not installed or the extension could not be
        loaded (e.g. SQLite compiled without extension support).
    """
    try:
        import sqlite_vec  # type: ignore[import-untyped]
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except (ImportError, OSError, sqlite3.OperationalError) as exc:
        logger.debug("sqlite-vec not available: %s", exc)
        return False


class SemanticSearch:
    """Semantic vector search backed by configurable embedding providers + sqlite-vec.

    Manages a ``vec0`` virtual table (``vec_memory``) alongside a companion
    ``vec_index_log`` table that tracks which facts have already been indexed.
    Embedding generation is delegated to an :class:`EmbeddingProvider`.

    Attributes:
        _db_path (str): Path to the SQLite database (same file as the store).
        _vec_available (bool): Whether ``sqlite-vec`` was loaded successfully.
        _conn (sqlite3.Connection | None): Dedicated connection for vec
            operations (separate from the store's per-thread connections).
        _provider (EmbeddingProvider | None): Active embedding provider.

    Args:
        db_path: Path to the SQLite database for the ``vec0`` virtual table.
        config: Plugin config dict (used to create the embedding provider
            via :func:`create_embedding_provider`).
        provider: Explicit provider instance — overrides config-based creation.

    Example::

        sem = SemanticSearch("/tmp/memory.db", config={"embedding_provider": "gemini"})
        if sem.is_available():
            results = sem.search("dark mode preference", k=3)
    """

    def __init__(
        self,
        db_path: str,
        config: dict[str, Any] | None = None,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialise the semantic search engine.

        Opens a dedicated SQLite connection, attempts to load ``sqlite-vec``,
        and creates the ``vec_memory`` / ``vec_index_log`` tables if the
        extension is available and a provider is configured.

        Args:
            db_path: Path to the SQLite database file.
            config: Plugin configuration dict for provider creation.
            provider: Pre-built provider (takes precedence over *config*).
        """
        self._db_path = db_path
        self._vec_available = False
        self._conn: sqlite3.Connection | None = None

        # Create or use provided embedding provider
        if provider is not None:
            self._provider = provider
        elif config:
            self._provider = create_embedding_provider(config)
        else:
            # Legacy fallback: default to Gemini when no config is given.
            from .embedding_providers import GeminiEmbedding
            self._provider = GeminiEmbedding()

        # Open a dedicated connection for vec operations.  This is separate
        # from the store's per-thread connections because sqlite-vec requires
        # the extension to be loaded on the connection that uses it.
        try:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._vec_available = _load_sqlite_vec(self._conn)
        except sqlite3.Error as exc:
            logger.error("Failed to open vec database: %s", exc)
            return

        # Only create tables if we have both the extension and a provider.
        if self._vec_available and self._provider:
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create the ``vec0`` virtual table and ``vec_index_log`` if needed.

        The ``vec_memory`` table stores (fact_id, embedding) pairs using the
        ``vec0`` module from ``sqlite-vec``.  The ``vec_index_log`` table
        tracks which fact IDs have been indexed and from which source table.

        On failure, ``_vec_available`` is set to ``False`` to gracefully
        disable vector search.
        """
        dims = self._provider.dims if self._provider else 3072
        cur = self._conn.cursor()
        try:
            # vec0 virtual table: stores embedding blobs keyed by fact_id.
            cur.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory
                USING vec0(
                    fact_id INTEGER PRIMARY KEY,
                    embedding float[{dims}]
                )
                """
            )
            # Index log: tracks which facts have been embedded so we can
            # efficiently discover unindexed facts.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vec_index_log (
                    fact_id      INTEGER PRIMARY KEY,
                    source_table TEXT NOT NULL,
                    indexed_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("Failed to create vec tables: %s", exc)
            self._vec_available = False

    def is_available(self) -> bool:
        """Return ``True`` when sqlite-vec is loaded AND the embedding provider is ready.

        Both conditions must be met for vector search to function.

        Returns:
            bool: Whether semantic search can be performed.
        """
        return (
            self._vec_available
            and self._provider is not None
            and self._provider.is_available()
        )

    @property
    def provider_name(self) -> str:
        """Name of the active embedding provider.

        Returns:
            str: Provider name (e.g. ``'gemini'``) or ``'none'`` if no
            provider is configured.
        """
        return self._provider.name if self._provider else "none"

    @property
    def dims(self) -> int:
        """Embedding dimensions from the active provider.

        Returns:
            int: Number of floats per embedding vector, or ``0`` if no
            provider is configured.
        """
        return self._provider.dims if self._provider else 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_facts(self, facts: list[dict[str, Any]], source_table: str) -> int:
        """Embed facts and insert them into the ``vec_memory`` table.

        Each fact is embedded via the configured provider, then stored as
        a binary blob.  For condensed entries, the fact ID is mapped to
        negative space (``-(id + 10_000)``) to avoid collisions with
        raw_facts IDs in the shared virtual table.

        Args:
            facts: List of dicts with ``'id'`` (int) and ``'content'`` (str).
            source_table: ``'raw_facts'`` or ``'condensed'`` — determines
                the ID mapping strategy.

        Returns:
            int: Number of facts successfully indexed (may be less than
            ``len(facts)`` if individual inserts fail).
        """
        if not facts or not self.is_available():
            return 0

        texts = [f["content"] for f in facts]
        try:
            embeddings = self._provider.embed_texts(texts)
        except Exception as exc:
            logger.error("Embedding failed during indexing: %s", exc)
            return 0

        indexed = 0
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()

        for fact, emb in zip(facts, embeddings):
            raw_id: int = fact["id"]
            # Map condensed IDs to negative space to avoid PK collision
            # with raw_facts IDs in the shared vec_memory table.
            vec_id = -(raw_id + _CONDENSED_ID_OFFSET) if source_table == "condensed" else raw_id
            blob = _serialize_f32(emb)  # convert float list → binary blob

            try:
                cur.execute(
                    "INSERT OR REPLACE INTO vec_memory(fact_id, embedding) VALUES (?, ?)",
                    (vec_id, blob),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO vec_index_log(fact_id, source_table, indexed_at) "
                    "VALUES (?, ?, ?)",
                    (vec_id, source_table, now),
                )
                indexed += 1
            except sqlite3.Error as exc:
                logger.warning("Failed to index fact %d (vec_id=%d): %s", raw_id, vec_id, exc)

        self._conn.commit()
        logger.info("Indexed %d/%d facts from %s", indexed, len(facts), source_table)
        return indexed

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Run a KNN vector search for the query string.

        The query is embedded via the provider, then matched against
        ``vec_memory`` using the ``sqlite-vec`` ``MATCH`` operator which
        performs approximate nearest-neighbour search.

        Args:
            query: Natural-language search query.
            k: Maximum number of nearest neighbours to return.

        Returns:
            list[dict[str, Any]]: Results sorted by ascending distance,
            each containing:

            - ``fact_id`` (int): The stored fact ID (negative for condensed).
            - ``distance`` (float): L2 distance from the query embedding.
            - ``similarity`` (float): ``1.0 - distance`` (higher = better).

            Returns an empty list if the provider is unavailable or
            embedding fails.
        """
        if not self.is_available():
            return []

        try:
            emb = self._provider.embed_single(query)
        except Exception as exc:
            logger.error("Embedding failed during search: %s", exc)
            return []

        blob = _serialize_f32(emb)
        try:
            rows = self._conn.execute(
                "SELECT fact_id, distance FROM vec_memory "
                "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (blob, k),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.error("Vec search failed: %s", exc)
            return []

        return [
            {
                "fact_id": row["fact_id"],
                "distance": row["distance"],
                "similarity": 1.0 - row["distance"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Unindexed discovery
    # ------------------------------------------------------------------

    def get_unindexed(self, conn_or_store: Any) -> dict[str, list[dict[str, Any]]]:
        """Find facts that have not yet been indexed in ``vec_memory``.

        Compares the ``vec_index_log`` against the ``raw_facts`` and
        ``condensed`` tables in the memory store to identify facts that
        need embedding and indexing.

        Args:
            conn_or_store: Either a ``sqlite3.Connection`` or an object
                with a ``.conn`` property (e.g. :class:`EnhancedMemoryStore`).

        Returns:
            dict[str, list[dict[str, Any]]]: Keys ``'raw_facts'`` and
            ``'condensed'``, each mapping to a list of ``{id, content}``
            dicts for unindexed entries.
        """
        result: dict[str, list[dict[str, Any]]] = {"raw_facts": [], "condensed": []}
        if not self._vec_available:
            return result

        # Use the .conn property if available (EnhancedMemoryStore), otherwise
        # treat as a raw sqlite3.Connection.
        mem_conn = conn_or_store.conn if hasattr(conn_or_store, "conn") else conn_or_store

        # Collect already-indexed IDs from the log, split by source table.
        indexed_raw: set[int] = set()
        indexed_condensed: set[int] = set()
        try:
            for row in self._conn.execute("SELECT fact_id, source_table FROM vec_index_log"):
                if row["source_table"] == "condensed":
                    indexed_condensed.add(row["fact_id"])
                else:
                    indexed_raw.add(row["fact_id"])
        except sqlite3.Error:
            pass

        # Discover raw facts not yet in the index.
        try:
            for row in mem_conn.execute("SELECT id, content FROM raw_facts"):
                # Handle both Row objects and plain tuples.
                fid = row[0] if isinstance(row, (tuple, list)) else row["id"]
                content = row[1] if isinstance(row, (tuple, list)) else row["content"]
                if fid not in indexed_raw:
                    result["raw_facts"].append({"id": fid, "content": content})
        except sqlite3.Error:
            pass

        # Discover condensed entries not yet in the index.
        try:
            for row in mem_conn.execute("SELECT id, summary FROM condensed"):
                fid = row[0] if isinstance(row, (tuple, list)) else row["id"]
                content = row[1] if isinstance(row, (tuple, list)) else row["summary"]
                # Apply the same negative-space mapping used during indexing.
                vec_id = -(fid + _CONDENSED_ID_OFFSET)
                if vec_id not in indexed_condensed:
                    result["condensed"].append({"id": fid, "content": content})
        except sqlite3.Error:
            pass

        return result

    # ------------------------------------------------------------------
    # Reindex
    # ------------------------------------------------------------------

    def reindex(self, conn_or_store: Any | None = None) -> dict[str, int]:
        """Drop and rebuild the entire vector index from scratch.

        Clears both ``vec_memory`` and ``vec_index_log``, then re-indexes
        all raw facts and condensed entries from the memory store.

        Args:
            conn_or_store: A ``sqlite3.Connection`` or an object with a
                ``.conn`` property.  If ``None``, only the index is cleared
                (no re-indexing is performed).

        Returns:
            dict[str, int]: Counts of re-indexed entries keyed by
            ``'raw_facts'`` and ``'condensed'``.
        """
        counts: dict[str, int] = {"raw_facts": 0, "condensed": 0}
        if not self.is_available():
            return counts

        try:
            self._conn.execute("DELETE FROM vec_memory")
            self._conn.execute("DELETE FROM vec_index_log")
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("Failed to clear vec tables: %s", exc)
            return counts

        if conn_or_store is None:
            return counts

        mem_conn = conn_or_store.conn if hasattr(conn_or_store, "conn") else conn_or_store

        # Re-index all raw facts.
        try:
            rows = mem_conn.execute("SELECT id, content FROM raw_facts").fetchall()
            raw_facts = [
                {"id": r[0] if isinstance(r, (tuple, list)) else r["id"],
                 "content": r[1] if isinstance(r, (tuple, list)) else r["content"]}
                for r in rows
            ]
            if raw_facts:
                counts["raw_facts"] = self.index_facts(raw_facts, "raw_facts")
        except sqlite3.Error:
            pass

        # Re-index all condensed entries.
        try:
            rows = mem_conn.execute("SELECT id, summary FROM condensed").fetchall()
            condensed = [
                {"id": r[0] if isinstance(r, (tuple, list)) else r["id"],
                 "content": r[1] if isinstance(r, (tuple, list)) else r["summary"]}
                for r in rows
            ]
            if condensed:
                counts["condensed"] = self.index_facts(condensed, "condensed")
        except sqlite3.Error:
            pass

        return counts

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return statistics about the vector index.

        Returns:
            dict[str, Any]: Dictionary with keys:

            - ``available`` (bool): Whether semantic search is operational.
            - ``vec_loaded`` (bool): Whether sqlite-vec was loaded.
            - ``provider`` (str): Active provider name.
            - ``provider_available`` (bool): Whether the provider is ready.
            - ``embedding_dims`` (int): Embedding dimensionality.
            - ``total_indexed`` (int): Total entries in ``vec_index_log``.
            - ``by_source`` (dict[str, int]): Counts per source table.
        """
        result: dict[str, Any] = {
            "available": self.is_available(),
            "vec_loaded": self._vec_available,
            "provider": self.provider_name,
            "provider_available": self._provider.is_available() if self._provider else False,
            "embedding_dims": self.dims,
            "total_indexed": 0,
            "by_source": {},
        }

        if not self._vec_available or not self._conn:
            return result

        try:
            row = self._conn.execute("SELECT COUNT(*) FROM vec_index_log").fetchone()
            result["total_indexed"] = row[0]
        except sqlite3.Error:
            pass

        try:
            for row in self._conn.execute(
                "SELECT source_table, COUNT(*) as cnt FROM vec_index_log GROUP BY source_table"
            ):
                result["by_source"][row["source_table"]] = row["cnt"]
        except sqlite3.Error:
            pass

        return result

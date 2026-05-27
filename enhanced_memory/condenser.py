"""Enhanced Memory Plugin — Fact Condensation Engine.

Two-tier architecture: ``raw_facts`` are periodically condensed into
high-priority summaries grouped by category and topic.  The condenser
deduplicates, prioritises, and merges entries so that the memory
injected into the system prompt stays compact and relevant.

Pipeline overview::

    raw_facts (uncondensed=0)
        │
        ▼
    group by category
        │
        ▼
    deduplicate (80% word-overlap threshold)
        │
        ▼
    compute priority (category base + keyword boosts)
        │
        ▼
    upsert into condensed table (create or merge)
        │
        ▼
    mark source raw_facts as condensed=1

Priority calculation:
    Each category has a ``(lo, hi)`` base range.  Keywords in the text can
    boost the score by +1 or +2 (e.g. "password" → +2).  The final priority
    is clamped to ``[lo, 10]``.

Deduplication:
    Uses Jaccard-style word overlap on the *smaller* set.  Facts with ≥ 80%
    overlap with an already-accepted fact are discarded (first-in wins).

Usage::

    condenser = FactCondenser(store)
    results = condenser.condense()              # write results to DB
    results = condenser.condense(dry_run=True)  # preview without writing
    memory  = condenser.get_top_for_memory(char_limit=2200)
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .store import EnhancedMemoryStore

logger = logging.getLogger("enhanced-memory.condenser")

# ── Topic display names (bilingual) ──────────────────────────────────────────
# Human-readable topic labels used as the ``condensed.topic`` column value.
# Keys correspond to valid category labels from :data:`store.VALID_CATEGORIES`.
TOPIC_NAMES: dict[str, str] = {
    "user_pref": "Пользователь: предпочтения",
    "project": "Проекты и работа",
    "tool": "Инструменты и настройки",
    "env": "Среда и инфраструктура",
    "decision": "Решения и выборы",
    "security": "Безопасность",
    "general": "Общее",
}

# ── Base priority ranges per category ────────────────────────────────────────
# Tuple of (lo, hi) for each category.  ``lo`` is the starting priority;
# ``hi`` is the theoretical ceiling *before* keyword boosts.  Keyword boosts
# can push the value above ``hi`` but never above 10.
_CATEGORY_PRIORITY: dict[str, tuple[int, int]] = {
    "security": (9, 10),
    "user_pref": (8, 9),
    "decision": (7, 9),
    "project": (7, 7),
    "tool": (6, 8),
    "env": (5, 5),
    "general": (4, 4),
}

# ── Keyword boost tables ────────────────────────────────────────────────────
# If any of these keywords appear (case-insensitive) in the fact text, the
# priority gets a +1 boost.  Includes both English and Russian equivalents.
_BOOST_1_KEYWORDS: set[str] = {
    "prefers", "always", "never",
    "предпочитает", "всегда", "никогда",
}

# +2 boost keywords — security-sensitive terms that should surface higher.
_BOOST_2_KEYWORDS: set[str] = {
    "password", "key", "secret",
    "пароль", "ключ", "секрет",
}

# ── Deduplication threshold ─────────────────────────────────────────────────
# Minimum word overlap ratio (Jaccard on the smaller set) to consider two
# facts as near-duplicates.  0.80 means 80% of the smaller token set must
# appear in the larger.
_OVERLAP_THRESHOLD: float = 0.80


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Tokenise text into a set of lowercase words.

    Uses a simple ``\\w+`` regex so punctuation and whitespace are stripped.

    Args:
        text: The input string.

    Returns:
        set[str]: Unique lowercase word tokens.
    """
    return set(re.findall(r"[\w]+", text.lower()))


def _word_overlap(a: str, b: str) -> float:
    """Compute the word overlap ratio between two strings.

    The ratio is ``|intersection| / min(|A|, |B|)`` — i.e. what fraction
    of the *smaller* token set is contained in the *larger*.  This is
    biased toward detecting subsets (a short fact that is fully contained
    in a longer one scores 1.0).

    Args:
        a: First string.
        b: Second string.

    Returns:
        float: Overlap ratio in ``[0.0, 1.0]``.  Returns ``0.0`` if
        either string produces no tokens.
    """
    wa, wb = _tokenize(a), _tokenize(b)
    if not wa or not wb:
        return 0.0
    intersection = wa & wb
    smaller = min(len(wa), len(wb))
    # Guard against zero-division (shouldn't happen given the check above).
    return len(intersection) / smaller if smaller else 0.0


def _compute_priority(category: str, text: str) -> int:
    """Determine priority score for a fact based on category and keyword boosts.

    Algorithm:
        1. Look up the ``(lo, hi)`` base range for the category (default ``(4, 4)``).
        2. Start at ``lo``.
        3. Scan the text for boost keywords: ``+2`` for security-sensitive
           terms, ``+1`` for preference/absoluteness terms (boosts stack).
        4. Clamp the result to ``[lo, 10]``.

    Args:
        category: The fact's category label.
        text: The fact text to scan for boost keywords.

    Returns:
        int: Priority score in the range ``[1, 10]``.
    """
    lo, hi = _CATEGORY_PRIORITY.get(category, (4, 4))
    base = lo  # start at the category's lower bound

    words_lower = text.lower()

    boost = 0
    # +2 boost for security-sensitive keywords (stacks with +1)
    if any(kw in words_lower for kw in _BOOST_2_KEYWORDS):
        boost += 2
    # +1 boost for preference / absoluteness keywords
    if any(kw in words_lower for kw in _BOOST_1_KEYWORDS):
        boost += 1

    priority = min(base + boost, 10)  # hard cap at 10 regardless of boosts
    # Ensure we never drop below the category floor
    priority = max(priority, lo)
    return priority


def _merge_source_ids(existing: list[int] | str | None, new_ids: list[int]) -> str:
    """Merge two lists of source IDs, deduplicate, and return a JSON string.

    Handles multiple input formats for *existing* because the value may
    come directly from the database (JSON string) or from Python code
    (list or ``None``).

    Args:
        existing: The current source IDs — may be a Python list, a JSON
            string, or ``None``.
        new_ids: New IDs to merge in.

    Returns:
        str: JSON-encoded sorted list of unique IDs.
    """
    if existing is None:
        prev: list[int] = []
    elif isinstance(existing, str):
        try:
            prev = json.loads(existing)
        except (json.JSONDecodeError, TypeError):
            prev = []
    else:
        prev = list(existing)

    merged = sorted(set(prev) | set(new_ids))
    return json.dumps(merged)


# ── Main condenser class ────────────────────────────────────────────────────


class FactCondenser:
    """Groups, deduplicates, and summarises raw facts into condensed entries.

    The condenser operates on the :class:`EnhancedMemoryStore` and implements
    the core condensation pipeline described in the module docstring.

    Attributes:
        store (EnhancedMemoryStore): The backing store instance used for
            reading uncondensed facts and writing condensed summaries.

    Args:
        store: An initialised :class:`EnhancedMemoryStore`.

    Example::

        condenser = FactCondenser(store)
        results = condenser.condense()              # production run
        results = condenser.condense(dry_run=True)  # preview only
        memory_text = condenser.get_top_for_memory()
    """

    def __init__(self, store: "EnhancedMemoryStore") -> None:
        self.store = store

    # ── Public API ───────────────────────────────────────────────────────

    def condense(self, dry_run: bool = False) -> list[dict[str, Any]]:
        """Run the full condensation pipeline.

        Steps:
            1. Load uncondensed raw_facts from the store.
            2. Group by category.
            3. Deduplicate within each group (80 % word-overlap threshold).
            4. Assign priority per category + keyword boosts.
            5. Create or update condensed entries (merge ``source_ids``
               when a matching topic + category already exists).
            6. Mark processed raw_facts as condensed.
            7. Return a list of created / updated condensed entry dicts.

        Args:
            dry_run: If *True*, compute results but do **not** write to the
                store or mark any facts as condensed.

        Returns:
            List of dicts, each with keys ``topic``, ``category``,
            ``summary``, ``priority``, ``source_ids``, ``action``
            (``'created'`` or ``'updated'``).
        """
        raw_facts = self._load_uncondensed()
        if not raw_facts:
            logger.info("No uncondensed facts to process.")
            return []

        grouped = self._group_by_category(raw_facts)
        results: list[dict[str, Any]] = []
        all_processed_ids: list[int] = []

        for category, facts in grouped.items():
            deduplicated = self._deduplicate(facts)
            topic = TOPIC_NAMES.get(category, TOPIC_NAMES["general"])

            # Build a single summary from the surviving facts
            source_ids: list[int] = []
            summaries: list[str] = []
            for fact in deduplicated:
                fid = fact.get("id")
                if fid is not None:
                    source_ids.append(int(fid))
                    all_processed_ids.append(int(fid))
                text = fact.get("content", fact.get("text", "")).strip()
                if text:
                    summaries.append(text)

            if not summaries:
                continue

            summary = "; ".join(summaries)
            priority = max(
                (_compute_priority(category, s) for s in summaries),
                default=_CATEGORY_PRIORITY.get(category, (4, 4))[0],
            )

            entry: dict[str, Any] = {
                "topic": topic,
                "category": category,
                "summary": summary,
                "priority": priority,
                "source_ids": source_ids,
            }

            if not dry_run:
                action = self._upsert_condensed(entry)
                entry["action"] = action
            else:
                entry["action"] = "dry_run"

            results.append(entry)

        # Mark originals as condensed
        if not dry_run and all_processed_ids:
            self._mark_condensed(all_processed_ids)

        logger.info(
            "Condensation complete: %d entries %s.",
            len(results),
            "previewed (dry-run)" if dry_run else "written",
        )
        return results

    def get_top_for_memory(self, char_limit: int = 2200) -> str:
        """Return a compact string of the highest-priority condensed entries.

        Entries are sorted by ``priority DESC, updated_at DESC`` and
        concatenated with the ``§`` separator until *char_limit* is reached.
        If even the first entry exceeds the limit it is truncated to fit.

        Args:
            char_limit: Maximum character count for the returned string.
                Defaults to 2200 which is roughly the space available in
                a typical system prompt memory block.

        Returns:
            str: A ``§``-separated string of condensed summaries, or an
            empty string if no condensed entries exist.
        """
        entries = self._load_condensed_sorted()
        parts: list[str] = []
        current_len = 0
        separator = "§"  # non-ASCII separator unlikely to appear in content
        sep_len = len(separator)

        for entry in entries:
            summary = entry.get("summary", "").strip()
            if not summary:
                continue

            # Calculate space needed including separator if not the first part.
            addition_len = len(summary) + (sep_len if parts else 0)
            if current_len + addition_len > char_limit:
                # If nothing has been added yet, truncate to fit at least something.
                if not parts:
                    parts.append(summary[: char_limit])
                break

            parts.append(summary)
            current_len += addition_len

        return separator.join(parts)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _load_uncondensed(self) -> list[dict[str, Any]]:
        """Fetch raw facts where ``condensed = 0``, ordered chronologically.

        Returns:
            list[dict[str, Any]]: Facts with keys ``id``, ``content``,
            ``category``, ``created_at``.  Returns an empty list on error.
        """
        try:
            conn = self.store.get_connection()
            cursor = conn.execute(
                "SELECT id, content, category, created_at "
                "FROM raw_facts WHERE condensed = 0 ORDER BY created_at ASC"
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            logger.exception("Failed to load uncondensed facts.")
            return []

    def _group_by_category(
        self, facts: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group a flat list of facts by their ``category`` field.

        Args:
            facts: List of fact dicts, each with a ``category`` key.

        Returns:
            dict[str, list[dict[str, Any]]]: Category → list of facts.
            Facts with a missing or empty category default to ``'general'``.
        """
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fact in facts:
            cat = fact.get("category", "general") or "general"
            grouped[cat].append(fact)
        return dict(grouped)

    def _deduplicate(
        self, facts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove near-duplicates using word-overlap threshold.

        Keeps the *first* occurrence (oldest by insertion order).  A fact is
        considered a duplicate if it shares ≥ 80% word overlap with any
        already-accepted fact.

        Args:
            facts: Flat list of fact dicts within one category.

        Returns:
            list[dict[str, Any]]: Deduplicated facts in original order.
        """
        accepted: list[dict[str, Any]] = []
        accepted_texts: list[str] = []  # parallel list of texts for O(n²) comparison

        for fact in facts:
            text = fact.get("content", fact.get("text", "")).strip()
            if not text:
                continue

            # Check against every already-accepted text for overlap.
            is_dup = False
            for existing_text in accepted_texts:
                if _word_overlap(text, existing_text) >= _OVERLAP_THRESHOLD:
                    is_dup = True
                    break

            if not is_dup:
                accepted.append(fact)
                accepted_texts.append(text)

        removed = len(facts) - len(accepted)
        if removed:
            logger.debug("Deduplicated %d/%d facts.", removed, len(facts))
        return accepted

    def _upsert_condensed(self, entry: dict[str, Any]) -> str:
        """Insert or update a condensed entry in the database.

        If a row with the same ``(topic, category)`` already exists, the
        summary text is *appended* (separated by ``"; "``), source IDs are
        merged, and the higher priority wins.  Otherwise a new row is created.

        Args:
            entry: Dict with keys ``topic``, ``category``, ``summary``,
                ``priority``, ``source_ids``.

        Returns:
            str: ``'created'`` if a new row was inserted, ``'updated'``
            if an existing row was modified.
        """
        conn = self.store.get_connection()
        # Look up existing row by (topic, category) unique index.
        cursor = conn.execute(
            "SELECT id, source_ids FROM condensed "
            "WHERE topic = ? AND category = ?",
            (entry["topic"], entry["category"]),
        )
        row = cursor.fetchone()

        # Use Unix timestamp for updated_at/created_at in condensed entries.
        now = int(time.time())

        if row:
            existing_id, existing_source_ids = row
            merged_ids = _merge_source_ids(existing_source_ids, entry["source_ids"])
            # Append the new summary text to the existing one.
            cursor2 = conn.execute(
                "SELECT summary FROM condensed WHERE id = ?", (existing_id,)
            )
            old_summary = cursor2.fetchone()[0] or ""
            new_summary = (
                f"{old_summary}; {entry['summary']}" if old_summary else entry["summary"]
            )
            # Keep whichever priority is higher (existing vs incoming).
            cursor3 = conn.execute(
                "SELECT priority FROM condensed WHERE id = ?", (existing_id,)
            )
            old_priority = cursor3.fetchone()[0] or 0
            final_priority = max(old_priority, entry["priority"])

            conn.execute(
                "UPDATE condensed SET summary = ?, priority = ?, "
                "source_ids = ?, updated_at = ? WHERE id = ?",
                (new_summary, final_priority, merged_ids, now, existing_id),
            )
            conn.commit()
            logger.debug("Updated condensed entry id=%d topic=%r.", existing_id, entry["topic"])
            return "updated"
        else:
            source_ids_json = json.dumps(entry["source_ids"])
            conn.execute(
                "INSERT INTO condensed (topic, category, summary, priority, "
                "source_ids, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry["topic"],
                    entry["category"],
                    entry["summary"],
                    entry["priority"],
                    source_ids_json,
                    now,
                    now,
                ),
            )
            conn.commit()
            logger.debug("Created condensed entry topic=%r.", entry["topic"])
            return "created"

    def _mark_condensed(self, fact_ids: list[int]) -> None:
        """Set ``condensed = 1`` on the given raw_fact IDs.

        Args:
            fact_ids: List of ``raw_facts.id`` values.  An empty list
                is a no-op.

        Raises:
            Exception: Logged and suppressed; the caller is not interrupted.
        """
        if not fact_ids:
            return
        try:
            conn = self.store.get_connection()
            # Dynamic IN clause with positional placeholders.
            placeholders = ", ".join("?" for _ in fact_ids)
            conn.execute(
                f"UPDATE raw_facts SET condensed = 1 WHERE id IN ({placeholders})",
                fact_ids,
            )
            conn.commit()
            logger.debug("Marked %d raw facts as condensed.", len(fact_ids))
        except Exception:
            logger.exception("Failed to mark facts as condensed.")

    def _load_condensed_sorted(self) -> list[dict[str, Any]]:
        """Load all condensed entries ordered by priority descending.

        Secondary sort is ``updated_at DESC`` so recently-updated entries
        of the same priority appear first.

        Returns:
            list[dict[str, Any]]: All condensed rows as dicts, or an
            empty list on error.
        """
        try:
            conn = self.store.get_connection()
            cursor = conn.execute(
                "SELECT id, topic, category, summary, priority, source_ids, "
                "created_at, updated_at FROM condensed ORDER BY priority DESC, updated_at DESC"
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            logger.exception("Failed to load condensed entries.")
            return []

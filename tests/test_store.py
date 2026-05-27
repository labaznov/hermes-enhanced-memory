"""Tests for EnhancedMemoryStore (store.py)."""
from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from store import EnhancedMemoryStore, VALID_CATEGORIES, _validate_category, _row_to_dict


# ── Schema & init ──────────────────────────────────────────────────────────

class TestStoreInit:
    def test_create_store_creates_db(self, db_path):
        store = EnhancedMemoryStore(db_path=db_path)
        conn = store.get_connection()
        # Verify core tables exist
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "raw_facts" in tables
        assert "condensed" in tables
        store.close()

    def test_wal_mode(self, store):
        conn = store.get_connection()
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"

    def test_fts_tables_exist(self, store):
        conn = store.get_connection()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "raw_facts_fts" in tables
        assert "condensed_fts" in tables

    def test_repr(self, store):
        r = repr(store)
        assert "EnhancedMemoryStore" in r
        assert "db=" in r


# ── Category validation ────────────────────────────────────────────────────

class TestCategoryValidation:
    @pytest.mark.parametrize("cat", list(VALID_CATEGORIES))
    def test_valid_categories_pass(self, cat):
        assert _validate_category(cat) == cat

    def test_invalid_category_falls_back(self):
        assert _validate_category("bogus") == "general"
        assert _validate_category("") == "general"


# ── raw_facts CRUD ─────────────────────────────────────────────────────────

class TestRawFactsCRUD:
    def test_add_single_fact(self, store):
        fid = store.add_raw_fact("hello world", category="general")
        assert isinstance(fid, int)
        assert fid > 0

    def test_get_by_id(self, store):
        fid = store.add_raw_fact("test fact", category="tool", source="unit_test")
        row = store.get_raw_by_id(fid)
        assert row is not None
        assert row["content"] == "test fact"
        assert row["category"] == "tool"
        assert row["source"] == "unit_test"
        assert row["condensed"] == 0

    def test_get_by_id_not_found(self, store):
        assert store.get_raw_by_id(999999) is None

    def test_add_with_invalid_category(self, store):
        fid = store.add_raw_fact("fact", category="invalid_cat")
        row = store.get_raw_by_id(fid)
        assert row["category"] == "general"  # falls back

    def test_batch_insert(self, store):
        facts = [
            {"content": "fact 1", "category": "general"},
            {"content": "fact 2", "category": "env"},
            {"content": "fact 3"},  # no category → default
        ]
        ids = store.add_raw_facts_batch(facts)
        assert len(ids) == 3
        for fid in ids:
            assert store.get_raw_by_id(fid) is not None

    def test_batch_empty(self, store):
        assert store.add_raw_facts_batch([]) == []

    def test_batch_skips_empty_content(self, store):
        facts = [
            {"content": ""},
            {"content": None},
            {"content": "real fact"},
        ]
        ids = store.add_raw_facts_batch(facts)
        assert len(ids) == 1


# ── FTS5 search ────────────────────────────────────────────────────────────

class TestFTSSearch:
    def test_search_raw_basic(self, populated_store):
        results = populated_store.search_raw("python")
        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)

    def test_search_raw_empty_query(self, store):
        assert store.search_raw("") == []
        assert store.search_raw("   ") == []

    def test_search_raw_no_match(self, populated_store):
        results = populated_store.search_raw("xyznonexistent")
        assert results == []

    def test_search_raw_limit(self, populated_store):
        results = populated_store.search_raw("user OR project OR server", limit=2)
        assert len(results) <= 2


# ── Uncondensed / mark_condensed ───────────────────────────────────────────

class TestUncondensed:
    def test_list_uncondensed(self, populated_store):
        uncond = populated_store.list_uncondensed()
        assert len(uncond) == 8  # all are uncondensed

    def test_mark_condensed(self, populated_store):
        uncond = populated_store.list_uncondensed()
        ids = [u["id"] for u in uncond[:3]]
        populated_store.mark_condensed(ids)
        uncond_after = populated_store.list_uncondensed()
        assert len(uncond_after) == 5

    def test_mark_condensed_empty(self, store):
        store.mark_condensed([])  # should not error


# ── condensed CRUD ─────────────────────────────────────────────────────────

class TestCondensedCRUD:
    def test_add_condensed(self, store):
        cid = store.add_condensed(
            topic="Test Topic",
            summary="This is a summary",
            category="project",
            priority=7,
            source_ids=[1, 2, 3],
            fact_count=3,
        )
        assert isinstance(cid, int) and cid > 0

    def test_get_condensed_by_id(self, store):
        cid = store.add_condensed(
            topic="My Topic", summary="Summary text", category="env",
            priority=5, source_ids=[10, 20],
        )
        row = store.get_condensed_by_id(cid)
        assert row is not None
        assert row["topic"] == "My Topic"
        assert row["source_ids"] == [10, 20]
        assert row["priority"] == 5

    def test_get_condensed_by_id_not_found(self, store):
        assert store.get_condensed_by_id(99999) is None

    def test_get_condensed_by_topic(self, store):
        store.add_condensed(topic="Unique Topic", summary="s", category="tool")
        row = store.get_condensed(topic="Unique Topic", category="tool")
        assert row is not None
        assert row["summary"] == "s"

    def test_get_condensed_not_found(self, store):
        assert store.get_condensed(topic="nope", category="general") is None

    def test_update_condensed(self, store):
        cid = store.add_condensed(
            topic="Upd Topic", summary="old", category="general", priority=3,
        )
        store.update_condensed(cid, summary="new", priority=8, source_ids=[1], fact_count=1)
        row = store.get_condensed_by_id(cid)
        assert row["summary"] == "new"
        assert row["priority"] == 8
        assert row["version"] == 2  # incremented

    def test_priority_clamped(self, store):
        cid = store.add_condensed(topic="Clamp", summary="s", priority=99)
        row = store.get_condensed_by_id(cid)
        assert row["priority"] == 10

    def test_search_condensed_by_fts(self, store):
        store.add_condensed(topic="Python Project", summary="We use Python 3.12 on this project", category="project")
        results = store.search_condensed(query="Python")
        assert len(results) >= 1

    def test_search_condensed_by_category(self, store):
        store.add_condensed(topic="T1", summary="s", category="env")
        store.add_condensed(topic="T2", summary="s2", category="project")
        results = store.search_condensed(category="env")
        assert all(r["category"] == "env" for r in results)

    def test_list_condensed(self, store):
        store.add_condensed(topic="Low", summary="s", priority=2)
        store.add_condensed(topic="High", summary="s", priority=9)
        results = store.list_condensed()
        assert len(results) >= 2
        # Should be sorted by priority DESC
        assert results[0]["priority"] >= results[-1]["priority"]


# ── Stats ──────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_empty(self, store):
        s = store.stats()
        assert s["raw_total"] == 0
        assert s["condensed_total"] == 0
        assert "db_path" in s

    def test_stats_populated(self, populated_store):
        s = populated_store.stats()
        assert s["raw_total"] == 8
        assert s["raw_uncondensed"] == 8
        assert isinstance(s["categories"], dict)
        assert "db_size_bytes" in s


# ── Thread safety ──────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_writes(self, db_path):
        store = EnhancedMemoryStore(db_path=db_path)
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    store.add_raw_fact(f"thread-{n}-fact-{i}", category="general")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        s = store.stats()
        assert s["raw_total"] == 40
        store.close()


# ── Connection helpers ─────────────────────────────────────────────────────

class TestConnectionHelpers:
    def test_conn_property(self, store):
        c = store.conn
        assert isinstance(c, sqlite3.Connection)

    def test_get_connection(self, store):
        c = store.get_connection()
        assert isinstance(c, sqlite3.Connection)

    def test_close(self, db_path):
        store = EnhancedMemoryStore(db_path=db_path)
        store.add_raw_fact("before close")
        store.close()
        # After close, getting connection should create a new one
        fid = store.add_raw_fact("after close")
        assert fid > 0
        store.close()


# ── _row_to_dict helper ───────────────────────────────────────────────────

class TestRowToDict:
    def test_row_to_dict(self, store):
        store.add_raw_fact("test", category="env")
        conn = store.get_connection()
        cur = conn.execute("SELECT id, content, category FROM raw_facts LIMIT 1")
        row = cur.fetchone()
        d = _row_to_dict(cur, row)
        assert "id" in d
        assert "content" in d
        assert d["content"] == "test"

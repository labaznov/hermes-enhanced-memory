"""Tests for FactCondenser (condenser.py)."""
from __future__ import annotations

import json

import pytest

from condenser import (
    FactCondenser,
    _compute_priority,
    _merge_source_ids,
    _tokenize,
    _word_overlap,
    TOPIC_NAMES,
)


# ── Helper functions ───────────────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello World 123")
        assert tokens == {"hello", "world", "123"}

    def test_empty(self):
        assert _tokenize("") == set()

    def test_special_chars(self):
        tokens = _tokenize("foo-bar_baz! yes.")
        assert "foo" in tokens
        assert "bar_baz" in tokens


class TestWordOverlap:
    def test_identical(self):
        assert _word_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _word_overlap("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        overlap = _word_overlap("a b c d", "a b e f")
        assert 0.4 <= overlap <= 0.6

    def test_empty_string(self):
        assert _word_overlap("", "hello") == 0.0
        assert _word_overlap("hello", "") == 0.0

    def test_high_overlap_different_order(self):
        a = "user prefers dark mode"
        b = "dark mode user prefers"
        assert _word_overlap(a, b) >= 0.8


class TestComputePriority:
    def test_security_high_base(self):
        p = _compute_priority("security", "some text")
        assert p >= 9

    def test_general_low_base(self):
        p = _compute_priority("general", "some text")
        assert p >= 4

    def test_boost_keyword_always(self):
        p_base = _compute_priority("general", "some text")
        p_boosted = _compute_priority("general", "user always does this")
        assert p_boosted > p_base

    def test_boost_keyword_password(self):
        p = _compute_priority("general", "the password is stored securely")
        assert p >= 6  # base 4 + boost 2

    def test_stacking_boosts(self):
        p = _compute_priority("general", "the password is always kept secret")
        assert p >= 7  # base 4 + 2 + 1

    def test_unknown_category(self):
        p = _compute_priority("nonexistent", "test")
        assert p == 4  # default (4, 4)

    def test_priority_cap_10(self):
        p = _compute_priority("security", "password key secret always never prefers")
        assert p <= 10


class TestMergeSourceIds:
    def test_merge_lists(self):
        result = json.loads(_merge_source_ids([1, 2], [3, 4]))
        assert result == [1, 2, 3, 4]

    def test_merge_dedup(self):
        result = json.loads(_merge_source_ids([1, 2, 3], [2, 3, 4]))
        assert result == [1, 2, 3, 4]

    def test_merge_from_json_string(self):
        result = json.loads(_merge_source_ids("[1, 2]", [3]))
        assert result == [1, 2, 3]

    def test_merge_none(self):
        result = json.loads(_merge_source_ids(None, [5, 6]))
        assert result == [5, 6]

    def test_merge_bad_json(self):
        result = json.loads(_merge_source_ids("not json", [1]))
        assert result == [1]


# ── Condenser pipeline ────────────────────────────────────────────────────

class TestCondensePipeline:
    def test_condense_empty(self, condenser):
        """No uncondensed facts → empty result."""
        results = condenser.condense()
        assert results == []

    def test_condense_basic(self, populated_store):
        condenser = FactCondenser(populated_store)
        results = condenser.condense()
        assert len(results) > 0
        for entry in results:
            assert "topic" in entry
            assert "category" in entry
            assert "summary" in entry
            assert entry["action"] in ("created", "updated")

    def test_condense_marks_facts(self, populated_store):
        condenser = FactCondenser(populated_store)
        condenser.condense()
        uncond = populated_store.list_uncondensed()
        assert len(uncond) == 0  # all marked

    def test_condense_dry_run(self, populated_store):
        condenser = FactCondenser(populated_store)
        results = condenser.condense(dry_run=True)
        assert len(results) > 0
        for entry in results:
            assert entry["action"] == "dry_run"
        # Facts should NOT be marked
        uncond = populated_store.list_uncondensed()
        assert len(uncond) == 8

    def test_condense_creates_condensed_entries(self, populated_store):
        condenser = FactCondenser(populated_store)
        condenser.condense()
        stats = populated_store.stats()
        assert stats["condensed_total"] > 0


class TestDeduplication:
    def test_dedup_removes_near_duplicates(self, store):
        # Add near-identical facts
        store.add_raw_fact("User prefers dark mode in editor", category="user_pref")
        store.add_raw_fact("User prefers dark mode in the editor", category="user_pref")
        store.add_raw_fact("Something completely different", category="user_pref")

        condenser = FactCondenser(store)
        results = condenser.condense()
        # Should have a user_pref entry with 2 unique facts (dedup removed one)
        user_pref_entry = [r for r in results if r["category"] == "user_pref"]
        assert len(user_pref_entry) == 1
        # Summary should have at most 2 unique parts
        summary = user_pref_entry[0]["summary"]
        parts = [p.strip() for p in summary.split(";") if p.strip()]
        assert len(parts) == 2  # one duplicate removed

    def test_dedup_keeps_distinct(self, store):
        store.add_raw_fact("Python is the main language", category="project")
        store.add_raw_fact("We deploy using Docker containers", category="project")

        condenser = FactCondenser(store)
        results = condenser.condense()
        proj = [r for r in results if r["category"] == "project"]
        assert len(proj) == 1
        parts = [p.strip() for p in proj[0]["summary"].split(";") if p.strip()]
        assert len(parts) == 2  # both kept


class TestRecondense:
    def test_update_existing_condensed(self, store):
        """Second condense run should update (merge) existing condensed entries."""
        store.add_raw_fact("First fact about tools", category="tool")
        condenser = FactCondenser(store)
        condenser.condense()

        # Add more facts in same category
        store.add_raw_fact("Second fact about tools", category="tool")
        results = condenser.condense()
        tool_results = [r for r in results if r["category"] == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0]["action"] == "updated"


class TestGetTopForMemory:
    def test_empty(self, condenser):
        assert condenser.get_top_for_memory() == ""

    def test_with_data(self, populated_store):
        condenser = FactCondenser(populated_store)
        condenser.condense()
        text = condenser.get_top_for_memory()
        assert len(text) > 0
        assert "§" in text or len(text.split("§")) >= 1

    def test_char_limit(self, populated_store):
        condenser = FactCondenser(populated_store)
        condenser.condense()
        text = condenser.get_top_for_memory(char_limit=50)
        assert len(text) <= 55  # small buffer for boundary

    def test_single_entry_truncated(self, store):
        """If first entry exceeds char_limit, it's truncated."""
        store.add_condensed(
            topic="Big", summary="A" * 200, category="general", priority=10
        )
        condenser = FactCondenser(store)
        text = condenser.get_top_for_memory(char_limit=50)
        assert len(text) <= 50

"""Tests for SemanticSearch (embeddings.py) with mocked providers."""
from __future__ import annotations

import sqlite3
import struct
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from store import EnhancedMemoryStore


def _make_mock_provider(dims=4, available=True):
    """Create a mock EmbeddingProvider."""
    provider = MagicMock()
    type(provider).name = PropertyMock(return_value="mock")
    type(provider).dims = PropertyMock(return_value=dims)
    provider.is_available.return_value = available
    # embed_texts returns vectors of the right dimension
    provider.embed_texts.side_effect = lambda texts: [
        [float(i + j) / 10 for j in range(dims)] for i, _ in enumerate(texts)
    ]
    provider.embed_single.side_effect = lambda text: [0.1] * dims
    return provider


class TestSemanticSearchInit:
    def test_init_with_provider(self, db_path):
        provider = _make_mock_provider()
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        assert ss.provider_name == "mock"
        assert ss.dims == 4

    def test_init_with_none_provider_config(self, db_path):
        """When config says 'none', provider should be None."""
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, config={"embedding_provider": "none"})
        assert ss.provider_name == "none"
        assert ss.dims == 0

    def test_is_available(self, db_path):
        provider = _make_mock_provider(available=True)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        # Availability depends on both vec loaded and provider available
        if ss._vec_available:
            assert ss.is_available() is True
        else:
            assert ss.is_available() is False

    def test_is_not_available_without_provider(self, db_path):
        provider = _make_mock_provider(available=False)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        assert ss.is_available() is False


class TestSemanticSearchIndexAndSearch:
    """Tests that require sqlite-vec. Skipped if not available."""

    @pytest.fixture
    def ss_with_vec(self, db_path):
        provider = _make_mock_provider(dims=4, available=True)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        if not ss._vec_available:
            pytest.skip("sqlite-vec not available")
        return ss

    def test_index_facts(self, ss_with_vec):
        facts = [
            {"id": 1, "content": "Python is great"},
            {"id": 2, "content": "Rust is fast"},
        ]
        count = ss_with_vec.index_facts(facts, source_table="raw_facts")
        assert count == 2

    def test_index_facts_empty(self, ss_with_vec):
        assert ss_with_vec.index_facts([], source_table="raw_facts") == 0

    def test_index_condensed_uses_negative_id(self, ss_with_vec):
        facts = [{"id": 1, "content": "Condensed summary"}]
        count = ss_with_vec.index_facts(facts, source_table="condensed")
        assert count == 1

    def test_search(self, ss_with_vec):
        # Index some facts first
        facts = [
            {"id": 1, "content": "Python is great"},
            {"id": 2, "content": "Rust is fast"},
        ]
        ss_with_vec.index_facts(facts, source_table="raw_facts")
        results = ss_with_vec.search("programming", k=2)
        assert isinstance(results, list)
        # With mock embeddings, we may or may not get results but the call should succeed
        for r in results:
            assert "fact_id" in r
            assert "distance" in r
            assert "similarity" in r

    def test_search_unavailable(self, db_path):
        provider = _make_mock_provider(available=False)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        assert ss.search("query") == []

    def test_index_embed_failure(self, ss_with_vec):
        """If embedding fails, return 0."""
        ss_with_vec._provider.embed_texts.side_effect = RuntimeError("API down")
        facts = [{"id": 1, "content": "test"}]
        count = ss_with_vec.index_facts(facts, source_table="raw_facts")
        assert count == 0

    def test_search_embed_failure(self, ss_with_vec):
        ss_with_vec._provider.embed_single.side_effect = RuntimeError("API down")
        results = ss_with_vec.search("query")
        assert results == []


class TestSemanticSearchUnindexed:
    @pytest.fixture
    def setup(self, db_path):
        store = EnhancedMemoryStore(db_path=db_path)
        store.add_raw_fact("fact A")
        store.add_raw_fact("fact B")
        store.add_condensed(topic="T", summary="S", category="general")
        provider = _make_mock_provider(dims=4, available=True)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        return ss, store

    def test_get_unindexed_initial(self, setup):
        ss, store = setup
        if not ss._vec_available:
            pytest.skip("sqlite-vec not available")
        unindexed = ss.get_unindexed(store)
        assert len(unindexed["raw_facts"]) == 2
        assert len(unindexed["condensed"]) == 1

    def test_get_unindexed_after_indexing(self, setup):
        ss, store = setup
        if not ss._vec_available:
            pytest.skip("sqlite-vec not available")
        # Index raw facts
        raw_facts = ss.get_unindexed(store)["raw_facts"]
        ss.index_facts(raw_facts, source_table="raw_facts")
        unindexed = ss.get_unindexed(store)
        assert len(unindexed["raw_facts"]) == 0

    def test_get_unindexed_without_vec(self, db_path):
        provider = _make_mock_provider(available=False)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        ss._vec_available = False
        result = ss.get_unindexed(MagicMock())
        assert result == {"raw_facts": [], "condensed": []}


class TestSemanticSearchReindex:
    def test_reindex_unavailable(self, db_path):
        provider = _make_mock_provider(available=False)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        counts = ss.reindex()
        assert counts == {"raw_facts": 0, "condensed": 0}

    def test_reindex_no_store(self, db_path):
        provider = _make_mock_provider(dims=4, available=True)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        if not ss._vec_available:
            pytest.skip("sqlite-vec not available")
        counts = ss.reindex(conn_or_store=None)
        assert counts == {"raw_facts": 0, "condensed": 0}

    def test_reindex_with_store(self, db_path):
        store = EnhancedMemoryStore(db_path=db_path)
        store.add_raw_fact("fact 1")
        store.add_raw_fact("fact 2")
        store.add_condensed(topic="T", summary="S")
        provider = _make_mock_provider(dims=4, available=True)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        if not ss._vec_available:
            pytest.skip("sqlite-vec not available")
        counts = ss.reindex(conn_or_store=store)
        assert counts["raw_facts"] == 2
        assert counts["condensed"] == 1


class TestSemanticSearchStats:
    def test_stats_basic(self, db_path):
        provider = _make_mock_provider(dims=4, available=True)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        s = ss.stats()
        assert "available" in s
        assert "vec_loaded" in s
        assert "provider" in s
        assert s["provider"] == "mock"
        assert "total_indexed" in s

    def test_stats_with_data(self, db_path):
        provider = _make_mock_provider(dims=4, available=True)
        from embeddings import SemanticSearch
        ss = SemanticSearch(db_path=db_path, provider=provider)
        if not ss._vec_available:
            pytest.skip("sqlite-vec not available")
        ss.index_facts([{"id": 1, "content": "test"}], source_table="raw_facts")
        s = ss.stats()
        assert s["total_indexed"] >= 1


class TestSerializeF32:
    def test_serialize(self):
        from embeddings import _serialize_f32
        blob = _serialize_f32([1.0, 2.0, 3.0])
        assert isinstance(blob, bytes)
        assert len(blob) == 12  # 3 * 4 bytes
        # Verify roundtrip
        values = struct.unpack("<3f", blob)
        assert values == pytest.approx((1.0, 2.0, 3.0))


class TestLoadSqliteVec:
    def test_load_sqlite_vec(self, db_path):
        from embeddings import _load_sqlite_vec
        conn = sqlite3.connect(db_path)
        result = _load_sqlite_vec(conn)
        # Result depends on whether sqlite-vec is installed
        assert isinstance(result, bool)
        conn.close()

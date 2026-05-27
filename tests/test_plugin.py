"""Tests for EnhancedMemoryProvider (__init__.py) — integration-level tests."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# We need to import the plugin as a package.  Since __init__.py lives at the
# plugin root (/root/enhanced-memory-plugin/__init__.py), we import it
# as a top-level module via the sys.path set up in conftest.py.
# But __init__.py uses relative imports (.store, .condenser, .embeddings),
# so we set up the package properly.

import sys
import importlib
import types

# Ensure the plugin directory is importable as a package
_PLUGIN_ROOT = "/root/enhanced-memory-plugin"
_PARENT = os.path.dirname(_PLUGIN_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


def _get_provider_class():
    """Import EnhancedMemoryProvider, handling the package structure."""
    # Import the package
    import importlib
    pkg_name = "enhanced-memory-plugin"
    # Can't use hyphens in package names for normal imports, so we use importlib
    # First, let's just import from the files directly using the already-on-path modules

    # The __init__.py uses relative imports, so we need to set up the package
    # We mock the agent/tools imports since they come from hermes-agent
    spec = importlib.util.spec_from_file_location(
        "enhanced_memory",
        os.path.join(_PLUGIN_ROOT, "__init__.py"),
        submodule_search_locations=[_PLUGIN_ROOT],
    )
    # First ensure sub-modules are loadable
    for mod_name in ("store", "condenser", "embedding_providers", "embeddings"):
        full_name = f"enhanced_memory.{mod_name}"
        if full_name not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(
                full_name,
                os.path.join(_PLUGIN_ROOT, f"{mod_name}.py"),
            )
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sys.modules[full_name] = sub_mod
            sub_spec.loader.exec_module(sub_mod)

    mod = importlib.util.module_from_spec(spec)
    sys.modules["enhanced_memory"] = mod
    spec.loader.exec_module(mod)
    return mod.EnhancedMemoryProvider, mod


_EnhancedMemoryProvider, _plugin_mod = _get_provider_class()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider(tmp_path):
    """Create an EnhancedMemoryProvider with a temp db and semantic search disabled."""
    db = str(tmp_path / "test_mem.db")
    config = {
        "db_path": db,
        "auto_extract": True,
        "auto_condense": False,
        "semantic_search": False,
    }
    p = _EnhancedMemoryProvider(config=config)
    p.initialize(session_id="test-session-001")
    return p


@pytest.fixture
def provider_with_semantic(tmp_path):
    """Provider with semantic search mocked as available."""
    db = str(tmp_path / "test_mem_sem.db")
    config = {
        "db_path": db,
        "auto_extract": False,
        "auto_condense": False,
        "semantic_search": True,
        "embedding_provider": "none",  # avoid real API calls
    }
    p = _EnhancedMemoryProvider(config=config)
    p.initialize(session_id="test-session-002")
    # Mock semantic search
    mock_semantic = MagicMock()
    mock_semantic.is_available.return_value = True
    mock_semantic.search.return_value = []
    mock_semantic.stats.return_value = {"available": True, "total_indexed": 0}
    mock_semantic.index_facts.return_value = 0
    mock_semantic.get_unindexed.return_value = {"raw_facts": [], "condensed": []}
    p._semantic = mock_semantic
    return p


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestProviderBasics:
    def test_name(self, provider):
        assert provider.name == "enhanced-memory"

    def test_is_available(self, provider):
        assert provider.is_available() is True

    def test_get_config_schema(self, provider):
        schema = provider.get_config_schema()
        assert isinstance(schema, list)
        assert len(schema) > 0
        keys = {s["key"] for s in schema}
        assert "db_path" in keys
        assert "auto_extract" in keys

    def test_get_tool_schemas(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "enhanced_memory"


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_initialize_creates_store(self, provider):
        assert provider._store is not None
        assert provider._condenser is not None

    def test_initialize_session_id(self, provider):
        assert provider._session_id == "test-session-001"


# ---------------------------------------------------------------------------
# system_prompt_block
# ---------------------------------------------------------------------------

class TestSystemPromptBlock:
    def test_block_content(self, provider):
        block = provider.system_prompt_block()
        assert "Enhanced Memory" in block
        assert "raw facts" in block

    def test_block_no_store(self):
        p = _EnhancedMemoryProvider(config={})
        # Don't initialize
        assert p.system_prompt_block() == ""


# ---------------------------------------------------------------------------
# prefetch
# ---------------------------------------------------------------------------

class TestPrefetch:
    def test_prefetch_empty(self, provider):
        result = provider.prefetch("test query", session_id="s")
        # No facts yet, so should be empty
        assert result == ""

    def test_prefetch_with_data(self, provider):
        # Add facts first
        provider._store.add_raw_fact("User likes Python programming", category="user_pref")
        result = provider.prefetch("Python", session_id="s")
        # May or may not find it depending on FTS match
        # Just check it doesn't crash
        assert isinstance(result, str)

    def test_prefetch_no_store(self):
        p = _EnhancedMemoryProvider(config={})
        assert p.prefetch("query") == ""

    def test_prefetch_empty_query(self, provider):
        assert provider.prefetch("") == ""


# ---------------------------------------------------------------------------
# handle_tool_call — add
# ---------------------------------------------------------------------------

class TestToolAdd:
    def test_add_fact(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "add", "content": "User prefers Python", "category": "user_pref"},
        ))
        assert result["status"] == "added"
        assert "fact_id" in result

    def test_add_no_content(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "add", "content": ""},
        ))
        assert "error" in result

    def test_add_missing_content(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "add"},
        ))
        assert "error" in result

    def test_add_with_semantic_indexing(self, provider_with_semantic):
        result = json.loads(provider_with_semantic.handle_tool_call(
            "enhanced_memory",
            {"action": "add", "content": "test fact"},
        ))
        assert result["status"] == "added"
        # Semantic index should have been called
        provider_with_semantic._semantic.index_facts.assert_called_once()


# ---------------------------------------------------------------------------
# handle_tool_call — search
# ---------------------------------------------------------------------------

class TestToolSearch:
    def test_search(self, provider):
        provider._store.add_raw_fact("Python is great for data science", category="project")
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "search", "query": "Python"},
        ))
        assert "raw_facts" in result
        assert "condensed" in result
        assert "total" in result

    def test_search_no_query(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "search", "query": ""},
        ))
        assert "error" in result

    def test_search_missing_query(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "search"},
        ))
        assert "error" in result


# ---------------------------------------------------------------------------
# handle_tool_call — semantic_search
# ---------------------------------------------------------------------------

class TestToolSemanticSearch:
    def test_semantic_search_unavailable(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "semantic_search", "query": "test"},
        ))
        assert "error" in result
        assert "fallback" in result

    def test_semantic_search_available(self, provider_with_semantic):
        provider_with_semantic._semantic.search.return_value = [
            {"fact_id": 1, "distance": 0.1, "similarity": 0.9},
        ]
        # Need a raw fact to resolve
        provider_with_semantic._store.add_raw_fact("test fact", category="general")
        result = json.loads(provider_with_semantic.handle_tool_call(
            "enhanced_memory",
            {"action": "semantic_search", "query": "test"},
        ))
        assert "results" in result
        assert "count" in result

    def test_semantic_search_no_query(self, provider_with_semantic):
        result = json.loads(provider_with_semantic.handle_tool_call(
            "enhanced_memory",
            {"action": "semantic_search", "query": ""},
        ))
        assert "error" in result


# ---------------------------------------------------------------------------
# handle_tool_call — condense
# ---------------------------------------------------------------------------

class TestToolCondense:
    def test_condense_empty(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "condense"},
        ))
        assert result["count"] == 0
        assert result["dry_run"] is False

    def test_condense_with_facts(self, provider):
        provider._store.add_raw_fact("User loves vim", category="user_pref")
        provider._store.add_raw_fact("Server runs Linux", category="env")
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "condense"},
        ))
        assert result["count"] >= 1
        for entry in result["entries"]:
            assert "topic" in entry
            assert "category" in entry
            assert "action" in entry

    def test_condense_dry_run(self, provider):
        provider._store.add_raw_fact("test", category="general")
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "condense", "dry_run": True},
        ))
        assert result["dry_run"] is True


# ---------------------------------------------------------------------------
# handle_tool_call — list_condensed
# ---------------------------------------------------------------------------

class TestToolListCondensed:
    def test_list_condensed_empty(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "list_condensed"},
        ))
        assert result["count"] == 0
        assert result["condensed"] == []

    def test_list_condensed_with_data(self, provider):
        provider._store.add_condensed(
            topic="Test", summary="Summary", category="general", priority=5
        )
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "list_condensed"},
        ))
        assert result["count"] >= 1


# ---------------------------------------------------------------------------
# handle_tool_call — stats
# ---------------------------------------------------------------------------

class TestToolStats:
    def test_stats(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "stats"},
        ))
        assert "raw_facts" in result or "raw_total" in str(result)
        assert "condensed" in result
        assert "semantic_search" in result

    def test_stats_with_semantic(self, provider_with_semantic):
        result = json.loads(provider_with_semantic.handle_tool_call(
            "enhanced_memory",
            {"action": "stats"},
        ))
        assert result["semantic_search"]["enabled"] is True


# ---------------------------------------------------------------------------
# handle_tool_call — unknown action and unknown tool
# ---------------------------------------------------------------------------

class TestToolErrors:
    def test_unknown_action(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {"action": "delete_everything"},
        ))
        assert "error" in result

    def test_unknown_tool(self, provider):
        result = json.loads(provider.handle_tool_call(
            "some_other_tool",
            {"action": "add"},
        ))
        assert "error" in result

    def test_missing_action(self, provider):
        result = json.loads(provider.handle_tool_call(
            "enhanced_memory",
            {},
        ))
        assert "error" in result


# ---------------------------------------------------------------------------
# on_memory_write hook
# ---------------------------------------------------------------------------

class TestOnMemoryWrite:
    def test_memory_write_add(self, provider):
        provider.on_memory_write("add", "user", "User prefers dark mode")
        stats = provider._store.stats()
        assert stats["raw_total"] == 1

    def test_memory_write_non_add(self, provider):
        provider.on_memory_write("delete", "user", "something")
        stats = provider._store.stats()
        assert stats["raw_total"] == 0

    def test_memory_write_user_category(self, provider):
        provider.on_memory_write("add", "user", "Prefers Python")
        fact = provider._store.get_raw_by_id(1)
        assert fact["category"] == "user_pref"

    def test_memory_write_general_category(self, provider):
        provider.on_memory_write("add", "other", "Some note")
        fact = provider._store.get_raw_by_id(1)
        assert fact["category"] == "general"

    def test_memory_write_no_store(self):
        p = _EnhancedMemoryProvider(config={})
        # Should not crash even without initialization
        p.on_memory_write("add", "user", "test")


# ---------------------------------------------------------------------------
# Auto-extract patterns
# ---------------------------------------------------------------------------

class TestAutoExtract:
    def test_extract_preference(self, provider):
        messages = [
            {"role": "user", "content": "I prefer using Python for all my projects and scripts"},
        ]
        count = provider._auto_extract_facts(messages)
        assert count >= 1

    def test_extract_decision(self, provider):
        messages = [
            {"role": "user", "content": "We decided to use PostgreSQL for the database backend"},
        ]
        count = provider._auto_extract_facts(messages)
        assert count >= 1

    def test_extract_env(self, provider):
        messages = [
            {"role": "user", "content": "I'm running Python version v3.12 on this machine"},
        ]
        count = provider._auto_extract_facts(messages)
        assert count >= 1

    def test_skip_short_messages(self, provider):
        messages = [
            {"role": "user", "content": "hi"},
        ]
        count = provider._auto_extract_facts(messages)
        assert count == 0

    def test_skip_assistant_messages(self, provider):
        messages = [
            {"role": "assistant", "content": "I prefer using Python for everything"},
        ]
        count = provider._auto_extract_facts(messages)
        assert count == 0

    def test_skip_duplicates(self, provider):
        messages = [
            {"role": "user", "content": "I prefer using Python for all scripting tasks"},
            {"role": "user", "content": "I prefer using Python for all scripting tasks"},
        ]
        count = provider._auto_extract_facts(messages)
        assert count == 1  # second is deduped

    def test_empty_messages(self, provider):
        assert provider._auto_extract_facts([]) == 0


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    def test_sync_turn(self, provider):
        provider.sync_turn("hello", "hi there", session_id="s")
        assert provider._session_turns == 1

    def test_on_session_switch(self, provider):
        provider.on_session_switch("new-session-id", reset=True)
        assert provider._session_id == "new-session-id"
        assert provider._session_turns == 0

    def test_on_session_switch_no_reset(self, provider):
        provider._session_turns = 5
        provider.on_session_switch("new-id", reset=False)
        assert provider._session_id == "new-id"
        assert provider._session_turns == 5

    def test_shutdown(self, provider):
        provider.shutdown()
        assert provider._store is None
        assert provider._condenser is None
        assert provider._semantic is None

    def test_on_session_end(self, provider):
        messages = [
            {"role": "user", "content": "I always use vim keybindings in my editor"},
        ]
        provider.on_session_end(messages)
        # Should have extracted facts
        stats = provider._store.stats()
        assert stats["raw_total"] >= 1

    def test_on_session_end_empty(self, provider):
        provider.on_session_end([])  # should not crash

    def test_on_session_end_no_store(self):
        p = _EnhancedMemoryProvider(config={"auto_extract": True})
        p.on_session_end([{"role": "user", "content": "I prefer Python"}])


# ---------------------------------------------------------------------------
# on_pre_compress
# ---------------------------------------------------------------------------

class TestOnPreCompress:
    def test_pre_compress(self, provider):
        messages = [
            {"role": "user", "content": "I prefer using dark mode in all editors and tools"},
        ]
        result = provider.on_pre_compress(messages)
        assert "Enhanced Memory" in result or result == ""

    def test_pre_compress_no_store(self):
        p = _EnhancedMemoryProvider(config={})
        assert p.on_pre_compress([]) == ""

    def test_pre_compress_empty(self, provider):
        assert provider.on_pre_compress([]) == ""

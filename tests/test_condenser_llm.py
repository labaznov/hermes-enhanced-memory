"""Tests for LLM-based condensation in condenser.py."""

import json
import pytest
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from enhanced_memory.condenser import (
    FactCondenser,
    _LLMClient,
    _extract_json,
    _validate_results,
    CONDENSATION_SYSTEM_PROMPT,
    CONDENSATION_USER_TEMPLATE,
)


class TestExtractJson:
    """Tests for _extract_json helper."""

    def test_direct_json(self):
        result = _extract_json('{"results": []}')
        assert result == {"results": []}

    def test_markdown_fences(self):
        text = '```json\n{"results": [{"a": 1}]}\n```'
        result = _extract_json(text)
        assert result == {"results": [{"a": 1}]}

    def test_markdown_fences_no_lang(self):
        text = '```\n{"results": []}\n```'
        result = _extract_json(text)
        assert result == {"results": []}

    def test_json_embedded_in_text(self):
        text = 'Here is the result:\n{"results": [{"x": 1}]}\nDone.'
        result = _extract_json(text)
        assert result == {"results": [{"x": 1}]}

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_none(self):
        assert _extract_json(None) is None

    def test_invalid_json(self):
        assert _extract_json("not json at all") is None

    def test_real_llm_response(self):
        text = '```json\n{"results": [\n  {"topic": "Server", "category": "security", "summary": "SSH keys.", "priority": 10},\n  {"topic": "User", "category": "user_pref", "summary": "Bilingual.", "priority": 9}\n]}\n```'
        result = _extract_json(text)
        assert result is not None
        assert len(result["results"]) == 2


class TestValidateResults:
    """Tests for _validate_results helper."""

    def test_valid_results(self):
        data = {"results": [
            {"topic": "Test", "category": "security", "summary": "text", "priority": 10}
        ]}
        result = _validate_results(data)
        assert len(result) == 1
        assert result[0]["priority"] == 10

    def test_priority_clamping_up(self):
        """security has lo=9, so priority 5 should be clamped to 9."""
        data = {"results": [
            {"topic": "T", "category": "security", "summary": "t", "priority": 5}
        ]}
        result = _validate_results(data)
        assert result[0]["priority"] == 9

    def test_priority_clamping_down(self):
        data = {"results": [
            {"topic": "T", "category": "general", "summary": "t", "priority": 15}
        ]}
        result = _validate_results(data)
        assert result[0]["priority"] == 10

    def test_missing_category(self):
        data = {"results": [{"topic": "T", "summary": "t", "priority": 5}]}
        assert _validate_results(data) == []

    def test_missing_summary(self):
        data = {"results": [{"topic": "T", "category": "general", "priority": 5}]}
        assert _validate_results(data) == []

    def test_empty_results(self):
        assert _validate_results({"results": []}) == []

    def test_no_results_key(self):
        assert _validate_results({}) == []

    def test_non_dict_items_skipped(self):
        data = {"results": ["not a dict", {"category": "general", "summary": "ok"}]}
        result = _validate_results(data)
        assert len(result) == 1

    def test_default_topic_from_category(self):
        data = {"results": [{"category": "security", "summary": "t", "priority": 10}]}
        result = _validate_results(data)
        assert result[0]["topic"] == "\u0411\u0435\u0437\u043e\u043f\u0430\u0441\u043d\u043e\u0441\u0442\u044c"

    def test_unknown_category_priority(self):
        data = {"results": [{"category": "unknown_cat", "summary": "t", "priority": 3}]}
        result = _validate_results(data)
        assert result[0]["priority"] == 4  # default lo=4


class TestLLMClient:
    """Tests for _LLMClient."""

    def test_no_config_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            client = _LLMClient()
            client._available = None
            assert client.available is False
            assert client.model_name == "none"

    def test_explicit_config(self):
        config = {"model": "test-model", "provider": "openai", "api_key": "key"}
        client = _LLMClient(config)
        assert client.available is True
        assert client.model_name == "test-model"

    def test_google_key_auto_detect(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            client = _LLMClient()
            assert client.available is True
            assert "gemini" in client.model_name

    def test_call_returns_none_when_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            client = _LLMClient()
            client._available = False
            result = client.call("system", "user")
            assert result is None


class TestFactCondenserLLM:
    """Tests for LLM-based condensation in FactCondenser."""

    @pytest.fixture
    def store(self, tmp_path):
        from enhanced_memory.store import EnhancedMemoryStore
        db_path = str(tmp_path / "test.db")
        return EnhancedMemoryStore(db_path=db_path)

    @pytest.fixture
    def condenser_no_llm(self, store):
        c = FactCondenser(store)
        c._llm._available = False
        return c

    def test_condense_uses_algorithmic_when_no_llm(self, store, condenser_no_llm):
        store.add_raw_fact("Test fact 1", category="general")
        store.add_raw_fact("Test fact 2", category="general")
        results = condenser_no_llm.condense(dry_run=True)
        assert len(results) > 0
        assert all(r["method"] == "algorithmic" for r in results)

    def test_condense_with_mock_llm(self, store):
        store.add_raw_fact("User likes Python", category="user_pref")
        store.add_raw_fact("User prefers dark mode", category="user_pref")

        condenser = FactCondenser(store)
        condenser._llm._available = True

        mock_response = json.dumps({"results": [
            {"topic": "User Preferences", "category": "user_pref",
             "summary": "Likes Python, prefers dark mode.", "priority": 9}
        ]})

        with patch.object(condenser._llm, 'call', return_value=mock_response):
            results = condenser.condense(dry_run=True, use_llm=True)

        assert len(results) == 1
        assert results[0]["method"] == "llm"
        assert results[0]["summary"] == "Likes Python, prefers dark mode."
        assert results[0]["priority"] == 9

    def test_condense_llm_failure_falls_back(self, store):
        store.add_raw_fact("Test fact", category="general")
        condenser = FactCondenser(store)
        condenser._llm._available = True

        with patch.object(condenser._llm, 'call', return_value=None):
            results = condenser.condense(dry_run=True, use_llm=True)

        assert len(results) > 0
        assert all(r["method"] == "algorithmic" for r in results)

    def test_condense_llm_invalid_json_falls_back(self, store):
        store.add_raw_fact("Test fact", category="general")
        condenser = FactCondenser(store)
        condenser._llm._available = True

        with patch.object(condenser._llm, 'call', return_value="not json"):
            results = condenser.condense(dry_run=True, use_llm=True)

        assert len(results) > 0
        assert all(r["method"] == "algorithmic" for r in results)

    def test_condense_use_llm_false(self, store):
        store.add_raw_fact("Test fact", category="general")
        condenser = FactCondenser(store)
        results = condenser.condense(dry_run=True, use_llm=False)
        assert len(results) > 0
        assert all(r["method"] == "algorithmic" for r in results)

    def test_condense_llm_replaces_summary(self, store):
        """LLM mode should replace (not append) existing summaries."""
        store.add_raw_fact("Fact A", category="security")
        condenser = FactCondenser(store)
        condenser._llm._available = True

        mock_resp1 = json.dumps({"results": [
            {"topic": "Security", "category": "security",
             "summary": "First summary.", "priority": 10}
        ]})
        with patch.object(condenser._llm, 'call', return_value=mock_resp1):
            condenser.condense(dry_run=False, use_llm=True)

        store.add_raw_fact("Fact B", category="security")

        mock_resp2 = json.dumps({"results": [
            {"topic": "Security Updated", "category": "security",
             "summary": "Updated summary with A and B.", "priority": 10}
        ]})
        with patch.object(condenser._llm, 'call', return_value=mock_resp2):
            condenser.condense(dry_run=False, use_llm=True)

        conn = store.get_connection()
        row = conn.execute(
            "SELECT summary FROM condensed WHERE category = 'security'"
        ).fetchone()
        assert row is not None
        assert row[0] == "Updated summary with A and B."
        assert "First summary" not in row[0]

    def test_condense_passes_existing_summary_to_llm(self, store):
        conn = store.get_connection()
        conn.execute(
            "INSERT INTO condensed (topic, category, summary, priority, "
            "source_ids, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 0, 0)",
            ("Test", "user_pref", "Existing user info.", 9, "[]"),
        )
        conn.commit()

        store.add_raw_fact("New preference", category="user_pref")
        condenser = FactCondenser(store)
        condenser._llm._available = True

        captured_args = {}

        def capture_call(system_prompt, user_message):
            captured_args["user_message"] = user_message
            return json.dumps({"results": [
                {"topic": "Prefs", "category": "user_pref",
                 "summary": "Updated prefs.", "priority": 9}
            ]})

        with patch.object(condenser._llm, 'call', side_effect=capture_call):
            condenser.condense(dry_run=True, use_llm=True)

        assert "Existing user info." in captured_args["user_message"]


class TestPromptTemplates:
    """Tests for prompt constants."""

    def test_system_prompt_not_empty(self):
        assert len(CONDENSATION_SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_categories(self):
        for cat in ["security", "user_pref", "decision", "project", "tool", "env", "general"]:
            assert cat in CONDENSATION_SYSTEM_PROMPT

    def test_user_template_has_placeholder(self):
        assert "{categories_json}" in CONDENSATION_USER_TEMPLATE

    def test_user_template_renders(self):
        rendered = CONDENSATION_USER_TEMPLATE.format(categories_json='{"test": "data"}')
        assert '"test": "data"' in rendered

    def test_system_prompt_contains_json_format(self):
        assert '"results"' in CONDENSATION_SYSTEM_PROMPT

    def test_system_prompt_bilingual(self):
        assert "Bilingual" in CONDENSATION_SYSTEM_PROMPT

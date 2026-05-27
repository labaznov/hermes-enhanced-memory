"""Tests for embedding_providers.py — provider ABC, factory, graceful degradation."""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from embedding_providers import (
    EmbeddingProvider,
    GeminiEmbedding,
    OpenAIEmbedding,
    LocalEmbedding,
    create_embedding_provider,
    PROVIDER_DEFAULTS,
)


# ── Abstract base class ───────────────────────────────────────────────────

class TestEmbeddingProviderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_concrete_must_implement(self):
        class Broken(EmbeddingProvider):
            pass
        with pytest.raises(TypeError):
            Broken()


# ── GeminiEmbedding ──────────────────────────────────────────────────────

class TestGeminiEmbedding:
    def test_properties(self):
        g = GeminiEmbedding(api_key="fake-key")
        assert g.name == "gemini"
        assert g.dims == 3072

    def test_custom_dims(self):
        g = GeminiEmbedding(dimensions=768, api_key="fake-key")
        assert g.dims == 768

    def test_is_available_with_key(self):
        g = GeminiEmbedding(api_key="fake-key")
        assert g.is_available() is True

    def test_is_available_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(GeminiEmbedding, "_resolve_key", return_value=None):
                g = GeminiEmbedding(api_key=None)
                g._api_key = None  # ensure
                assert g.is_available() is False

    def test_embed_texts_empty(self):
        g = GeminiEmbedding(api_key="fake-key")
        assert g.embed_texts([]) == []

    def test_embed_texts_no_key_raises(self):
        g = GeminiEmbedding(api_key=None)
        g._api_key = None
        with pytest.raises(RuntimeError, match="No Gemini API key"):
            g.embed_texts(["hello"])

    @patch.object(GeminiEmbedding, "_api_request")
    def test_embed_single(self, mock_api):
        mock_api.return_value = {"embedding": {"values": [0.1, 0.2, 0.3]}}
        g = GeminiEmbedding(api_key="fake-key")
        result = g.embed_texts(["test"])
        assert result == [[0.1, 0.2, 0.3]]
        mock_api.assert_called_once()

    @patch.object(GeminiEmbedding, "_api_request")
    def test_embed_batch(self, mock_api):
        mock_api.return_value = {
            "embeddings": [
                {"values": [0.1, 0.2]},
                {"values": [0.3, 0.4]},
            ]
        }
        g = GeminiEmbedding(api_key="fake-key")
        result = g.embed_texts(["a", "b"])
        assert len(result) == 2

    def test_embed_single_convenience(self):
        g = GeminiEmbedding(api_key="fake-key")
        with patch.object(g, "embed_texts", return_value=[[1.0, 2.0]]):
            result = g.embed_single("test")
            assert result == [1.0, 2.0]

    def test_resolve_key_from_env(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            key = GeminiEmbedding._resolve_key()
            assert key == "env-key"

    def test_resolve_key_gemini_env(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gem-key"}, clear=True):
            # remove GOOGLE_API_KEY if present
            os.environ.pop("GOOGLE_API_KEY", None)
            key = GeminiEmbedding._resolve_key()
            assert key == "gem-key"

    def test_resolve_key_from_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("GOOGLE_API_KEY=file-key\n")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            os.environ.pop("GEMINI_API_KEY", None)
            key = GeminiEmbedding._resolve_key()
            assert key == "file-key"

    def test_resolve_key_none(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            os.environ.pop("GEMINI_API_KEY", None)
            key = GeminiEmbedding._resolve_key()
            assert key is None


# ── OpenAIEmbedding ──────────────────────────────────────────────────────

class TestOpenAIEmbedding:
    def test_properties(self):
        o = OpenAIEmbedding(api_key="fake")
        assert o.name == "openai"
        assert o.dims == 1536

    def test_is_available_with_key(self):
        o = OpenAIEmbedding(api_key="fake")
        assert o.is_available() is True

    def test_is_available_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            o = OpenAIEmbedding(api_key=None)
            assert o.is_available() is False

    def test_embed_texts_empty(self):
        o = OpenAIEmbedding(api_key="fake")
        assert o.embed_texts([]) == []

    def test_embed_texts_no_key_raises(self):
        o = OpenAIEmbedding(api_key=None)
        o._api_key = None
        with pytest.raises(RuntimeError, match="No OpenAI API key"):
            o.embed_texts(["hello"])


# ── LocalEmbedding ────────────────────────────────────────────────────────

class TestLocalEmbedding:
    def test_properties(self):
        l = LocalEmbedding()
        assert l.name == "local"
        assert l.dims == 384  # default for all-MiniLM-L6-v2

    def test_custom_dims(self):
        l = LocalEmbedding(dimensions=768)
        assert l.dims == 768

    def test_known_model_dims(self):
        l = LocalEmbedding(model="all-mpnet-base-v2")
        assert l.dims == 768

    def test_unknown_model_default_dims(self):
        l = LocalEmbedding(model="some-unknown-model")
        assert l.dims == 384

    def test_is_available_with_mock(self):
        # Simulate sentence_transformers being importable
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            l = LocalEmbedding()
            assert l.is_available() is True

    def test_embed_texts_empty(self):
        l = LocalEmbedding()
        assert l.embed_texts([]) == []

    def test_actual_dims_override(self):
        l = LocalEmbedding()
        l._actual_dims = 512
        assert l.dims == 512


# ── Factory ───────────────────────────────────────────────────────────────

class TestFactory:
    def test_create_gemini(self):
        p = create_embedding_provider({"embedding_provider": "gemini", "embedding_api_key": "k"})
        assert isinstance(p, GeminiEmbedding)

    def test_create_openai(self):
        p = create_embedding_provider({"embedding_provider": "openai", "embedding_api_key": "k"})
        assert isinstance(p, OpenAIEmbedding)

    def test_create_openai_large(self):
        p = create_embedding_provider({"embedding_provider": "openai-large", "embedding_api_key": "k"})
        assert isinstance(p, OpenAIEmbedding)

    def test_create_local(self):
        p = create_embedding_provider({"embedding_provider": "local"})
        assert isinstance(p, LocalEmbedding)

    def test_create_local_multilingual(self):
        p = create_embedding_provider({"embedding_provider": "local-multilingual"})
        assert isinstance(p, LocalEmbedding)

    def test_create_none(self):
        p = create_embedding_provider({"embedding_provider": "none"})
        assert p is None

    def test_create_disabled(self):
        p = create_embedding_provider({"embedding_provider": "disabled"})
        assert p is None

    def test_create_unknown_falls_back_to_local(self):
        p = create_embedding_provider({"embedding_provider": "mystery"})
        assert isinstance(p, LocalEmbedding)

    def test_custom_model_and_dims(self):
        p = create_embedding_provider({
            "embedding_provider": "gemini",
            "embedding_model": "custom-model",
            "embedding_dims": 512,
            "embedding_api_key": "k",
        })
        assert isinstance(p, GeminiEmbedding)
        assert p.dims == 512

    def test_provider_defaults_known(self):
        assert "gemini" in PROVIDER_DEFAULTS
        assert "openai" in PROVIDER_DEFAULTS
        assert "local" in PROVIDER_DEFAULTS

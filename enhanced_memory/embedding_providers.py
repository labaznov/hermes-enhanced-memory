"""Embedding providers for the enhanced-memory plugin.

Supports multiple backends for generating text embeddings used by the
:class:`~embeddings.SemanticSearch` module for KNN vector search:

- Gemini API (Google, default, 3072-dim)
- OpenAI API (1536 or 3072-dim)
- Local sentence-transformers (configurable model and dimensions)

All providers implement the :class:`EmbeddingProvider` ABC so they can be
used interchangeably.  The :func:`create_embedding_provider` factory reads
the plugin configuration and returns the appropriate instance.

Configuration (``config.yaml``)::

  plugins:
    enhanced-memory:
      embedding_provider: gemini        # or "openai", "local", "none"
      embedding_model: gemini-embedding-001
      embedding_dims: 3072

For the local provider, install::

    pip install sentence-transformers
"""

from __future__ import annotations

import json
import logging
import os
import struct
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All concrete providers must implement :attr:`name`, :attr:`dims`,
    :meth:`is_available`, and :meth:`embed_texts`.  The convenience
    method :meth:`embed_single` is provided for one-off embeddings.

    Subclasses:
        - :class:`GeminiEmbedding`
        - :class:`OpenAIEmbedding`
        - :class:`LocalEmbedding`
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider name used in logging and configuration.

        Returns:
            str: e.g. ``'gemini'``, ``'openai'``, ``'local'``.
        """

    @property
    @abstractmethod
    def dims(self) -> int:
        """Dimensionality of the embedding vectors produced.

        Returns:
            int: Number of floats per embedding vector.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this provider is ready to generate embeddings.

        Returns:
            bool: ``True`` if all prerequisites are met (API key set,
            library installed, etc.).
        """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: Strings to embed.  An empty list should return ``[]``.

        Returns:
            list[list[float]]: One embedding vector per input text,
            in the same order.

        Raises:
            RuntimeError: If the provider is not available or the API
                call fails.
        """

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string.

        Delegates to :meth:`embed_texts` with a one-element list.

        Args:
            text: The string to embed.

        Returns:
            list[float]: The embedding vector.
        """
        return self.embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class GeminiEmbedding(EmbeddingProvider):
    """Google Gemini Embedding API provider.

    Uses the ``generativelanguage.googleapis.com`` REST API directly via
    ``urllib.request`` — no third-party SDK required.  Supports both
    single and batch embedding endpoints.

    Attributes:
        BATCH_SIZE (int): Maximum texts per ``batchEmbedContents`` call (50).
        BATCH_DELAY (float): Seconds to sleep between batches to avoid
            rate-limiting.

    Args:
        model: Gemini embedding model name.
        dimensions: Output embedding dimensionality.
        api_key: Explicit API key.  If ``None``, resolved from environment
            variables or ``$HERMES_HOME/.env``.
    """

    _API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
    # Google recommends ≤100 per batch, but 50 is safer for rate limits.
    BATCH_SIZE = 50
    # Small delay between batches to stay under quota.
    BATCH_DELAY = 0.2

    def __init__(self, model: str = "gemini-embedding-001",
                 dimensions: int = 3072, api_key: str | None = None):
        self._model = model
        self._dims = dimensions
        self._api_key = api_key or self._resolve_key()
        # Pre-build endpoint URLs for single and batch calls.
        self._embed_url = f"{self._API_BASE}/{self._model}:embedContent"
        self._batch_url = f"{self._API_BASE}/{self._model}:batchEmbedContents"

    @property
    def name(self) -> str:
        """Return ``'gemini'``."""
        return "gemini"

    @property
    def dims(self) -> int:
        """Return the configured embedding dimensions."""
        return self._dims

    def is_available(self) -> bool:
        """``True`` if an API key has been resolved."""
        return self._api_key is not None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via the Gemini API.

        Automatically chooses the single-text or batch endpoint based on
        the input size.  For large inputs, texts are chunked into batches
        of :attr:`BATCH_SIZE` with a :attr:`BATCH_DELAY` pause between
        them to respect rate limits.

        Args:
            texts: Strings to embed.

        Returns:
            list[list[float]]: Embeddings in the same order as *texts*.

        Raises:
            RuntimeError: If no API key is configured.
            urllib.error.HTTPError: On API errors.
        """
        if not texts:
            return []
        if not self._api_key:
            raise RuntimeError("No Gemini API key configured")
        if len(texts) == 1:
            return [self._embed_single(texts[0])]
        return self._embed_batch(texts)

    def _embed_single(self, text: str) -> list[float]:
        """Call the single-text ``embedContent`` endpoint.

        Args:
            text: The text to embed.

        Returns:
            list[float]: The embedding vector.
        """
        payload = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text}]},
        }
        resp = self._api_request(self._embed_url, payload)
        return resp["embedding"]["values"]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the ``batchEmbedContents`` endpoint in chunks.

        Args:
            texts: Texts to embed (may exceed :attr:`BATCH_SIZE`).

        Returns:
            list[list[float]]: All embeddings concatenated in order.
        """
        all_embeddings: list[list[float]] = []
        model_name = f"models/{self._model}"

        for batch_start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[batch_start:batch_start + self.BATCH_SIZE]
            requests_payload = [
                {"model": model_name, "content": {"parts": [{"text": t}]}}
                for t in batch
            ]
            resp = self._api_request(self._batch_url, {"requests": requests_payload})
            for emb_obj in resp["embeddings"]:
                all_embeddings.append(emb_obj["values"])

            # Sleep between batches only if there are more to process.
            if batch_start + self.BATCH_SIZE < len(texts):
                time.sleep(self.BATCH_DELAY)

        return all_embeddings

    def _api_request(self, url: str, payload: dict) -> Any:
        """Send a JSON POST request to the Gemini API.

        Args:
            url: The endpoint URL (without the ``key=`` query parameter).
            payload: JSON-serialisable request body.

        Returns:
            Any: Parsed JSON response.

        Raises:
            urllib.error.HTTPError: On HTTP error responses.
            urllib.error.URLError: On network errors.
        """
        # Append the API key as a query parameter (Gemini's auth mechanism).
        full_url = f"{url}?key={self._api_key}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            full_url, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode() if exc.fp else ""
            logger.error("Gemini API error %s: %s", exc.code, body[:200])
            raise
        except urllib.error.URLError as exc:
            logger.error("Gemini API network error: %s", exc.reason)
            raise

    @staticmethod
    def _resolve_key() -> str | None:
        """Attempt to find a Gemini/Google API key from the environment.

        Resolution order:
            1. ``$GOOGLE_API_KEY`` environment variable.
            2. ``$GEMINI_API_KEY`` environment variable.
            3. ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` in
               ``$HERMES_HOME/.env`` file.

        Returns:
            str | None: The API key, or ``None`` if not found.
        """
        # Check environment variables first.
        for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
            val = os.environ.get(var)
            if val:
                return val
        # Fall back to parsing the .env file in the Hermes home directory.
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        env_file = Path(hermes_home) / ".env"
        if env_file.is_file():
            try:
                for line in env_file.read_text().splitlines():
                    line = line.strip()  # noqa: PLW2901
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        # Simple KEY=VALUE parsing (handles surrounding quotes).
                        key, _, value = line.partition("=")
                        key, value = key.strip(), value.strip().strip("'\"")
                        if key in ("GOOGLE_API_KEY", "GEMINI_API_KEY") and value:
                            return value
            except OSError:
                pass
        return None


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI Embedding API provider (text-embedding-3-small/large/ada-002).

    Uses the OpenAI REST API directly via ``urllib.request`` — no SDK needed.
    Supports custom ``base_url`` for OpenAI-compatible third-party endpoints
    (e.g. Azure OpenAI, Ollama, LM Studio).

    Attributes:
        BATCH_SIZE (int): Maximum texts per API call (100).

    Args:
        model: OpenAI embedding model name.  The ``dimensions`` parameter
            is only sent for ``text-embedding-3-*`` models.
        dimensions: Output embedding dimensionality.
        api_key: Explicit API key.  Falls back to ``$OPENAI_API_KEY``.
        base_url: API base URL.  Falls back to ``$OPENAI_BASE_URL`` or
            ``https://api.openai.com/v1``.

    Example::

        provider = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1536)
        vectors = provider.embed_texts(["hello", "world"])
    """

    BATCH_SIZE = 100

    def __init__(self, model: str = "text-embedding-3-small",
                 dimensions: int = 1536, api_key: str | None = None,
                 base_url: str | None = None):
        """Initialise the OpenAI embedding provider.

        Args:
            model: Model identifier (e.g. ``'text-embedding-3-small'``).
            dimensions: Desired output dimensions.
            api_key: Explicit key or ``None`` to read ``$OPENAI_API_KEY``.
            base_url: API base URL or ``None`` for the official endpoint.
        """
        self._model = model
        self._dims = dimensions
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # Resolve base URL: explicit arg > env var > default OpenAI endpoint.
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")
                          or "https://api.openai.com/v1")

    @property
    def name(self) -> str:
        """Return ``'openai'``."""
        return "openai"

    @property
    def dims(self) -> int:
        """Return the configured embedding dimensions."""
        return self._dims

    def is_available(self) -> bool:
        """``True`` if an API key has been resolved."""
        return self._api_key is not None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via the OpenAI API.

        Texts are chunked into batches of :attr:`BATCH_SIZE`.  The response
        items are sorted by their ``index`` field to guarantee order
        consistency even if the API returns them out of order.

        Args:
            texts: Strings to embed.

        Returns:
            list[list[float]]: Embeddings in the same order as *texts*.

        Raises:
            RuntimeError: If no API key is configured.
            urllib.error.HTTPError: On API errors.
        """
        if not texts:
            return []
        if not self._api_key:
            raise RuntimeError("No OpenAI API key configured")

        all_embeddings: list[list[float]] = []
        for batch_start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[batch_start:batch_start + self.BATCH_SIZE]
            payload: dict[str, Any] = {
                "input": batch,
                "model": self._model,
            }
            # Only text-embedding-3-* models accept the 'dimensions' parameter;
            # older models (ada-002) would reject it.
            if self._model.startswith("text-embedding-3"):
                payload["dimensions"] = self._dims

            url = f"{self._base_url}/embeddings"
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode())
                # Sort by index to ensure order matches the input list.
                for item in sorted(result["data"], key=lambda x: x["index"]):
                    all_embeddings.append(item["embedding"])
            except urllib.error.HTTPError as exc:
                body = exc.read().decode() if exc.fp else ""
                logger.error("OpenAI API error %s: %s", exc.code, body[:200])
                raise
            except urllib.error.URLError as exc:
                logger.error("OpenAI API network error: %s", exc.reason)
                raise

        return all_embeddings


# ---------------------------------------------------------------------------
# Local sentence-transformers provider
# ---------------------------------------------------------------------------

class LocalEmbedding(EmbeddingProvider):
    """Local embedding via ``sentence-transformers`` (no API calls required).

    Models are loaded lazily on first use and cached for the lifetime of the
    instance.  The actual embedding dimensionality is detected at load time
    by encoding a test string; until then, the configured or well-known
    default is reported by :attr:`dims`.

    Install the dependency with::

        pip install sentence-transformers

    Attributes:
        _model_name (str): Hugging Face model identifier.
        _device (str): Torch device string (``'cpu'``, ``'cuda'``, ``'mps'``).
        _model: Lazily-loaded ``SentenceTransformer`` instance.
        _actual_dims (int | None): Detected dimensions after first encode.

    Args:
        model: Hugging Face model name or path.
        dimensions: Override for the output dimensionality.  ``None``
            means auto-detect from well-known defaults or the model itself.
        device: Torch device to use for inference.

    Example::

        provider = LocalEmbedding(model="all-MiniLM-L6-v2")
        if provider.is_available():
            vectors = provider.embed_texts(["hello", "world"])
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2",
                 dimensions: int | None = None, device: str = "cpu"):
        """Initialise the local embedding provider (model loaded lazily).

        Args:
            model: Hugging Face model identifier.
            dimensions: Explicit dimension override, or ``None`` to auto-detect.
            device: Torch device (``'cpu'``, ``'cuda'``, ``'mps'``).
        """
        self._model_name = model
        self._device = device
        self._model = None  # loaded lazily by _ensure_model()
        self._dims_override = dimensions
        self._actual_dims: int | None = None

    @property
    def name(self) -> str:
        """Return ``'local'``."""
        return "local"

    @property
    def dims(self) -> int:
        """Return the embedding dimensionality.

        Resolution order:
            1. Actual dimensions detected after loading the model.
            2. Explicit ``dimensions`` override from the constructor.
            3. Well-known defaults for popular model names.
            4. Fallback: ``384``.

        Returns:
            int: Number of floats per embedding vector.
        """
        if self._actual_dims:
            return self._actual_dims
        if self._dims_override:
            return self._dims_override
        # Well-known dimensionality defaults for popular sentence-transformer models.
        defaults = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "nomic-embed-text-v1": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "intfloat/e5-small-v2": 384,
            "intfloat/e5-base-v2": 768,
            "intfloat/e5-large-v2": 1024,
            "intfloat/multilingual-e5-large": 1024,
        }
        return defaults.get(self._model_name, 384)

    def is_available(self) -> bool:
        """``True`` if the ``sentence_transformers`` package is importable.

        Returns:
            bool: Whether the required library is installed.
        """
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_model(self) -> None:
        """Load the ``SentenceTransformer`` model if not already loaded.

        On first call, imports the library, instantiates the model on the
        configured device, and runs a test encode to detect the actual
        output dimensionality.

        Raises:
            Exception: Propagated from ``SentenceTransformer`` if the model
                cannot be loaded (e.g. model not found, CUDA unavailable).
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._model_name, device=self._device
                )
                # Encode a test string to detect the actual output dimensions.
                test = self._model.encode(["test"])
                self._actual_dims = len(test[0])
                logger.info(
                    "Loaded local embedding model %s (%d dims, device=%s)",
                    self._model_name, self._actual_dims, self._device,
                )
            except Exception as exc:
                logger.error("Failed to load local model %s: %s", self._model_name, exc)
                raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using the local sentence-transformer model.

        The model is loaded lazily on first call via :meth:`_ensure_model`.

        Args:
            texts: Strings to embed.

        Returns:
            list[list[float]]: Embeddings in the same order as *texts*.

        Raises:
            Exception: If the model cannot be loaded or encoding fails.
        """
        if not texts:
            return []
        self._ensure_model()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Known provider configs with sensible defaults.
# Each entry maps a short provider name to its class, default model, and
# default embedding dimensions.  Used by :func:`create_embedding_provider`.
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "gemini": {"class": GeminiEmbedding, "model": "gemini-embedding-001", "dims": 3072},
    "openai": {"class": OpenAIEmbedding, "model": "text-embedding-3-small", "dims": 1536},
    "openai-large": {"class": OpenAIEmbedding, "model": "text-embedding-3-large", "dims": 3072},
    "local": {"class": LocalEmbedding, "model": "all-MiniLM-L6-v2", "dims": 384},
    "local-multilingual": {"class": LocalEmbedding, "model": "intfloat/multilingual-e5-large", "dims": 1024},
}


def create_embedding_provider(config: dict[str, Any]) -> EmbeddingProvider | None:
    """Factory function: create an embedding provider from plugin configuration.

    Reads the ``embedding_provider`` key from *config* and instantiates the
    corresponding :class:`EmbeddingProvider` subclass with overrides from
    ``embedding_model``, ``embedding_dims``, ``embedding_api_key``, etc.

    Supported provider names:
        - ``'gemini'`` → :class:`GeminiEmbedding`
        - ``'openai'`` / ``'openai-large'`` → :class:`OpenAIEmbedding`
        - ``'local'`` / ``'local-multilingual'`` → :class:`LocalEmbedding`
        - ``'none'`` / ``'disabled'`` → returns ``None``

    Args:
        config: Plugin configuration dict.  Recognised keys:

            - ``embedding_provider``: Provider name (default ``'gemini'``).
            - ``embedding_model``: Model name override.
            - ``embedding_dims``: Dimension override (cast to ``int``).
            - ``embedding_api_key``: Explicit API key.
            - ``embedding_base_url``: For OpenAI-compatible endpoints.
            - ``embedding_device``: For local models (``'cpu'``, ``'cuda'``,
              ``'mps'``).

    Returns:
        EmbeddingProvider | None: The configured provider, or ``None`` if
        the provider is ``'none'`` / ``'disabled'`` or unavailable.

    Example::

        provider = create_embedding_provider({"embedding_provider": "openai"})
    """
    provider_name = config.get("embedding_provider", "gemini").lower().strip()

    if provider_name == "none" or provider_name == "disabled":
        logger.info("Semantic search disabled by config")
        return None

    defaults = PROVIDER_DEFAULTS.get(provider_name, {})
    provider_class = defaults.get("class")
    default_model = defaults.get("model", "")
    default_dims = defaults.get("dims", 384)

    model = config.get("embedding_model", default_model)
    dims = int(config.get("embedding_dims", default_dims))
    api_key = config.get("embedding_api_key")

    if provider_name in ("gemini",):
        # Gemini provider — API key resolved internally from env / .env file.
        return GeminiEmbedding(model=model, dimensions=dims, api_key=api_key)

    elif provider_name in ("openai", "openai-large"):
        # OpenAI provider — supports custom base_url for compatible APIs.
        base_url = config.get("embedding_base_url")
        return OpenAIEmbedding(
            model=model, dimensions=dims,
            api_key=api_key, base_url=base_url,
        )

    elif provider_name in ("local", "local-multilingual"):
        # Local sentence-transformers — no API key needed, runs on device.
        device = config.get("embedding_device", "cpu")
        return LocalEmbedding(model=model, dimensions=dims, device=device)

    else:
        # Unknown provider name: attempt to treat it as a local model name
        # so users can specify e.g. "BAAI/bge-small-en-v1.5" directly.
        logger.warning("Unknown embedding provider '%s', trying as local", provider_name)
        device = config.get("embedding_device", "cpu")
        return LocalEmbedding(model=provider_name, dimensions=dims, device=device)

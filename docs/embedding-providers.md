# Embedding Providers

Enhanced Memory supports **three embedding backends** for semantic vector search, plus a "none" option to disable embeddings entirely. All providers implement the same `EmbeddingProvider` interface and can be swapped by changing a single config line.

---

## Provider Overview {#overview}

| Provider | Key | Default Model | Dimensions | Requires | Best For |
|----------|-----|---------------|-----------|----------|----------|
| **Gemini** | `gemini` | `gemini-embedding-001` | 3072 | `GOOGLE_API_KEY` | Best quality, free tier available |
| **OpenAI** | `openai` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` | Good quality, widely used |
| **OpenAI Large** | `openai-large` | `text-embedding-3-large` | 3072 | `OPENAI_API_KEY` | Highest OpenAI quality |
| **Local** | `local` | `all-MiniLM-L6-v2` | 384 | `sentence-transformers` | Fully offline, fast |
| **Local Multilingual** | `local-multilingual` | `intfloat/multilingual-e5-large` | 1024 | `sentence-transformers` | Multilingual, offline |
| **Disabled** | `none` | — | — | — | FTS5 search only |

---

## Gemini (Default) {#gemini}

Google's Gemini embedding model offers high-quality embeddings with a generous free tier.

### Setup

1. Get a Google API key from [Google AI Studio](https://aistudio.google.com/apikey)

2. Set the environment variable:

    ```bash
    export GOOGLE_API_KEY="AIzaSy..."
    ```

    Or add to `$HERMES_HOME/.env`:

    ```
    GOOGLE_API_KEY=AIzaSy...
    ```

3. Configure the plugin:

    ```yaml
    plugins:
      enhanced-memory:
        embedding_provider: gemini
    ```

### Details

- **API:** `generativelanguage.googleapis.com/v1beta` (REST, no SDK needed)
- **Model:** `gemini-embedding-001`
- **Dimensions:** 3072
- **Batch size:** 50 texts per request
- **Rate limiting:** 0.2s delay between batches
- **Auth:** API key as query parameter
- **Cost:** Free tier includes substantial usage; paid tier for high volume

### Key Resolution

The provider looks for API keys in this order:

1. `embedding_api_key` in config.yaml
2. `$GOOGLE_API_KEY` environment variable
3. `$GEMINI_API_KEY` environment variable
4. `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `$HERMES_HOME/.env`

### Custom Model

```yaml
plugins:
  enhanced-memory:
    embedding_provider: gemini
    embedding_model: gemini-embedding-001
    embedding_dims: 3072
```

---

## OpenAI {#openai}

OpenAI's text embedding models, available in standard and large variants.

### Setup

1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)

2. Set the environment variable:

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

3. Configure the plugin:

    ```yaml
    plugins:
      enhanced-memory:
        embedding_provider: openai       # standard (1536 dims)
        # or
        embedding_provider: openai-large  # large (3072 dims)
    ```

### Available Models

| Config Key | Model | Dimensions | Quality | Cost |
|-----------|-------|-----------|---------|------|
| `openai` | `text-embedding-3-small` | 1536 | Good | Low |
| `openai-large` | `text-embedding-3-large` | 3072 | Best | Higher |

### Details

- **API:** `api.openai.com/v1/embeddings` (REST, no SDK needed)
- **Batch size:** 100 texts per request
- **Auth:** Bearer token in Authorization header
- **Supports dimension override** for `text-embedding-3-*` models

### Custom API Endpoint

For Azure OpenAI, local OpenAI-compatible servers, or other proxies:

```yaml
plugins:
  enhanced-memory:
    embedding_provider: openai
    embedding_base_url: https://your-resource.openai.azure.com/openai/deployments/embedding
    embedding_api_key: your-azure-key
```

You can also set `OPENAI_BASE_URL` as an environment variable.

### Using with LM Studio, Ollama, or vLLM

Any server exposing an OpenAI-compatible `/v1/embeddings` endpoint can be used:

```yaml
plugins:
  enhanced-memory:
    embedding_provider: openai
    embedding_model: nomic-embed-text
    embedding_dims: 768
    embedding_base_url: http://localhost:1234/v1
    embedding_api_key: not-needed
```

---

## Local (Sentence-Transformers) {#local}

Run embeddings entirely locally with no API calls. Uses the [sentence-transformers](https://www.sbert.net/) library.

### Setup

1. Install sentence-transformers:

    ```bash
    pip install sentence-transformers
    ```

2. Configure the plugin:

    ```yaml
    plugins:
      enhanced-memory:
        embedding_provider: local
        embedding_device: cpu    # or cuda, mps
    ```

### Available Presets

| Config Key | Model | Dimensions | Size | Speed |
|-----------|-------|-----------|------|-------|
| `local` | `all-MiniLM-L6-v2` | 384 | 80MB | Fast |
| `local-multilingual` | `intfloat/multilingual-e5-large` | 1024 | 2.2GB | Slower |

### Supported Models

Any model compatible with sentence-transformers can be used. Popular options:

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `all-MiniLM-L6-v2` | 384 | Default. Fast, good English quality |
| `all-mpnet-base-v2` | 768 | Better quality, slower |
| `BAAI/bge-small-en-v1.5` | 384 | Strong performance for size |
| `BAAI/bge-base-en-v1.5` | 768 | Excellent English embeddings |
| `BAAI/bge-large-en-v1.5` | 1024 | Top-tier English |
| `intfloat/e5-small-v2` | 384 | Good with instruction prefixes |
| `intfloat/e5-large-v2` | 1024 | High quality |
| `intfloat/multilingual-e5-large` | 1024 | 100+ languages |
| `nomic-ai/nomic-embed-text-v1` | 768 | Long context (8192 tokens) |

### Custom Model

```yaml
plugins:
  enhanced-memory:
    embedding_provider: local
    embedding_model: BAAI/bge-large-en-v1.5
    embedding_dims: 1024
    embedding_device: cuda
```

### Device Options

| Device | When to Use |
|--------|-------------|
| `cpu` | Default. Works everywhere, slower for large batches |
| `cuda` | NVIDIA GPU available. Much faster for batch embedding |
| `mps` | Apple Silicon (M1/M2/M3). Metal acceleration |

### First Run

On first use, the model will be downloaded from HuggingFace Hub. This happens once and is cached locally. Download sizes range from ~80MB (MiniLM) to ~2.2GB (multilingual-e5-large).

---

## Disabling Semantic Search {#disable}

To use only FTS5 keyword search with no embeddings:

```yaml
plugins:
  enhanced-memory:
    semantic_search: false
    embedding_provider: none
```

Or simply set `embedding_provider: none`. The plugin works perfectly fine with keyword search only — semantic search is an optional enhancement.

---

## Switching Providers {#switching}

You can switch providers at any time by changing the config. However, note that:

1. **Existing vectors become unusable** when switching to a provider with different dimensions
2. The vector table will be recreated with the new dimensions
3. Facts need to be **re-indexed** to generate new embeddings
4. Raw facts and condensed entries in SQLite are unaffected

To re-index after switching:

1. Change the `embedding_provider` in config
2. Restart Hermes Agent
3. The plugin will detect the dimension mismatch and recreate the vector table
4. New facts will be indexed automatically
5. For existing facts, you can trigger re-indexing by running `condense`

---

## Architecture {#architecture}

All providers implement the `EmbeddingProvider` abstract base class:

```
EmbeddingProvider (ABC)
├── name: str           # Provider identifier
├── dims: int           # Vector dimensions
├── is_available(): bool  # Check prerequisites
├── embed_texts([str]): [[float]]  # Batch embedding
└── embed_single(str): [float]     # Single text (convenience)
```

The `create_embedding_provider(config)` factory function reads the plugin configuration and returns the appropriate provider instance. If the provider is `"none"` or unavailable, it returns `None` and semantic search is gracefully disabled.

---

## Troubleshooting {#troubleshooting}

### "Semantic search unavailable"

Check in order:

1. Is `semantic_search: true` in config?
2. Is `sqlite-vec` installed? (`pip install sqlite-vec`)
3. Is `embedding_provider` set to something other than `none`?
4. Is the required API key set (for cloud providers)?
5. Is `sentence-transformers` installed (for local providers)?

### "No Gemini API key configured"

1. Check `$GOOGLE_API_KEY` or `$GEMINI_API_KEY` is set
2. Or add it to `$HERMES_HOME/.env`
3. Or set `embedding_api_key` in config.yaml

### "No OpenAI API key configured"

1. Check `$OPENAI_API_KEY` is set
2. Or set `embedding_api_key` in config.yaml

### "Failed to load local model"

1. Ensure `sentence-transformers` is installed: `pip install sentence-transformers`
2. Check the model name is valid (typos in model name)
3. Ensure enough disk space for model download
4. Check internet connectivity (for first download)

### Vector table dimension mismatch

If you switched providers and get dimension errors:

1. Delete the old vector table: the plugin will recreate it
2. Or delete the entire database and let it rebuild (you'll lose existing memories)

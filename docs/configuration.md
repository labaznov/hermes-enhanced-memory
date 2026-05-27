# Configuration

Complete reference for all Enhanced Memory plugin configuration options.

---

## Basic Configuration {#basic}

Add the following to your Hermes Agent profile's `config.yaml` (typically at `~/.hermes/config.yaml` or `$HERMES_HOME/config.yaml`):

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    db_path: $HERMES_HOME/memory_store.db
    auto_extract: true
    auto_condense: true
    semantic_search: true
    embedding_provider: gemini
```

---

## Full Configuration Reference {#reference}

```yaml
plugins:
  enhanced-memory:
    # ── Storage ──────────────────────────────────────────────
    # Path to the SQLite database file.
    # Supports $HERMES_HOME and ${HERMES_HOME} variable expansion.
    # Default: $HERMES_HOME/memory_store.db
    db_path: $HERMES_HOME/memory_store.db

    # ── Automatic Behavior ───────────────────────────────────
    # Auto-extract facts from conversations at session end.
    # Default: true
    auto_extract: true

    # Auto-condense facts periodically (every 20 turns and at session end).
    # Default: true
    auto_condense: true

    # ── Semantic Search ──────────────────────────────────────
    # Enable semantic vector search via sqlite-vec.
    # Requires sqlite-vec and an embedding provider.
    # Default: true
    semantic_search: true

    # ── Embedding Provider ───────────────────────────────────
    # Which embedding provider to use for semantic search.
    # Options: gemini, openai, openai-large, local, local-multilingual, none
    # Default: gemini
    embedding_provider: gemini

    # Override the default model for the chosen provider.
    # Each provider has a sensible default (see table below).
    # embedding_model: gemini-embedding-001

    # Override embedding dimensions.
    # Usually auto-detected from the provider/model.
    # embedding_dims: 3072

    # Device for local models (sentence-transformers).
    # Options: cpu, cuda, mps
    # Default: cpu
    # embedding_device: cpu

    # Custom base URL for OpenAI-compatible APIs.
    # Useful for Azure OpenAI, local servers, etc.
    # embedding_base_url: https://your-api.example.com/v1

    # Explicit API key (overrides environment variable).
    # Generally, prefer environment variables for security.
    # embedding_api_key: sk-...
```

---

## Configuration Parameters {#parameters}

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | string | `$HERMES_HOME/memory_store.db` | Path to SQLite database. Supports `$HERMES_HOME` expansion |
| `auto_extract` | boolean | `true` | Extract facts from conversations at session end |
| `auto_condense` | boolean | `true` | Run condensation every 20 turns and at session end |
| `semantic_search` | boolean | `true` | Enable semantic vector search |

### Embedding Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_provider` | string | `gemini` | Provider: `gemini`, `openai`, `openai-large`, `local`, `local-multilingual`, `none` |
| `embedding_model` | string | *(auto)* | Model name override |
| `embedding_dims` | integer | *(auto)* | Embedding dimensions override |
| `embedding_device` | string | `cpu` | Device for local models: `cpu`, `cuda`, `mps` |
| `embedding_base_url` | string | *(none)* | Custom API base URL for OpenAI-compatible APIs |
| `embedding_api_key` | string | *(none)* | Explicit API key (overrides env vars) |

### Provider Defaults

| Provider Key | Default Model | Default Dimensions | Auth Required |
|-------------|---------------|-------------------|---------------|
| `gemini` | `gemini-embedding-001` | 3072 | `GOOGLE_API_KEY` |
| `openai` | `text-embedding-3-small` | 1536 | `OPENAI_API_KEY` |
| `openai-large` | `text-embedding-3-large` | 3072 | `OPENAI_API_KEY` |
| `local` | `all-MiniLM-L6-v2` | 384 | *(none)* |
| `local-multilingual` | `intfloat/multilingual-e5-large` | 1024 | *(none)* |
| `none` | — | — | — |

---

## Environment Variables {#env-vars}

Environment variables can be set in your shell or in the `$HERMES_HOME/.env` file.

| Variable | Provider | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Gemini | Google API key for Gemini embeddings |
| `GEMINI_API_KEY` | Gemini | Alternative to `GOOGLE_API_KEY` |
| `OPENAI_API_KEY` | OpenAI | OpenAI API key |
| `OPENAI_BASE_URL` | OpenAI | Custom API base URL |
| `HERMES_HOME` | All | Hermes Agent home directory (affects `db_path` resolution) |

### Key Resolution Order

**Gemini:**
1. `embedding_api_key` in config
2. `$GOOGLE_API_KEY` environment variable
3. `$GEMINI_API_KEY` environment variable
4. `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `$HERMES_HOME/.env`

**OpenAI:**
1. `embedding_api_key` in config
2. `$OPENAI_API_KEY` environment variable

### .env File Format

```bash
# $HERMES_HOME/.env
GOOGLE_API_KEY=AIza...your-key-here
# or
OPENAI_API_KEY=sk-...your-key-here
```

---

## Configuration Examples {#examples}

### Minimal (Keyword Search Only)

No semantic search, no API keys needed:

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    semantic_search: false
    embedding_provider: none
```

### Gemini Embeddings (Default)

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    embedding_provider: gemini
    # Set GOOGLE_API_KEY in env or .env file
```

### OpenAI Embeddings

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    embedding_provider: openai
    # Set OPENAI_API_KEY in env
```

### OpenAI Large Embeddings (Higher Quality)

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    embedding_provider: openai-large
    # 3072 dimensions — better quality, more storage
```

### Local Embeddings (Fully Offline)

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    embedding_provider: local
    embedding_device: cpu    # or cuda for GPU
```

Requires: `pip install sentence-transformers`

### Local Multilingual Embeddings

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    embedding_provider: local-multilingual
    embedding_device: cuda   # recommended — large model
```

Requires: `pip install sentence-transformers`

### Azure OpenAI / Custom API

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    embedding_provider: openai
    embedding_model: text-embedding-3-small
    embedding_base_url: https://your-resource.openai.azure.com/openai/deployments/embedding
    embedding_api_key: your-azure-key
```

### Custom Database Path

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    db_path: /custom/path/to/memory.db
```

### Disable All Automation

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    auto_extract: false
    auto_condense: false
    semantic_search: false
    embedding_provider: none
```

---

## Graceful Degradation {#degradation}

Enhanced Memory is designed to degrade gracefully:

1. **No sqlite-vec installed** → Semantic search disabled, FTS5 still works
2. **No API key for embedding provider** → Semantic search disabled, FTS5 still works
3. **embedding_provider: none** → Semantic search explicitly disabled
4. **auto_extract: false** → No automatic fact extraction; manual `add` still works
5. **auto_condense: false** → No automatic condensation; manual `condense` still works

The plugin always reports its status in the system prompt, indicating whether semantic search is enabled or disabled.

# Hermes Enhanced Memory Plugin

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 187 passed](https://img.shields.io/badge/tests-187%20passed-green.svg)]()
[![Coverage: 78%](https://img.shields.io/badge/coverage-78%25-brightgreen.svg)]()

A local-first, zero-external-dependency memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent) with two-tier fact storage, automatic condensation, and semantic vector search.

## Why?

Hermes Agent's built-in memory is a flat key-value store. Cloud providers like Honcho are unreliable. This plugin provides:

- **Two-tier storage**: raw facts → auto-condensation → priority-scored condensed knowledge
- **FTS5 full-text search** on both tiers (zero API calls, instant)
- **Semantic vector search** via sqlite-vec + pluggable embedding providers (Gemini/OpenAI/Local)
- **Automatic fact extraction** from conversations
- **Priority-based deduplication** — 80% overlap detection prevents bloat
- **100% local** — all data stays in a single SQLite file, no external services required

## Quick Start

### Install

```bash
# Clone into Hermes plugins directory
cd ~/.hermes/plugins
git clone https://github.com/labaznov/hermes-enhanced-memory.git enhanced_memory

# Or pip install (for entry point discovery)
pip install git+https://github.com/labaznov/hermes-enhanced-memory.git
```

### Configure

Add to `~/.hermes/config.yaml`:

```yaml
memory:
  provider: enhanced_memory

plugins:
  enhanced_memory:
    db_path: $HERMES_HOME/memory_store.db
    auto_extract: true       # Extract facts from conversations
    auto_condense: true      # Auto-condense after extraction
    semantic_search: true    # Enable vector search

    # Embedding provider (optional, for semantic search)
    embedding_provider: gemini   # gemini | openai | local | none
    embedding_model: gemini-embedding-001
    embedding_dims: 3072
```

### Embedding Providers

| Provider | Model | Dims | Requires |
|----------|-------|------|----------|
| `gemini` | gemini-embedding-001 | 3072 | `GOOGLE_API_KEY` |
| `openai` | text-embedding-3-small | 1536 | `OPENAI_API_KEY` |
| `openai` (custom) | any | varies | `base_url` config (vLLM, Ollama) |
| `local` | all-MiniLM-L6-v2 | 384 | `sentence-transformers` + `torch` |
| `none` | — | — | FTS5-only, no semantic search |

## Architecture

```
User message
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Facts   │────▶│  Condenser   │────▶│  Condensed   │
│  (FTS5)      │     │  (LLM-free)  │     │  (Priority)  │
└──────────────┘     └──────────────┘     └──────────────┘
    │                                          │
    ▼                                          ▼
┌──────────────┐                    ┌──────────────────┐
│ sqlite-vec   │◀───────────────────│  Embedding       │
│ (KNN search) │                    │  Provider        │
└──────────────┘                    └──────────────────┘
```

All data lives in a single `memory_store.db` SQLite file with WAL mode for concurrent access.

## Comparison with Alternatives

| Feature | Enhanced Memory | Honcho | Holographic | Built-in | Mem0 |
|---------|----------------|--------|-------------|----------|------|
| Local-first | ✅ | ❌ cloud | ✅ | ✅ | ❌ |
| Semantic search | ✅ sqlite-vec | ❌ | ❌ | ❌ | ✅ |
| Auto-condensation | ✅ | ❌ | ❌ | ❌ | ❌ |
| FTS5 search | ✅ | ❌ | ❌ | ❌ | ❌ |
| Zero external deps | ✅ | ❌ | ✅ | ✅ | ❌ |
| Priority scoring | ✅ | ❌ | ❌ | ❌ | ❌ |
| Two-tier storage | ✅ | ❌ | ❌ | ❌ | ❌ |

## Tool Actions

The plugin exposes these tools to the agent:

| Action | Description |
|--------|-------------|
| `memory_store` | Store a new fact with optional category |
| `memory_search` | FTS5 full-text search across all facts |
| `memory_condense` | Trigger manual condensation |
| `memory_semantic_search` | Vector similarity search (requires embeddings) |
| `memory_stats` | Database statistics and health |
| `memory_manage` | Update/delete individual facts |

## Development

```bash
# Install with dev deps
pip install -e ".[dev]"

# Run tests
cd tests && python -m pytest . -v

# Coverage
cd tests && python -m pytest . --cov=../enhanced_memory --cov-report=term-missing
```

### Test Results

```
187 passed in 2.6s

Coverage:
  store.py:              93%
  condenser.py:          93%
  embeddings.py:         80%
  embedding_providers.py: 75%
  __init__.py:           68%
  TOTAL:                 78%
```

## Documentation

Full documentation is in the [`docs/`](docs/) directory, built with [Diplodoc](https://diplodoc.com):

- [Overview](docs/index.md)
- [Getting Started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [API Reference](docs/api-reference.md)
- [Embedding Providers](docs/embedding-providers.md)
- [Comparison](docs/comparison.md)

Build docs locally:
```bash
npm i @diplodoc/cli -g
yfm build -i docs -o docs/_build
```

## License

MIT

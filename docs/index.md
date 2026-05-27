# Enhanced Memory Plugin

**Two-tier fact store with condensation, FTS5 full-text search, and pluggable semantic vector search for Hermes Agent.**

---

## What is Enhanced Memory? {#what-is-it}

Enhanced Memory is a **MemoryProvider plugin** for [Hermes Agent](https://hermes-agent.nousresearch.com/) that replaces the default flat key-value memory system with a structured, intelligent, two-tier architecture. It is designed to give AI agents a persistent, searchable, and self-maintaining long-term memory — entirely local, with zero external service dependencies.

Every fact captured during a session is stored as a **raw fact** with category metadata. A **condenser** periodically groups, deduplicates, and summarizes those raw facts into compact **condensed entries** ranked by priority. The condensed layer is what gets injected into the system prompt — keeping context windows small while retaining the most important information.

---

## Why Does It Exist? {#why}

AI assistants face a fundamental problem: **context windows are finite, but user knowledge accumulates indefinitely.** Without a structured memory system, important facts get lost as conversations scroll out of view.

Existing solutions either:

- Rely on **cloud services** that introduce latency, cost, and privacy concerns
- Use **simplistic flat stores** that don't scale or prioritize information
- Lack **semantic search**, making it impossible to find relevant facts by meaning rather than exact keywords

Enhanced Memory solves all of these problems with a local-first, SQLite-based architecture that requires nothing beyond Python's standard library for core functionality.

---

## Key Features {#features}

### Two-Tier Storage Architecture

Raw facts are stored individually and then condensed into high-priority summaries. This ensures that the system prompt always contains the most relevant, deduplicated information without consuming excessive tokens.

### FTS5 Full-Text Search

Both storage tiers are backed by SQLite FTS5 virtual tables with automatic trigger-based synchronization. This enables fast keyword search across the entire memory corpus, including prefix matching and phrase queries.

### Semantic Vector Search

Optionally embed all facts using a configurable provider (Gemini, OpenAI, or local sentence-transformers) and store vectors in **sqlite-vec** for true KNN similarity search. Find facts by meaning, not just keywords — across languages.

### Automatic Condensation

The condenser pipeline groups facts by category, deduplicates using 80% word-overlap detection, assigns priorities based on category and keyword boosts, and merges results into compact summaries. It runs automatically every 20 turns or at session end.

### Priority-Based Prompt Injection

Condensed entries are ranked by priority (1–10) and injected into the system prompt up to a configurable character limit (default: 2200 chars). Security-related facts always surface first; general observations surface last.

### Seven Fact Categories

Facts are classified into: `user_pref`, `project`, `tool`, `env`, `decision`, `security`, and `general`. Each category has a predefined priority range that determines how prominently facts appear in the agent's context.

### Lifecycle Hooks

The plugin integrates deeply with Hermes Agent via hooks:

- **on_session_end** — Auto-extract facts from conversations and condense
- **on_memory_write** — Mirror built-in memory writes as raw facts
- **on_pre_compress** — Extract facts before context-window compression discards messages

### Local-First, Zero External Dependencies

Core functionality (storage, search, condensation) requires only Python ≥ 3.10 and SQLite with FTS5 — both included in standard Python builds. Semantic search optionally requires `sqlite-vec` and one embedding provider.

---

## Quick Links {#quick-links}

- [Getting Started](getting-started.md) — Install, enable, and use in 5 minutes
- [Architecture](architecture.md) — How the two-tier system works internally
- [Configuration](configuration.md) — Full config.yaml reference
- [API Reference](api-reference.md) — All tool actions with parameters and examples
- [Embedding Providers](embedding-providers.md) — Setup guide for Gemini, OpenAI, and local models
- [Comparison with Alternatives](comparison.md) — Why Enhanced Memory vs. other solutions

---

## Requirements {#requirements}

### Core (always required)

- Python ≥ 3.10
- SQLite with FTS5 support (included in standard Python builds)

### Semantic Search (optional)

- **sqlite-vec** — `pip install sqlite-vec`
- **One embedding provider:**
  - Gemini: `GOOGLE_API_KEY` environment variable
  - OpenAI: `OPENAI_API_KEY` environment variable
  - Local: `pip install sentence-transformers`

---

## License {#license}

Part of the Hermes Agent ecosystem. See the main project license.

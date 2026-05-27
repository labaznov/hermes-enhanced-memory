# Architecture

Enhanced Memory uses a **two-tier architecture** where raw conversational facts flow through a condensation pipeline to produce compact, prioritized summaries. This page explains each component in detail.

---

## System Overview {#overview}

```
 Session N               Session N+1             System Prompt
 ─────────               ───────────             ─────────────
 ┌──────────┐            ┌──────────┐
 │  User    │            │  User    │
 │  message │            │  message │
 └────┬─────┘            └────┬─────┘
      │                       │
      ▼                       ▼
 ┌──────────────────────────────────────────────┐
 │              enhanced_memory tool            │
 │  actions: add / search / semantic_search /   │
 │           condense / list_condensed / stats  │
 └──────────┬──────────────────────┬────────────┘
            │                      │
            ▼                      ▼
 ┌─────────────────┐    ┌─────────────────────┐
 │   raw_facts     │    │   condensed         │
 │   (SQLite+FTS5) │───▶│   (SQLite+FTS5)     │
 │                 │    │                     │
 │  id, content    │    │  topic, category    │
 │  category       │    │  summary, priority  │
 │  source         │    │  source_ids [JSON]  │
 │  condensed flag │    │  version            │
 └─────────────────┘    └─────────┬───────────┘
                                  │
                        ┌─────────▼───────────┐
                        │  get_top_for_memory  │
                        │  (char_limit=2200)   │
                        └─────────┬───────────┘
                                  │  §-separated
                                  ▼
                        ┌─────────────────────┐
                        │  System Prompt       │
                        │  Memory Section      │
                        └─────────────────────┘
```

---

## Tier 1: Raw Facts {#raw-facts}

The `raw_facts` table stores individual factual statements as they arrive from conversations. Each fact records:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incrementing primary key |
| `content` | TEXT | The fact text |
| `category` | TEXT | Classification label (see Categories below) |
| `source` | TEXT | Origin: `dialog`, `manual`, `auto_extract`, `memory_write` |
| `session_id` | TEXT | Identifier of the originating session |
| `created_at` | TEXT | ISO-8601 UTC timestamp |
| `condensed` | INTEGER | Flag: 0 = not yet condensed, 1 = processed |

### FTS5 Index: raw_facts_fts

An external-content FTS5 virtual table indexes `content`, `category`, and `source` columns. Synchronization is maintained automatically via database triggers (INSERT, UPDATE, DELETE).

### How Facts Enter the System

Facts are added through multiple paths:

1. **Direct tool call** — `enhanced_memory` action: `add`
2. **Auto-extraction** — At session end, the plugin scans messages and extracts notable facts
3. **Memory write mirroring** — When the built-in Hermes memory tool writes, the fact is also captured
4. **Pre-compression extraction** — Before context window compression, facts are extracted from about-to-be-discarded messages

---

## Tier 2: Condensed Entries {#condensed}

The `condensed` table stores high-level summaries grouped by topic and category. Each entry is the product of the condensation pipeline.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incrementing primary key |
| `topic` | TEXT | Human-readable topic label |
| `summary` | TEXT | Condensed summary text |
| `category` | TEXT | Same classification as raw facts |
| `priority` | INTEGER | Importance score: 1 (low) to 10 (high) |
| `source_ids` | TEXT | JSON array of contributing raw_facts IDs |
| `fact_count` | INTEGER | Number of source facts |
| `version` | INTEGER | Increments on each update |
| `created_at` | TEXT | ISO-8601 UTC timestamp |
| `updated_at` | TEXT | Last modification timestamp |

A **unique index** on `(topic, category)` enforces that each topic-category pair maps to exactly one condensed row, enabling upsert semantics.

### FTS5 Index: condensed_fts

Indexes `topic`, `summary`, and `category` for full-text search across condensed entries.

---

## The Condensation Pipeline {#pipeline}

The `FactCondenser` processes uncondensed raw facts through a multi-stage pipeline:

```
raw_facts (condensed=0)
    │
    ▼
group by category
    │
    ▼
deduplicate (80% word-overlap threshold)
    │
    ▼
compute priority (category base + keyword boosts)
    │
    ▼
upsert into condensed table (create or merge)
    │
    ▼
mark source raw_facts as condensed=1
```

### Stage 1: Grouping

Uncondensed facts are fetched and grouped by their `category` label. Each group is processed independently.

### Stage 2: Deduplication

Within each group, facts are compared using Jaccard-style word overlap on the smaller set. If two facts share ≥ 80% of their words, the duplicate is discarded (first-in wins). This prevents the condensed layer from being cluttered with near-identical statements.

### Stage 3: Priority Calculation

Each category has a base priority range:

| Category | Base Range | Description |
|----------|-----------|-------------|
| `security` | 9–10 | Security-critical information |
| `user_pref` | 8–9 | User preferences and habits |
| `decision` | 7–9 | Decisions and choices made |
| `project` | 7 | Project and work details |
| `tool` | 6–8 | Tools and configurations |
| `env` | 5 | Environment and infrastructure |
| `general` | 4 | General observations |

**Keyword boosts** are applied on top of the base:

- **+1**: `prefers`, `always`, `never` (and Russian equivalents)
- **+2**: `password`, `key`, `secret` (and Russian equivalents)

Priority is clamped to a maximum of 10.

### Stage 4: Upsert

The condensed summary is inserted or merged into the `condensed` table. If a row with the same `(topic, category)` already exists, it is updated with the new summary and its `version` is incremented.

### Stage 5: Mark as Processed

Source raw facts have their `condensed` flag set to 1, so they are not reprocessed in future runs.

---

## Semantic Search Layer {#semantic-search}

The optional semantic layer adds vector similarity search on top of the keyword-based FTS5 system.

```
┌────────────────────────────────────────────┐
│  embedding_providers.py                    │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ Gemini   │ │ OpenAI   │ │  Local     │ │
│  │ API      │ │ API      │ │  (ST/e5)   │ │
│  └────┬─────┘ └────┬─────┘ └─────┬──────┘ │
│       └─────┬──────┘             │         │
│             ▼                    ▼         │
│        sqlite-vec (KNN search)             │
└────────────────────────────────────────────┘
```

### How It Works

1. When a fact is stored via `add`, it is also embedded using the configured provider
2. The embedding vector is stored in a `vec0` virtual table (sqlite-vec)
3. On `semantic_search`, the query is embedded and a KNN search finds the closest facts
4. Results include a distance score and similarity percentage

### ID Mapping

To store both raw facts and condensed entries in the same vector table without ID collisions, condensed entry IDs are mapped to negative space:

```
vector_id = -(condensed_id + 10000)
```

When resolving results, the system checks if `fact_id < -10000` to determine whether the match is from the condensed table.

---

## Prompt Injection {#prompt-injection}

The `get_top_for_memory` method retrieves condensed entries sorted by priority (descending) and concatenates them into a string, separated by section markers (`§`), up to a configurable character limit (default: 2200 characters).

This string is injected into the agent's system prompt, ensuring that the most important memories are always available in context without consuming excessive tokens.

### Prefetch

Before each agent turn, the `prefetch` method runs:

1. **Semantic search** (if available) — top 3 results by vector similarity
2. **FTS5 search on condensed** — top 3 keyword matches
3. **FTS5 search on raw facts** — supplementary matches

Results are deduplicated and the top 5 are injected as a "Memory Recall" section.

---

## SQLite Configuration {#sqlite-config}

The store uses several SQLite optimizations:

- **WAL mode** (Write-Ahead Logging) — Enables concurrent reads alongside writes
- **Foreign keys enabled** — Maintains referential integrity
- **Busy timeout: 5000ms** — Waits up to 5 seconds for locks instead of failing immediately
- **Thread-local connections** — Each thread gets its own connection via `threading.local()`
- **Write serialization** — A `threading.Lock()` ensures only one write transaction at a time

---

## File Structure {#files}

```
plugins/memory/enhanced_memory/
├── plugin.yaml              # Plugin metadata and hook declarations
├── __init__.py              # MemoryProvider implementation + tool handler
├── store.py                 # EnhancedMemoryStore — SQLite/FTS5 backend
├── condenser.py             # FactCondenser — grouping, dedup, prioritization
├── embedding_providers.py   # EmbeddingProvider ABC + Gemini/OpenAI/Local
├── embeddings.py            # SemanticSearch — provider-agnostic vec search
└── README.md                # Plugin documentation
```

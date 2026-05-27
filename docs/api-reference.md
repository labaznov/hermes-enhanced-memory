# API Reference

The Enhanced Memory plugin exposes a single tool called `enhanced_memory` with six actions. This page documents each action with parameters, example requests, and response formats.

---

## Tool: enhanced_memory {#tool}

**Type:** Hermes Agent tool (called by the AI agent during conversation)

**Description:** Two-tier persistent memory with condensation and semantic search. Stores facts across sessions, automatically groups and deduplicates them.

**Required parameter for all actions:** `action`

---

## memory_store (action: add) {#add}

Store a new raw fact in the memory database.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | ✅ | — | Must be `"add"` |
| `content` | string | ✅ | — | The fact text to store |
| `category` | string | — | `"general"` | One of: `user_pref`, `project`, `tool`, `env`, `decision`, `security`, `general` |
| `source` | string | — | `"dialog"` | Origin: `dialog`, `manual`, `auto_extract` |

### Example Request

```json
{
  "action": "add",
  "content": "User prefers Python 3.12 with type hints in all code",
  "category": "user_pref",
  "source": "dialog"
}
```

### Example Response

```json
{
  "fact_id": 42,
  "status": "added"
}
```

### Behavior

- The fact is inserted into the `raw_facts` table with `condensed=0`
- If semantic search is enabled, the fact is immediately embedded and indexed in the vector table
- The FTS5 index is automatically updated via database triggers
- The `session_id` is captured from the current session context

### Category Guide

| Category | When to Use | Example |
|----------|------------|---------|
| `user_pref` | User preferences, habits, style choices | "User prefers dark mode" |
| `project` | Project-specific details, tech stack, architecture | "Project uses FastAPI + PostgreSQL" |
| `tool` | Tool configurations, CLI preferences | "Uses neovim as primary editor" |
| `env` | Server details, infrastructure, paths | "Production server: 192.168.1.100" |
| `decision` | Decisions made, trade-offs chosen | "Decided to use SQLite over Postgres for simplicity" |
| `security` | Credentials, access patterns, security policies | "SSH key is at ~/.ssh/prod_key" |
| `general` | Anything that doesn't fit other categories | "Speaks English and Russian" |

---

## memory_search (action: search) {#search}

Full-text search across both raw facts and condensed entries using SQLite FTS5.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | ✅ | — | Must be `"search"` |
| `query` | string | ✅ | — | FTS5 search query |
| `limit` | integer | — | `10` | Maximum results per tier |

### Example Request

```json
{
  "action": "search",
  "query": "python preferences",
  "limit": 5
}
```

### Example Response

```json
{
  "raw_facts": [
    {
      "id": 42,
      "content": "User prefers Python 3.12 with type hints in all code",
      "category": "user_pref",
      "source": "dialog",
      "created_at": "2026-05-27T10:30:00+00:00"
    },
    {
      "id": 15,
      "content": "Always use ruff for Python linting",
      "category": "tool",
      "source": "dialog",
      "created_at": "2026-05-26T14:20:00+00:00"
    }
  ],
  "condensed": [
    {
      "id": 3,
      "topic": "User: preferences",
      "summary": "Prefers Python 3.12 with type hints. Uses ruff for linting. Favors explicit over implicit.",
      "category": "user_pref",
      "priority": 9
    }
  ],
  "total": 3
}
```

### FTS5 Query Syntax

The `query` parameter supports FTS5 query syntax:

| Syntax | Example | Description |
|--------|---------|-------------|
| Simple terms | `python` | Match documents containing "python" |
| Multiple terms | `python preferences` | Match documents containing both terms |
| Phrase | `"dark mode"` | Match exact phrase |
| Prefix | `pyth*` | Match terms starting with "pyth" |
| OR | `python OR javascript` | Match either term |
| NOT | `python NOT java` | Match "python" but not "java" |
| Column filter | `category:user_pref` | Search in specific FTS5 column |

---

## memory_condense (action: condense) {#condense}

Run the condensation pipeline: group uncondensed facts by category, deduplicate, prioritize, and upsert into the condensed table.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | ✅ | — | Must be `"condense"` |
| `dry_run` | boolean | — | `false` | If `true`, preview results without writing to database |

### Example Request

```json
{
  "action": "condense",
  "dry_run": false
}
```

### Example Response

```json
{
  "entries": [
    {
      "topic": "User: preferences",
      "category": "user_pref",
      "priority": 9,
      "fact_count": 5,
      "action": "updated"
    },
    {
      "topic": "Tools & configuration",
      "category": "tool",
      "priority": 7,
      "fact_count": 3,
      "action": "created"
    }
  ],
  "count": 2,
  "dry_run": false
}
```

### Action Values

| Action | Description |
|--------|-------------|
| `"created"` | New condensed entry was created |
| `"updated"` | Existing condensed entry was merged with new facts |

### Dry Run

When `dry_run: true`:

- The full pipeline executes (grouping, deduplication, priority calculation)
- Results are returned in the same format
- **Nothing is written to the database**
- Raw facts are NOT marked as condensed

This is useful for previewing what the condenser would do before committing.

### Automatic Condensation

If `auto_condense: true` in config, condensation runs automatically:

- Every 20 conversational turns (if there are > 10 uncondensed facts)
- At session end (if there are > 5 uncondensed facts)

---

## memory_semantic_search (action: semantic_search) {#semantic-search}

Find facts by semantic meaning using vector similarity search. Requires an active embedding provider and sqlite-vec.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | ✅ | — | Must be `"semantic_search"` |
| `query` | string | ✅ | — | Natural-language query |
| `limit` | integer | — | `5` | Maximum results |

### Example Request

```json
{
  "action": "semantic_search",
  "query": "what text editor does the user prefer?",
  "limit": 3
}
```

### Example Response (Success)

```json
{
  "results": [
    {
      "source": "raw_facts",
      "id": 18,
      "content": "Uses neovim as primary editor with LazyVim config",
      "category": "tool",
      "distance": 0.198,
      "similarity": 0.802
    },
    {
      "source": "condensed",
      "id": 5,
      "topic": "Tools & configuration",
      "content": "Primary editor: neovim (LazyVim). Uses tmux for terminal multiplexing. Prefers CLI tools over GUIs.",
      "category": "tool",
      "priority": 7,
      "distance": 0.312,
      "similarity": 0.688
    }
  ],
  "count": 2
}
```

### Example Response (Unavailable)

```json
{
  "error": "Semantic search unavailable. Check: sqlite-vec installed, embedding provider configured, and required API key set.",
  "fallback": "Use 'search' action for FTS5 keyword search instead."
}
```

### Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | `"raw_facts"` or `"condensed"` |
| `id` | integer | Row ID in the source table |
| `content` | string | The fact text (or summary for condensed) |
| `category` | string | Fact category |
| `distance` | float | Cosine distance (lower = more similar) |
| `similarity` | float | 1 - distance (higher = more similar) |
| `topic` | string | *(condensed only)* Topic label |
| `priority` | integer | *(condensed only)* Priority score |

---

## memory_stats (action: stats) {#stats}

Return counts and metadata about the memory store.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | ✅ | — | Must be `"stats"` |

### Example Request

```json
{
  "action": "stats"
}
```

### Example Response

```json
{
  "raw_total": 87,
  "raw_uncondensed": 12,
  "condensed_total": 8,
  "categories": {
    "user_pref": 23,
    "project": 18,
    "tool": 15,
    "env": 12,
    "decision": 9,
    "security": 4,
    "general": 6
  },
  "semantic_search": "enabled",
  "embedding_provider": "gemini"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `raw_total` | integer | Total number of raw facts |
| `raw_uncondensed` | integer | Facts not yet processed by the condenser |
| `condensed_total` | integer | Number of condensed entries |
| `categories` | object | Breakdown of raw facts by category |
| `semantic_search` | string | `"enabled"` or `"disabled"` |
| `embedding_provider` | string | Active provider name or `"none"` |

---

## memory_manage (action: list_condensed) {#list-condensed}

Return all condensed entries sorted by priority (highest first).

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | ✅ | — | Must be `"list_condensed"` |
| `category` | string | — | *(all)* | Filter by category |
| `limit` | integer | — | `20` | Maximum entries to return |

### Example Request

```json
{
  "action": "list_condensed",
  "limit": 5
}
```

### Example Response

```json
{
  "entries": [
    {
      "id": 1,
      "topic": "Security",
      "summary": "SSH key at ~/.ssh/prod_key. Production access via VPN only. 2FA enabled on all services.",
      "category": "security",
      "priority": 10,
      "fact_count": 4,
      "version": 2,
      "updated_at": "2026-05-27T11:00:00+00:00"
    },
    {
      "id": 2,
      "topic": "User: preferences",
      "summary": "Prefers Python 3.12 with type hints. Dark mode everywhere. Uses ruff for linting.",
      "category": "user_pref",
      "priority": 9,
      "fact_count": 6,
      "version": 3,
      "updated_at": "2026-05-27T10:45:00+00:00"
    }
  ],
  "count": 2
}
```

### Filtering by Category

```json
{
  "action": "list_condensed",
  "category": "security"
}
```

Returns only condensed entries with `category: "security"`.

---

## Error Handling {#errors}

All actions return JSON. Errors follow this format:

```json
{
  "error": "Description of what went wrong"
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `'content' is required for 'add' action` | Missing `content` parameter | Provide the fact text |
| `'query' is required for 'search' action` | Missing `query` parameter | Provide a search query |
| `Semantic search unavailable` | sqlite-vec not installed or no API key | Install sqlite-vec, set API key, or use FTS5 search |

---

## Usage Patterns {#patterns}

### When the Agent Should Use Each Action

| Situation | Recommended Action |
|-----------|--------------------|
| User shares a preference or correction | `add` with `category: "user_pref"` |
| Agent discovers environment details | `add` with `category: "env"` or `"tool"` |
| Need context about the user | `semantic_search` first, then `search` |
| After a long session | `condense` to compress facts |
| User asks what the agent remembers | `list_condensed` or `stats` |
| Debugging memory issues | `stats` to check counts and status |

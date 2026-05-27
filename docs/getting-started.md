# Getting Started

This guide walks you through installing, enabling, and using the Enhanced Memory plugin for Hermes Agent in under 5 minutes.

---

## Step 1: Install the Plugin {#install}

The Enhanced Memory plugin ships as a directory under Hermes Agent's plugin system. Ensure the plugin files are in place:

```
plugins/memory/enhanced_memory/
├── plugin.yaml
├── __init__.py
├── store.py
├── condenser.py
├── embedding_providers.py
├── embeddings.py
└── README.md
```

### Install Optional Dependencies

For semantic vector search (recommended but not required):

```bash
# sqlite-vec for vector storage
pip install sqlite-vec

# For Gemini embeddings (default) — just need an API key
# For OpenAI embeddings — just need an API key
# For local embeddings — install sentence-transformers
pip install sentence-transformers
```

---

## Step 2: Enable the Plugin {#enable}

Edit your Hermes Agent profile's `config.yaml`:

```yaml
memory:
  provider: enhanced-memory

plugins:
  enhanced-memory:
    db_path: $HERMES_HOME/memory_store.db
    auto_extract: true
    auto_condense: true
    semantic_search: true
    embedding_provider: gemini   # or: openai, local, none
```

If using Gemini (default), set your API key:

```bash
export GOOGLE_API_KEY="your-google-api-key"
```

Or add it to `$HERMES_HOME/.env`:

```
GOOGLE_API_KEY=your-google-api-key
```

---

## Step 3: Store Your First Facts {#first-facts}

Once the plugin is active, the agent has access to the `enhanced_memory` tool. You can ask the agent to store facts, or the agent will auto-extract them from conversations.

### Manual Storage

Tell the agent something it should remember:

> "Remember that I prefer dark mode in all applications."

The agent will call:

```json
{
  "action": "add",
  "content": "User prefers dark mode in all applications",
  "category": "user_pref",
  "source": "dialog"
}
```

**Response:**

```json
{
  "fact_id": 1,
  "status": "added"
}
```

### Store Different Categories

```json
{
  "action": "add",
  "content": "Production server is at 192.168.1.100, SSH port 2222",
  "category": "env",
  "source": "dialog"
}
```

```json
{
  "action": "add",
  "content": "Project uses Python 3.12, Poetry for deps, pytest for testing",
  "category": "project",
  "source": "dialog"
}
```

---

## Step 4: Search Your Memory {#search}

### Keyword Search (FTS5)

```json
{
  "action": "search",
  "query": "dark mode",
  "limit": 5
}
```

**Response:**

```json
{
  "raw_facts": [
    {
      "id": 1,
      "content": "User prefers dark mode in all applications",
      "category": "user_pref",
      "source": "dialog",
      "created_at": "2026-05-27T10:30:00+00:00"
    }
  ],
  "condensed": [],
  "total": 1
}
```

### Semantic Search

Find facts by meaning, not just keywords:

```json
{
  "action": "semantic_search",
  "query": "what theme does the user like?",
  "limit": 3
}
```

This will find the "dark mode" fact even though the query uses completely different words — because the embeddings capture semantic similarity.

**Response:**

```json
{
  "results": [
    {
      "source": "raw_facts",
      "id": 1,
      "content": "User prefers dark mode in all applications",
      "category": "user_pref",
      "distance": 0.234,
      "similarity": 0.766
    }
  ],
  "count": 1
}
```

---

## Step 5: Run Condensation {#condense}

After accumulating several facts, run the condenser to group, deduplicate, and summarize:

```json
{
  "action": "condense",
  "dry_run": false
}
```

**Response:**

```json
{
  "entries": [
    {
      "topic": "User: preferences",
      "category": "user_pref",
      "priority": 8,
      "fact_count": 3,
      "action": "created"
    },
    {
      "topic": "Environment & infrastructure",
      "category": "env",
      "priority": 5,
      "fact_count": 2,
      "action": "created"
    }
  ],
  "count": 2,
  "dry_run": false
}
```

### Preview Without Writing

Set `dry_run: true` to see what the condenser would do without making changes:

```json
{
  "action": "condense",
  "dry_run": true
}
```

---

## Step 6: Check Memory Stats {#stats}

```json
{
  "action": "stats"
}
```

**Response:**

```json
{
  "raw_total": 15,
  "raw_uncondensed": 3,
  "condensed_total": 4,
  "categories": {
    "user_pref": 5,
    "project": 4,
    "env": 3,
    "tool": 2,
    "general": 1
  },
  "semantic_search": "enabled",
  "embedding_provider": "gemini"
}
```

---

## Automatic Behavior {#automatic}

Once enabled, Enhanced Memory works automatically in the background:

1. **Auto-extraction**: At session end, the plugin scans the conversation and extracts notable facts
2. **Auto-condensation**: Every 20 conversational turns, or at session end, uncondensed facts are processed
3. **Prefetch**: Before each agent response, relevant memories are fetched and injected into context
4. **Memory mirroring**: Writes to the built-in Hermes memory system are also captured as raw facts

You don't need to explicitly call any tool actions for basic functionality — just have conversations and the memory builds itself.

---

## Next Steps {#next-steps}

- [Architecture](architecture.md) — Understand the two-tier design in depth
- [Configuration](configuration.md) — Tune every parameter
- [API Reference](api-reference.md) — Complete tool action documentation
- [Embedding Providers](embedding-providers.md) — Choose and configure your embedding backend

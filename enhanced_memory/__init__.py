"""Enhanced Memory — two-tier fact store with condensation and semantic search.

A Hermes Agent :class:`~agent.memory_provider.MemoryProvider` plugin that
provides:

- **Two-tier fact storage**: raw conversational facts are periodically
  condensed into higher-level summaries via the :class:`~condenser.FactCondenser`.
- **FTS5 full-text search** on both raw and condensed tiers.
- **Optional semantic vector search** via ``sqlite-vec`` with pluggable
  embedding providers (Gemini, OpenAI, local sentence-transformers).
- **Automatic fact extraction** from conversations using regex pattern
  matching at session end or before context compression.
- **Priority-based memory condensation** with category-aware scoring.

Plugin lifecycle::

    register(ctx)
        │
        ▼
    ctx.register_memory_provider(EnhancedMemoryProvider)
        │
        ▼
    initialize(session_id)  →  store + condenser + semantic search
        │
        ▼
    prefetch(query)  →  recall relevant facts for each turn
    sync_turn(...)   →  periodic auto-condensation
    on_session_end(...)  →  extract facts from conversation
    shutdown()       →  clean up

Configuration in ``$HERMES_HOME/config.yaml``::

    memory:
      provider: enhanced-memory

    plugins:
      enhanced-memory:
        db_path: $HERMES_HOME/memory_store.db
        auto_extract: true
        auto_condense: true
        semantic_search: true
        embedding_provider: gemini
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from hermes_cli.config import cfg_get

# try/except pattern: relative imports work when loaded as a package by
# Hermes Agent; absolute imports are the fallback for standalone / test use.
try:
    from .store import EnhancedMemoryStore
    from .condenser import FactCondenser
except ImportError:
    from store import EnhancedMemoryStore
    from condenser import FactCondenser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

ENHANCED_MEMORY_SCHEMA = {
    "name": "enhanced_memory",
    "description": (
        "Two-tier persistent memory with condensation and semantic search. "
        "Stores facts across sessions, automatically groups and deduplicates them.\n\n"
        "ACTIONS:\n"
        "• add — Store a fact (preference, decision, env detail, project info).\n"
        "• search — FTS5 keyword search across all facts.\n"
        "• semantic_search — Find facts by MEANING (cross-language, synonym-aware). "
        "Configurable: Gemini, OpenAI, or local sentence-transformers.\n"
        "• condense — Run the condensation pipeline: group, deduplicate, prioritize.\n"
        "• list_condensed — View condensed memory entries sorted by priority.\n"
        "• stats — Memory statistics (counts, categories, index status).\n\n"
        "CATEGORIES: user_pref, project, tool, env, decision, security, general.\n\n"
        "WHEN TO USE:\n"
        "- User shares preferences or corrections → add with category 'user_pref'\n"
        "- You discover env/tool details → add with 'tool' or 'env'\n"
        "- Need context about the user → semantic_search first, then search\n"
        "- After long sessions → condense to compress facts"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "semantic_search", "condense", "list_condensed", "stats"],
                "description": "Action to perform.",
            },
            "content": {
                "type": "string",
                "description": "Fact content (required for 'add').",
            },
            "query": {
                "type": "string",
                "description": "Search query (required for 'search' and 'semantic_search').",
            },
            "category": {
                "type": "string",
                "enum": ["user_pref", "project", "tool", "env", "decision", "security", "general"],
                "description": "Fact category. Default: 'general'.",
            },
            "source": {
                "type": "string",
                "description": "Fact source: 'dialog', 'manual', 'auto_extract'. Default: 'dialog'.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results for search (default: 10).",
            },
            "dry_run": {
                "type": "boolean",
                "description": "For 'condense': preview without writing. Default: false.",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    """Load enhanced-memory plugin configuration from ``config.yaml``.

    Reads ``plugins.enhanced-memory`` from the Hermes home ``config.yaml``.
    Falls back to ``~/.hermes/config.yaml`` if ``hermes_constants`` is not
    importable.

    Returns:
        dict: Plugin configuration dict, or ``{}`` if the config file is
        missing or cannot be parsed.
    """
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
    except Exception:
        from pathlib import Path
        config_path = Path.home() / ".hermes" / "config.yaml"

    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "enhanced-memory", default={}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class EnhancedMemoryProvider(MemoryProvider):
    """Two-tier memory with condensation and optional semantic vector search.

    Implements the Hermes Agent :class:`~agent.memory_provider.MemoryProvider`
    interface.  Manages:

    * An :class:`~store.EnhancedMemoryStore` for durable fact storage.
    * A :class:`~condenser.FactCondenser` for grouping and deduplicating facts.
    * An optional :class:`~embeddings.SemanticSearch` engine for KNN vector
      lookups.

    Attributes:
        _config (dict): Plugin configuration values.
        _store (EnhancedMemoryStore | None): Initialised store instance.
        _condenser (FactCondenser | None): Condensation engine.
        _semantic (SemanticSearch | None): Semantic search (lazy-loaded).
        _session_id (str): Current session identifier.
        _session_turns (int): Turn counter for auto-condensation scheduling.
        _auto_extract (bool): Whether to auto-extract facts from conversations.
        _auto_condense (bool): Whether to auto-condense periodically.
        _semantic_enabled (bool): Whether semantic search should be initialised.

    Args:
        config: Plugin configuration dict.  If ``None``, configuration is
            loaded from ``config.yaml`` via :func:`_load_plugin_config`.
    """

    def __init__(self, config: dict | None = None):
        """Initialise the provider (store is created later in :meth:`initialize`).

        Args:
            config: Plugin configuration dict or ``None`` to auto-load.
        """
        self._config = config or _load_plugin_config()
        self._store: Optional[EnhancedMemoryStore] = None
        self._condenser: Optional[FactCondenser] = None
        self._semantic: Optional[Any] = None  # SemanticSearch, lazy-loaded
        self._session_id: str = ""
        self._session_turns: int = 0
        self._auto_extract: bool = bool(self._config.get("auto_extract", True))
        self._auto_condense: bool = bool(self._config.get("auto_condense", True))
        self._semantic_enabled: bool = bool(self._config.get("semantic_search", True))

    @property
    def name(self) -> str:
        """Return the plugin name used for registration.

        Returns:
            str: ``'enhanced-memory'``.
        """
        return "enhanced-memory"

    def is_available(self) -> bool:
        """Check whether the plugin can operate.

        Always returns ``True`` because the core dependency (SQLite) is part
        of the Python standard library.  Optional features (semantic search)
        degrade gracefully when their dependencies are missing.

        Returns:
            bool: Always ``True``.
        """
        return True

    def get_config_schema(self) -> list:
        """Return the list of configurable keys for the setup wizard.

        Each entry is a dict with ``key``, ``description``, ``default``,
        and optionally ``choices``.

        Returns:
            list[dict]: Configuration schema entries.
        """
        try:
            from hermes_constants import display_hermes_home
            _default_db = f"{display_hermes_home()}/memory_store.db"
        except Exception:
            _default_db = "~/.hermes/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts from conversations at session end",
             "default": "true", "choices": ["true", "false"]},
            {"key": "auto_condense", "description": "Auto-condense facts periodically",
             "default": "true", "choices": ["true", "false"]},
            {"key": "semantic_search", "description": "Enable semantic vector search",
             "default": "true", "choices": ["true", "false"]},
            {"key": "embedding_provider", "description": "Embedding provider: gemini, openai, local, none",
             "default": "gemini", "choices": ["gemini", "openai", "openai-large", "local", "local-multilingual", "none"]},
            {"key": "embedding_model", "description": "Embedding model name (provider-specific)",
             "default": "(auto from provider)"},
            {"key": "embedding_dims", "description": "Embedding dimensions (auto from provider, override if needed)",
             "default": "(auto)"},
            {"key": "embedding_device", "description": "Device for local models: cpu, cuda, mps",
             "default": "cpu", "choices": ["cpu", "cuda", "mps"]},
            {"key": "condenser_model", "description": "LLM model for condensation (e.g. gemini-2.5-flash, claude-haiku). Empty = auto-detect",
             "default": "(auto)"},
            {"key": "condenser_provider", "description": "LLM provider for condensation: google, openai, or custom base_url",
             "default": "(auto)"},
            {"key": "condenser_api_key", "description": "API key for condenser LLM (auto-detected from env if empty)",
             "default": ""},
            {"key": "condenser_base_url", "description": "Base URL for OpenAI-compatible condenser API",
             "default": ""},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write plugin configuration to ``config.yaml`` under ``plugins.enhanced-memory``.

        Merges the provided *values* into the existing YAML structure without
        overwriting other sections.

        Args:
            values: Key-value pairs to persist (e.g. ``{"db_path": "...", ...}``).
            hermes_home: Absolute path to the Hermes home directory.
        """
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["enhanced-memory"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as e:
            logger.warning("Failed to save enhanced-memory config: %s", e)

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialise the store, condenser, and optional semantic search engine.

        Called once by the Hermes Agent framework at session start.  Creates
        the :class:`EnhancedMemoryStore`, :class:`FactCondenser`, and
        optionally a :class:`SemanticSearch` engine.  The ``$HERMES_HOME``
        placeholder in ``db_path`` is expanded automatically.

        Args:
            session_id: Unique identifier for the current agent session.
            **kwargs: Reserved for future use by the framework.
        """
        try:
            from hermes_constants import get_hermes_home
            _hermes_home = str(get_hermes_home())
        except Exception:
            from pathlib import Path
            _hermes_home = str(Path.home() / ".hermes")

        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        if isinstance(db_path, str):
            # Expand the $HERMES_HOME placeholder so users can use it in config.
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)

        self._store = EnhancedMemoryStore(db_path=db_path)

        # Build LLM config for condenser from plugin config
        llm_config = self._build_condenser_llm_config()
        self._condenser = FactCondenser(self._store, llm_config=llm_config)
        self._session_id = session_id
        self._session_turns = 0

        # Lazy-init semantic search
        if self._semantic_enabled:
            self._init_semantic(db_path)

        logger.info("Enhanced memory initialized: %s", db_path)

    def _build_condenser_llm_config(self) -> Optional[Dict[str, Any]]:
        """Build LLM configuration dict for FactCondenser from plugin config.

        Reads ``condenser_model``, ``condenser_provider``, ``condenser_api_key``,
        and ``condenser_base_url`` from the plugin config.  Returns None if
        no explicit config is set (FactCondenser will auto-detect from env).

        Returns:
            dict | None: LLM config dict, or None for auto-detection.
        """
        model = self._config.get("condenser_model", "")
        if not model or model == "(auto)":
            return None  # Let FactCondenser auto-detect

        config: Dict[str, Any] = {"model": model}

        provider = self._config.get("condenser_provider", "")
        if provider and provider != "(auto)":
            config["provider"] = provider

        api_key = self._config.get("condenser_api_key", "")
        if api_key:
            config["api_key"] = api_key

        base_url = self._config.get("condenser_base_url", "")
        if base_url:
            config["base_url"] = base_url

        return config

    def _init_semantic(self, db_path: str) -> None:
        """Try to initialise semantic vector search; fail gracefully.

        Imports :class:`SemanticSearch` and checks provider availability.
        If ``sqlite-vec`` is not installed or the embedding provider is not
        configured, ``self._semantic`` remains ``None`` and all vector
        search calls are silently skipped.

        Args:
            db_path: Path to the SQLite database (shared with the store).
        """
        try:
            from .embeddings import SemanticSearch
            self._semantic = SemanticSearch(db_path=db_path, config=self._config)
            if not self._semantic.is_available():
                pname = self._semantic.provider_name
                logger.info("Semantic search unavailable (provider=%s)", pname)
                self._semantic = None
        except ImportError:
            logger.debug("sqlite-vec not installed, semantic search disabled")
            self._semantic = None
        except Exception as e:
            logger.debug("Semantic search init failed: %s", e)
            self._semantic = None

    def system_prompt_block(self) -> str:
        """Return a status summary for injection into the system prompt.

        Includes raw/condensed fact counts and semantic search status.

        Returns:
            str: Markdown-formatted status block, or empty string if the
            store has not been initialised.
        """
        if not self._store:
            return ""
        try:
            s = self._store.stats()
            raw = s.get("raw_total", 0)
            condensed = s.get("condensed_total", 0)
            semantic = "enabled" if self._semantic else "disabled"

            return (
                "# Enhanced Memory\n"
                f"Active. {raw} raw facts, {condensed} condensed entries. "
                f"Semantic search: {semantic}.\n"
                "Use enhanced_memory tool to add/search facts, run condensation, "
                "or perform semantic search."
            )
        except Exception:
            return "# Enhanced Memory\nActive."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant facts for the upcoming conversational turn.

        Combines semantic search (best quality, if available) with FTS5
        keyword search (always available) to produce up to 5 relevant
        memory entries.

        Args:
            query: The user's upcoming message or a representative query.
            session_id: Optional session filter (currently unused).

        Returns:
            str: A Markdown block titled ``## Enhanced Memory Recall``
            with bullet-pointed results, or empty string if nothing found.
        """
        if not self._store or not query:
            return ""

        results = []

        # 1. Try semantic search first (best quality)
        if self._semantic:
            try:
                sem_results = self._semantic.search(query, k=3)
                for r in sem_results:
                    fact_id = r["fact_id"]
                    # Resolve content: negative IDs (< -10000) are condensed
                    # entries mapped via the _CONDENSED_ID_OFFSET formula.
                    if fact_id < -10000:
                        real_id = -(fact_id + 10000)
                        c = self._store.get_condensed_by_id(real_id)
                        if c:
                            results.append(f"- [condensed] {c['summary'][:200]}")
                    else:
                        rf = self._store.get_raw_by_id(fact_id)
                        if rf:
                            results.append(f"- {rf['content'][:200]}")
            except Exception as e:
                logger.debug("Semantic prefetch failed: %s", e)

        # 2. Fallback/supplement with FTS5
        try:
            # Search condensed first (higher signal)
            condensed = self._store.search_condensed(query=query, limit=3)
            for c in condensed:
                line = f"- [{c.get('category', '')}] {c['summary'][:200]}"
                if line not in results:
                    results.append(line)

            # Then raw facts
            if len(results) < 5:
                raw = self._store.search_raw(query, limit=3)
                for r in raw:
                    line = f"- {r['content'][:200]}"
                    if line not in results:
                        results.append(line)
        except Exception as e:
            logger.debug("FTS prefetch failed: %s", e)

        if not results:
            return ""

        return "## Enhanced Memory Recall\n" + "\n".join(results[:5])

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Track turns and trigger periodic auto-condensation.

        Called after each conversational turn.  Every 20 turns, if
        ``auto_condense`` is enabled and there are more than 10 uncondensed
        facts, the condenser is run automatically.

        Args:
            user_content: The user's message text.
            assistant_content: The assistant's response text.
            session_id: Session identifier (currently unused).
        """
        self._session_turns += 1

        # Auto-condense every 20 turns
        if self._auto_condense and self._session_turns % 20 == 0:
            try:
                uncondensed = self._store.stats().get("raw_uncondensed", 0)
                if uncondensed > 10:
                    self._condenser.condense()
                    logger.info("Auto-condensed %d facts", uncondensed)
            except Exception as e:
                logger.debug("Auto-condense failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return the JSON schema for the ``enhanced_memory`` tool.

        Returns:
            list[dict]: Single-element list containing :data:`ENHANCED_MEMORY_SCHEMA`.
        """
        return [ENHANCED_MEMORY_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch an incoming tool call to the appropriate action handler.

        Args:
            tool_name: Must be ``'enhanced_memory'``.
            args: Tool arguments including ``action`` and action-specific keys.
            **kwargs: Reserved for future use by the framework.

        Returns:
            str: JSON-encoded result or error message.
        """
        if tool_name != "enhanced_memory":
            return tool_error(f"Unknown tool: {tool_name}")
        return self._handle_enhanced_memory(args)

    # -- Optional hooks -------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Extract facts from the conversation at session end.

        If ``auto_extract`` is enabled, runs regex-based fact extraction
        over user messages.  If ``auto_condense`` is also enabled and enough
        uncondensed facts exist, runs condensation and re-indexes for
        semantic search.

        Args:
            messages: The full conversation history (list of message dicts).
        """
        if not self._auto_extract or not self._store or not messages:
            return
        self._auto_extract_facts(messages)

        # Also condense if there are enough uncondensed facts
        if self._auto_condense:
            try:
                uncondensed = self._store.stats().get("raw_uncondensed", 0)
                if uncondensed > 5:
                    self._condenser.condense()
                    # Re-index for semantic search
                    if self._semantic:
                        self._index_new_facts()
            except Exception as e:
                logger.debug("Session-end condense failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in memory writes as raw facts in the enhanced store.

        Intercepts ``add`` actions from the standard memory system and
        stores the content as a raw fact so it is available to FTS and
        semantic search.

        Args:
            action: The memory action (only ``'add'`` is processed).
            target: Memory target (``'user'`` maps to ``user_pref`` category).
            content: The text being written.
            metadata: Optional metadata dict (currently unused).
        """
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_raw_fact(
                    content, category=category, source="memory_write",
                    session_id=self._session_id
                )
            except Exception as e:
                logger.debug("Memory write mirror failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract facts before context compression discards older messages.

        Called by the framework just before message history is truncated.
        Ensures valuable facts are preserved in the enhanced memory store
        before they would be lost.

        Args:
            messages: Messages about to be compressed/discarded.

        Returns:
            str: A status message indicating how many facts were extracted,
            or empty string if none were.
        """
        if not self._store or not messages:
            return ""
        count = self._auto_extract_facts(messages)
        if count > 0:
            return f"[Enhanced Memory extracted {count} facts before compression]"
        return ""

    def on_session_switch(self, new_session_id: str, *,
                          parent_session_id: str = "", reset: bool = False,
                          **kwargs) -> None:
        """Update session tracking when the session changes.

        Args:
            new_session_id: The new session identifier.
            parent_session_id: ID of the parent session (for sub-sessions).
            reset: If ``True``, reset the turn counter to zero.
            **kwargs: Reserved for future use.
        """
        self._session_id = new_session_id
        if reset:
            self._session_turns = 0

    def shutdown(self) -> None:
        """Clean shutdown — release all resources.

        Sets store, condenser, and semantic search references to ``None``
        so they can be garbage-collected.  Safe to call multiple times.
        """
        self._store = None
        self._condenser = None
        self._semantic = None

    # -- Tool handler ---------------------------------------------------------

    def _handle_enhanced_memory(self, args: dict) -> str:
        """Internal dispatcher for the ``enhanced_memory`` tool actions.

        Routes to the appropriate logic based on ``args['action']``:
        ``add``, ``search``, ``semantic_search``, ``condense``,
        ``list_condensed``, or ``stats``.

        Args:
            args: Tool arguments dict.  Must contain ``'action'``.

        Returns:
            str: JSON-encoded response or :func:`tool_error` string.
        """
        try:
            action = args["action"]

            if action == "add":
                content = args.get("content", "")
                if not content:
                    return tool_error("'content' is required for 'add' action")
                fact_id = self._store.add_raw_fact(
                    content,
                    category=args.get("category", "general"),
                    source=args.get("source", "dialog"),
                    session_id=self._session_id,
                )
                # Index for semantic search
                if self._semantic:
                    try:
                        self._semantic.index_facts(
                            [{"id": fact_id, "content": content}],
                            source_table="raw_facts"
                        )
                    except Exception:
                        pass
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                query = args.get("query", "")
                if not query:
                    return tool_error("'query' is required for 'search' action")
                limit = int(args.get("limit", 10))

                # Search both tiers
                raw_results = self._store.search_raw(query, limit=limit)
                condensed_results = self._store.search_condensed(
                    query=query, limit=limit
                )

                return json.dumps({
                    "raw_facts": raw_results,
                    "condensed": condensed_results,
                    "total": len(raw_results) + len(condensed_results),
                })

            elif action == "semantic_search":
                query = args.get("query", "")
                if not query:
                    return tool_error("'query' is required for 'semantic_search'")
                if not self._semantic:
                    return json.dumps({
                        "error": "Semantic search unavailable. Check: sqlite-vec installed, "
                                 "embedding provider configured, and required API key set.",
                        "fallback": "Use 'search' action for FTS5 keyword search instead."
                    })

                k = int(args.get("limit", 5))
                results = self._semantic.search(query, k=k)

                # Resolve fact content from IDs returned by vector search.
                enriched = []
                for r in results:
                    fact_id = r["fact_id"]
                    entry = {"distance": r["distance"], "similarity": r["similarity"]}

                    # Negative IDs below -10000 are condensed entries mapped
                    # via -(id + _CONDENSED_ID_OFFSET).  Reverse the mapping.
                    if fact_id < -10000:
                        real_id = -(fact_id + 10000)
                        c = self._store.get_condensed_by_id(real_id)
                        if c:
                            entry.update({
                                "source": "condensed", "id": real_id,
                                "topic": c["topic"], "content": c["summary"],
                                "category": c["category"], "priority": c["priority"],
                            })
                    else:
                        rf = self._store.get_raw_by_id(fact_id)
                        if rf:
                            entry.update({
                                "source": "raw_facts", "id": fact_id,
                                "content": rf["content"],
                                "category": rf["category"],
                            })

                    if "content" in entry:
                        enriched.append(entry)

                return json.dumps({"results": enriched, "count": len(enriched)})

            elif action == "condense":
                dry_run = bool(args.get("dry_run", False))
                entries = self._condenser.condense(dry_run=dry_run)

                # Re-index after condensation
                if not dry_run and self._semantic:
                    try:
                        self._index_new_facts()
                    except Exception:
                        pass

                return json.dumps({
                    "entries": [
                        {"topic": e["topic"], "category": e["category"],
                         "priority": e["priority"],
                         "fact_count": len(e.get("source_ids", [])),
                         "action": e.get("action", "unknown"),
                         "method": e.get("method", "unknown")}
                        for e in entries
                    ],
                    "count": len(entries),
                    "dry_run": dry_run,
                    "llm_available": self._condenser.llm_available,
                    "llm_model": self._condenser.llm_model,
                })

            elif action == "list_condensed":
                category = args.get("category")
                limit = int(args.get("limit", 20))
                results = self._store.search_condensed(
                    category=category, limit=limit
                )
                return json.dumps({"condensed": results, "count": len(results)})

            elif action == "stats":
                store_stats = self._store.stats()
                result = {
                    "raw_facts": store_stats.get("raw_facts", {}),
                    "condensed": store_stats.get("condensed", {}),
                    "semantic_search": {
                        "enabled": self._semantic is not None,
                        "stats": self._semantic.stats() if self._semantic else None,
                    },
                }
                return json.dumps(result)

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            logger.exception("Enhanced memory tool error")
            return tool_error(str(exc))

    # -- Auto-extraction from conversations -----------------------------------

    def _auto_extract_facts(self, messages: list) -> int:
        """Extract facts from conversation messages using regex pattern matching.

        Scans user messages for preference, decision, and environment patterns
        in both English and Russian.  Matched content is stored as raw facts
        with appropriate categories.

        Deduplication is performed by normalising the first 100 characters
        of each message; previously-seen content is skipped.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.

        Returns:
            int: Number of facts successfully extracted and stored.
        """
        # Preference patterns: "I prefer X", "my favorite X is Y", etc.
        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
            # Russian patterns
            re.compile(r'\bя\s+(?:предпочитаю|люблю|использую|хочу)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bмой\s+(?:любимый|предпочтительный)\s+\w+\s+(?:—|это)\s+(.+)', re.IGNORECASE),
        ]
        # Decision patterns: "we decided to X", "the project uses Y", etc.
        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
            # Russian
            re.compile(r'\bмы\s+(?:решили|выбрали|договорились)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bпроект\s+(?:использует|требует)\s+(.+)', re.IGNORECASE),
        ]
        # Environment patterns: version info, OS details, etc.
        _ENV_PATTERNS = [
            re.compile(r'\b(?:running|installed|configured|using)\s+(.+?\s+(?:version|v\d))', re.IGNORECASE),
            re.compile(r'\b(?:OS|server|machine)\s+(?:is|runs)\s+(.+)', re.IGNORECASE),
        ]

        extracted = 0
        seen = set()

        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue

            # Skip if we've seen very similar content
            norm = content[:100].lower().strip()
            if norm in seen:
                continue
            seen.add(norm)

            for pattern in _PREF_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_raw_fact(
                            content[:400], category="user_pref",
                            source="auto_extract", session_id=self._session_id
                        )
                        extracted += 1
                    except Exception:
                        pass
                    break

            for pattern in _DECISION_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_raw_fact(
                            content[:400], category="decision",
                            source="auto_extract", session_id=self._session_id
                        )
                        extracted += 1
                    except Exception:
                        pass
                    break

            for pattern in _ENV_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_raw_fact(
                            content[:400], category="env",
                            source="auto_extract", session_id=self._session_id
                        )
                        extracted += 1
                    except Exception:
                        pass
                    break

        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)
        return extracted

    # -- Semantic indexing helper ----------------------------------------------

    def _index_new_facts(self) -> None:
        """Index any unindexed facts for semantic vector search.

        Queries the semantic search engine for facts in both the ``raw_facts``
        and ``condensed`` tables that have not yet been embedded and indexed
        in the ``vec_memory`` virtual table.
        """
        if not self._semantic or not self._store:
            return
        try:
            unindexed = self._semantic.get_unindexed(self._store)
            raw = unindexed.get("raw_facts", [])
            condensed = unindexed.get("condensed", [])

            if raw:
                self._semantic.index_facts(raw, source_table="raw_facts")
            if condensed:
                self._semantic.index_facts(condensed, source_table="condensed")
        except Exception as e:
            logger.debug("Semantic indexing failed: %s", e)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Plugin entry point — register the enhanced memory provider.

    Called by the Hermes Agent plugin loader.  Creates an
    :class:`EnhancedMemoryProvider` with the loaded configuration and
    registers it as the active memory provider.

    Args:
        ctx: Plugin registration context exposing
            :meth:`register_memory_provider`.
    """
    config = _load_plugin_config()
    provider = EnhancedMemoryProvider(config=config)
    ctx.register_memory_provider(provider)

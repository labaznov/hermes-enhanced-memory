"""Enhanced Memory Plugin — Fact Condensation Engine (LLM + Algorithmic).

Two-tier condensation with LLM as primary method and algorithmic as fallback:

LLM Pipeline (primary)::

    raw_facts (uncondensed=0)
        │
        ▼
    group by category
        │
        ▼
    build prompt (facts + existing summaries)
        │
        ▼
    LLM call → structured JSON response
        │
        ▼
    parse, validate, upsert condensed entries
        │
        ▼
    mark source raw_facts as condensed=1

Algorithmic Pipeline (fallback when LLM unavailable)::

    raw_facts (uncondensed=0)
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
    upsert into condensed table
        │
        ▼
    mark source raw_facts as condensed=1

Usage::

    condenser = FactCondenser(store)
    results = condenser.condense()              # LLM-first, algorithmic fallback
    results = condenser.condense(dry_run=True)  # preview without writing
    results = condenser.condense(use_llm=False) # force algorithmic mode
    memory  = condenser.get_top_for_memory(char_limit=2200)

LLM Configuration:
    Set via ``condenser_model`` in plugin config, or auto-detected from
    environment (CONDENSER_MODEL, OPENAI_API_KEY / GOOGLE_API_KEY).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .store import EnhancedMemoryStore

logger = logging.getLogger("enhanced-memory.condenser")

# ── Topic display names (bilingual) ──────────────────────────────────────────
TOPIC_NAMES: dict[str, str] = {
    "user_pref": "Пользователь: предпочтения",
    "project": "Проекты и работа",
    "tool": "Инструменты и настройки",
    "env": "Среда и инфраструктура",
    "decision": "Решения и выборы",
    "security": "Безопасность",
    "general": "Общее",
}

# ── Base priority ranges per category ────────────────────────────────────────
_CATEGORY_PRIORITY: dict[str, tuple[int, int]] = {
    "security": (9, 10),
    "user_pref": (8, 9),
    "decision": (7, 9),
    "project": (7, 7),
    "tool": (6, 8),
    "env": (5, 5),
    "general": (4, 4),
}

# ── Keyword boost tables (used by algorithmic fallback) ──────────────────────
_BOOST_1_KEYWORDS: set[str] = {
    "prefers", "always", "never",
    "предпочитает", "всегда", "никогда",
}
_BOOST_2_KEYWORDS: set[str] = {
    "password", "key", "secret",
    "пароль", "ключ", "секрет",
}

_OVERLAP_THRESHOLD: float = 0.80

# ── LLM Condensation Prompts ────────────────────────────────────────────────

CONDENSATION_SYSTEM_PROMPT = """\
You are a Memory Condenser for a personal AI assistant. Your task: merge raw memory facts into compact, high-quality summaries grouped by category.

## Rules

1. **Merge & deduplicate**: Combine overlapping/redundant facts into concise statements. Remove duplicates.
2. **Contradictions**: When facts conflict, the LATER fact (higher index / listed lower) wins. Drop the outdated version entirely.
3. **Preserve identifiers verbatim**: IPs, ports, URLs, repo paths, API key names, server names, usernames, exact commands — copy them exactly. Never paraphrase technical identifiers.
4. **Bilingual**: User communicates in Russian and English. Keep the dominant language of each category's facts. Mix RU/EN naturally where the user does.
5. **Compact output**: Each category summary should be 1-5 sentences. All summaries combined MUST fit under 2000 characters total. Use abbreviations, semicolons, telegram-style brevity.
6. **If existing_summary is provided**: Update it with new facts. Don't repeat what's already there unless correcting it.
7. **Priority scoring**: Assign based on category range AND content importance within that range.

## Categories & Priority Ranges

- `security` (9-10): Server hardening, passwords, keys, firewall, auth
- `user_pref` (8-9): User preferences, habits, personal details, communication style
- `decision` (7-9): Architectural decisions, technology choices, rejected alternatives
- `project` (7-7): Active projects, tasks, deadlines
- `tool` (6-8): Tools, configs, providers, API keys
- `env` (5-5): Environment details, OS, infrastructure
- `general` (4-4): Miscellaneous

## Output Format

Respond with ONLY valid JSON, no markdown fences, no commentary:

{"results": [
  {
    "topic": "short label (2-4 words)",
    "category": "category_name",
    "summary": "condensed text with all key facts merged",
    "priority": 8
  }
]}

One entry per category that has facts. Skip categories with no facts.

## Example

INPUT:
{"security": {"facts": ["SSH key-only auth", "UFW active, 22/tcp", "fail2ban 24h ban", "IPv6 disabled", "kernel hardening on", "SSH passwords disabled"], "existing_summary": null},
"user_pref": {"facts": ["Alex — software engineer", "Prefers dark mode in all tools", "Communication style: concise, no fluff", "Uses metric system"], "existing_summary": null}}

OUTPUT:
{"results": [
  {"topic": "Server Hardening", "category": "security", "summary": "SSH key-only (passwords off), UFW active (22/tcp), fail2ban 24h, kernel hardening, IPv6 off.", "priority": 10},
  {"topic": "User Identity & Style", "category": "user_pref", "summary": "Alex — software engineer. Concise communication, no fluff. Dark mode preferred. Metric system.", "priority": 9}
]}"""

CONDENSATION_USER_TEMPLATE = """\
Condense the following memory facts. Return ONLY valid JSON.

{categories_json}"""


# ── Helpers (algorithmic) ────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Tokenise text into a set of lowercase words.

    Args:
        text: The input string.

    Returns:
        set[str]: Unique lowercase word tokens.
    """
    return set(re.findall(r"[\w]+", text.lower()))


def _word_overlap(a: str, b: str) -> float:
    """Compute the word overlap ratio between two strings.

    The ratio is ``|intersection| / min(|A|, |B|)`` — biased toward
    detecting subsets.

    Args:
        a: First string.
        b: Second string.

    Returns:
        float: Overlap ratio in ``[0.0, 1.0]``.
    """
    wa, wb = _tokenize(a), _tokenize(b)
    if not wa or not wb:
        return 0.0
    intersection = wa & wb
    smaller = min(len(wa), len(wb))
    return len(intersection) / smaller if smaller else 0.0


def _compute_priority(category: str, text: str) -> int:
    """Determine priority score based on category and keyword boosts.

    Args:
        category: The fact's category label.
        text: The fact text to scan for boost keywords.

    Returns:
        int: Priority score in the range ``[1, 10]``.
    """
    lo, _hi = _CATEGORY_PRIORITY.get(category, (4, 4))
    base = lo

    words_lower = text.lower()
    boost = 0
    if any(kw in words_lower for kw in _BOOST_2_KEYWORDS):
        boost += 2
    if any(kw in words_lower for kw in _BOOST_1_KEYWORDS):
        boost += 1

    priority = min(base + boost, 10)
    priority = max(priority, lo)
    return priority


def _merge_source_ids(existing: list[int] | str | None, new_ids: list[int]) -> str:
    """Merge two lists of source IDs, deduplicate, return JSON string.

    Args:
        existing: Current source IDs — list, JSON string, or None.
        new_ids: New IDs to merge in.

    Returns:
        str: JSON-encoded sorted list of unique IDs.
    """
    if existing is None:
        prev: list[int] = []
    elif isinstance(existing, str):
        try:
            prev = json.loads(existing)
        except (json.JSONDecodeError, TypeError):
            prev = []
    else:
        prev = list(existing)

    merged = sorted(set(prev) | set(new_ids))
    return json.dumps(merged)


# ── LLM Client ──────────────────────────────────────────────────────────────

class _LLMClient:
    """Lightweight LLM client for condensation calls.

    Supports OpenAI-compatible APIs (vLLM, Ollama, LiteLLM, etc.)
    and Google Gemini via google-generativeai SDK.

    Auto-detects configuration from:
    1. Explicit config dict passed to FactCondenser
    2. Environment variables (CONDENSER_MODEL, OPENAI_API_KEY, GOOGLE_API_KEY)

    Args:
        config: Optional dict with keys ``model``, ``api_key``, ``base_url``,
            ``provider``. If not provided, auto-detects from environment.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._provider: str | None = None
        self._model: str | None = None
        self._api_key: str | None = None
        self._base_url: str | None = None
        self._client: Any = None
        self._available: bool | None = None

    def _detect_config(self) -> bool:
        """Auto-detect LLM configuration from config dict or environment.

        Returns:
            bool: True if a usable LLM configuration was found.
        """
        # Priority 1: explicit config
        if self._config.get("model"):
            self._model = self._config["model"]
            self._api_key = self._config.get("api_key", "")
            self._base_url = self._config.get("base_url", "")
            self._provider = self._config.get("provider", "openai")
            return True

        # Priority 2: CONDENSER_MODEL env var
        model = os.environ.get("CONDENSER_MODEL")
        if model:
            self._model = model
            # Try to detect provider from model name
            if "gemini" in model.lower():
                self._provider = "google"
                self._api_key = os.environ.get("GOOGLE_API_KEY", "")
            else:
                self._provider = "openai"
                self._api_key = os.environ.get("OPENAI_API_KEY", "")
                self._base_url = os.environ.get("OPENAI_BASE_URL", "")
            return True

        # Priority 3: check available API keys
        # Prefer Gemini Flash (cheap, fast) for condensation
        google_key = os.environ.get("GOOGLE_API_KEY")
        if google_key:
            self._provider = "google"
            self._model = "gemini-2.5-flash"
            self._api_key = google_key
            return True

        # Fallback: OpenAI-compatible
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self._provider = "openai"
            self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self._api_key = openai_key
            self._base_url = os.environ.get("OPENAI_BASE_URL", "")
            return True

        return False

    @property
    def available(self) -> bool:
        """Whether an LLM client can be initialized."""
        if self._available is None:
            self._available = self._detect_config()
        return self._available

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return self._model or "none"

    def call(self, system_prompt: str, user_message: str) -> str | None:
        """Make a single LLM call with system + user messages.

        Args:
            system_prompt: The system instructions.
            user_message: The user message content.

        Returns:
            str | None: The model's response text, or None on failure.
        """
        if not self.available:
            logger.warning("LLM client not available for condensation.")
            return None

        try:
            if self._provider == "google":
                return self._call_google(system_prompt, user_message)
            else:
                return self._call_openai(system_prompt, user_message)
        except Exception:
            logger.exception("LLM condensation call failed (model=%s).", self._model)
            return None

    def _call_openai(self, system_prompt: str, user_message: str) -> str | None:
        """Call OpenAI-compatible API.

        Args:
            system_prompt: System instructions.
            user_message: User message content.

        Returns:
            str | None: Response text or None.
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai package not installed. pip install openai")
            return None

        if self._client is None:
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        content = response.choices[0].message.content
        return content.strip() if content else None

    def _call_google(self, system_prompt: str, user_message: str) -> str | None:
        """Call Google Gemini API.

        Args:
            system_prompt: System instructions.
            user_message: User message content.

        Returns:
            str | None: Response text or None.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            logger.error("google-genai package not installed. pip install google-genai")
            return None

        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)

        response = self._client.models.generate_content(
            model=self._model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=4000,
            ),
        )
        return response.text.strip() if response.text else None


# ── JSON parsing helpers ─────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract JSON from LLM response, handling markdown fences.

    Tries multiple strategies:
    1. Direct json.loads
    2. Strip markdown code fences
    3. Find JSON object pattern in text

    Args:
        text: Raw LLM response text.

    Returns:
        dict | None: Parsed JSON dict, or None on failure.
    """
    if not text:
        return None

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find JSON object in text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
    return None


def _validate_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize parsed LLM condensation results.

    Ensures each result has required fields with correct types.

    Args:
        data: Parsed JSON dict from LLM response.

    Returns:
        list[dict]: List of validated result entries.
    """
    results = data.get("results", [])
    if not isinstance(results, list):
        return []

    validated = []
    for item in results:
        if not isinstance(item, dict):
            continue

        category = item.get("category", "")
        summary = item.get("summary", "")
        if not category or not summary:
            continue

        # Normalize priority to valid range
        try:
            priority = int(item.get("priority", 5))
        except (ValueError, TypeError):
            priority = 5
        lo, _hi = _CATEGORY_PRIORITY.get(category, (4, 4))
        priority = max(lo, min(priority, 10))

        # Use provided topic or generate from category
        topic = item.get("topic", TOPIC_NAMES.get(category, category))

        validated.append({
            "topic": str(topic),
            "category": str(category),
            "summary": str(summary).strip(),
            "priority": priority,
        })

    return validated


# ── Main condenser class ────────────────────────────────────────────────────


class FactCondenser:
    """Groups, deduplicates, and summarises raw facts into condensed entries.

    Uses LLM as primary condensation method (smarter merging, contradiction
    resolution, natural language summaries) with algorithmic fallback when
    LLM is unavailable.

    Attributes:
        store (EnhancedMemoryStore): The backing store instance.

    Args:
        store: An initialised :class:`EnhancedMemoryStore`.
        llm_config: Optional dict with LLM configuration:
            ``model``, ``api_key``, ``base_url``, ``provider``.

    Example::

        condenser = FactCondenser(store)
        results = condenser.condense()              # LLM + fallback
        results = condenser.condense(use_llm=False)  # force algorithmic
        results = condenser.condense(dry_run=True)   # preview only
        memory_text = condenser.get_top_for_memory()
    """

    def __init__(
        self,
        store: "EnhancedMemoryStore",
        llm_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.store = store
        self._llm = _LLMClient(llm_config)

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def llm_available(self) -> bool:
        """Whether LLM condensation is available."""
        return self._llm.available

    @property
    def llm_model(self) -> str:
        """Current LLM model name."""
        return self._llm.model_name

    def condense(
        self,
        dry_run: bool = False,
        use_llm: bool = True,
    ) -> list[dict[str, Any]]:
        """Run the condensation pipeline.

        Attempts LLM-based condensation first (if ``use_llm=True`` and LLM
        is available). Falls back to algorithmic condensation on failure.

        Args:
            dry_run: If *True*, compute results but do **not** write to DB.
            use_llm: If *True* (default), try LLM condensation first.

        Returns:
            List of dicts with keys ``topic``, ``category``, ``summary``,
            ``priority``, ``source_ids``, ``action``, ``method``.
        """
        raw_facts = self._load_uncondensed()
        if not raw_facts:
            logger.info("No uncondensed facts to process.")
            return []

        grouped = self._group_by_category(raw_facts)

        # Collect all source IDs for marking
        all_source_ids: list[int] = []
        for facts in grouped.values():
            for fact in facts:
                fid = fact.get("id")
                if fid is not None:
                    all_source_ids.append(int(fid))

        results: list[dict[str, Any]] = []
        method_used = "algorithmic"

        # Try LLM condensation
        if use_llm and self._llm.available:
            llm_results = self._condense_with_llm(grouped)
            if llm_results:
                method_used = "llm"
                for entry in llm_results:
                    # Attach source IDs from the category
                    cat = entry["category"]
                    cat_source_ids = []
                    for fact in grouped.get(cat, []):
                        fid = fact.get("id")
                        if fid is not None:
                            cat_source_ids.append(int(fid))
                    entry["source_ids"] = cat_source_ids
                    entry["method"] = "llm"

                    if not dry_run:
                        action = self._upsert_condensed_replace(entry)
                        entry["action"] = action
                    else:
                        entry["action"] = "dry_run"

                    results.append(entry)
            else:
                logger.warning(
                    "LLM condensation failed, falling back to algorithmic."
                )

        # Algorithmic fallback
        if not results:
            results = self._condense_algorithmic(grouped, dry_run)
            method_used = "algorithmic"

        # Mark originals as condensed
        if not dry_run and all_source_ids:
            self._mark_condensed(all_source_ids)

        logger.info(
            "Condensation complete (%s): %d entries %s.",
            method_used,
            len(results),
            "previewed (dry-run)" if dry_run else "written",
        )
        return results

    def get_top_for_memory(self, char_limit: int = 2200) -> str:
        """Return a compact string of the highest-priority condensed entries.

        Entries are sorted by ``priority DESC, updated_at DESC`` and
        concatenated with the ``§`` separator until *char_limit* is reached.

        Args:
            char_limit: Maximum character count for the returned string.

        Returns:
            str: A ``§``-separated string of condensed summaries.
        """
        entries = self._load_condensed_sorted()
        parts: list[str] = []
        current_len = 0
        separator = "§"
        sep_len = len(separator)

        for entry in entries:
            summary = entry.get("summary", "").strip()
            if not summary:
                continue

            addition_len = len(summary) + (sep_len if parts else 0)
            if current_len + addition_len > char_limit:
                if not parts:
                    parts.append(summary[: char_limit])
                break

            parts.append(summary)
            current_len += addition_len

        return separator.join(parts)

    # ── LLM condensation ─────────────────────────────────────────────────

    def _condense_with_llm(
        self, grouped: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]] | None:
        """Run LLM-based condensation on grouped facts.

        Builds a prompt with current facts and existing summaries,
        sends to LLM, parses structured JSON response.

        Args:
            grouped: Category → list of fact dicts.

        Returns:
            list[dict] | None: Validated condensation results, or None on
            failure (triggering algorithmic fallback).
        """
        # Build categories payload for the prompt
        categories_payload: dict[str, dict[str, Any]] = {}

        for category, facts in grouped.items():
            fact_texts = []
            for fact in facts:
                text = fact.get("content", fact.get("text", "")).strip()
                if text:
                    fact_texts.append(text)

            if not fact_texts:
                continue

            # Load existing condensed summary for this category
            existing_summary = self._get_existing_summary(category)

            categories_payload[category] = {
                "facts": fact_texts,
                "existing_summary": existing_summary,
            }

        if not categories_payload:
            return None

        # Format user message
        categories_json = json.dumps(
            categories_payload, ensure_ascii=False, indent=2
        )
        user_message = CONDENSATION_USER_TEMPLATE.format(
            categories_json=categories_json
        )

        logger.info(
            "Calling LLM (%s) for condensation: %d categories, %d total facts.",
            self._llm.model_name,
            len(categories_payload),
            sum(len(v["facts"]) for v in categories_payload.values()),
        )

        # Make LLM call
        response = self._llm.call(CONDENSATION_SYSTEM_PROMPT, user_message)
        if not response:
            return None

        # Parse and validate response
        parsed = _extract_json(response)
        if not parsed:
            return None

        results = _validate_results(parsed)
        if not results:
            logger.warning("LLM returned no valid results.")
            return None

        logger.info(
            "LLM condensation produced %d entries.", len(results)
        )
        return results

    def _get_existing_summary(self, category: str) -> str | None:
        """Load current condensed summary for a category.

        Args:
            category: Category label (e.g. 'security', 'user_pref').

        Returns:
            str | None: Existing summary text, or None if not found.
        """
        try:
            conn = self.store.get_connection()
            cursor = conn.execute(
                "SELECT summary FROM condensed WHERE category = ? "
                "ORDER BY priority DESC LIMIT 1",
                (category,),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception:
            logger.exception("Failed to load existing summary for %s.", category)
            return None

    # ── Algorithmic condensation (fallback) ──────────────────────────────

    def _condense_algorithmic(
        self,
        grouped: dict[str, list[dict[str, Any]]],
        dry_run: bool,
    ) -> list[dict[str, Any]]:
        """Algorithmic condensation: dedup + priority boost + join.

        Args:
            grouped: Category → list of fact dicts.
            dry_run: If True, skip DB writes.

        Returns:
            list[dict]: Condensation results.
        """
        results: list[dict[str, Any]] = []

        for category, facts in grouped.items():
            deduplicated = self._deduplicate(facts)
            topic = TOPIC_NAMES.get(category, TOPIC_NAMES["general"])

            source_ids: list[int] = []
            summaries: list[str] = []
            for fact in deduplicated:
                fid = fact.get("id")
                if fid is not None:
                    source_ids.append(int(fid))
                text = fact.get("content", fact.get("text", "")).strip()
                if text:
                    summaries.append(text)

            if not summaries:
                continue

            summary = "; ".join(summaries)
            priority = max(
                (_compute_priority(category, s) for s in summaries),
                default=_CATEGORY_PRIORITY.get(category, (4, 4))[0],
            )

            entry: dict[str, Any] = {
                "topic": topic,
                "category": category,
                "summary": summary,
                "priority": priority,
                "source_ids": source_ids,
                "method": "algorithmic",
            }

            if not dry_run:
                action = self._upsert_condensed_append(entry)
                entry["action"] = action
            else:
                entry["action"] = "dry_run"

            results.append(entry)

        return results

    # ── Internal helpers ─────────────────────────────────────────────────

    def _load_uncondensed(self) -> list[dict[str, Any]]:
        """Fetch raw facts where ``condensed = 0``.

        Returns:
            list[dict]: Facts with keys ``id``, ``content``, ``category``,
            ``created_at``.
        """
        try:
            conn = self.store.get_connection()
            cursor = conn.execute(
                "SELECT id, content, category, created_at "
                "FROM raw_facts WHERE condensed = 0 ORDER BY created_at ASC"
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            logger.exception("Failed to load uncondensed facts.")
            return []

    def _group_by_category(
        self, facts: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group facts by their ``category`` field.

        Args:
            facts: List of fact dicts.

        Returns:
            dict[str, list[dict]]: Category → list of facts.
        """
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fact in facts:
            cat = fact.get("category", "general") or "general"
            grouped[cat].append(fact)
        return dict(grouped)

    def _deduplicate(
        self, facts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove near-duplicates using word-overlap threshold.

        Args:
            facts: Flat list of fact dicts within one category.

        Returns:
            list[dict]: Deduplicated facts in original order.
        """
        accepted: list[dict[str, Any]] = []
        accepted_texts: list[str] = []

        for fact in facts:
            text = fact.get("content", fact.get("text", "")).strip()
            if not text:
                continue

            is_dup = False
            for existing_text in accepted_texts:
                if _word_overlap(text, existing_text) >= _OVERLAP_THRESHOLD:
                    is_dup = True
                    break

            if not is_dup:
                accepted.append(fact)
                accepted_texts.append(text)

        removed = len(facts) - len(accepted)
        if removed:
            logger.debug("Deduplicated %d/%d facts.", removed, len(facts))
        return accepted

    def _upsert_condensed_replace(self, entry: dict[str, Any]) -> str:
        """Insert or REPLACE condensed entry (LLM mode).

        In LLM mode, the summary is a complete replacement since the LLM
        has already merged existing + new facts.

        Args:
            entry: Dict with ``topic``, ``category``, ``summary``,
                ``priority``, ``source_ids``.

        Returns:
            str: ``'created'`` or ``'updated'``.
        """
        conn = self.store.get_connection()
        now = int(time.time())

        cursor = conn.execute(
            "SELECT id, source_ids FROM condensed "
            "WHERE category = ?",
            (entry["category"],),
        )
        row = cursor.fetchone()

        source_ids_json = json.dumps(entry.get("source_ids", []))

        if row:
            existing_id, existing_source_ids = row
            merged_ids = _merge_source_ids(existing_source_ids, entry.get("source_ids", []))
            conn.execute(
                "UPDATE condensed SET topic = ?, summary = ?, priority = ?, "
                "source_ids = ?, updated_at = ? WHERE id = ?",
                (
                    entry["topic"],
                    entry["summary"],
                    entry["priority"],
                    merged_ids,
                    now,
                    existing_id,
                ),
            )
            conn.commit()
            logger.debug("Replaced condensed entry id=%d cat=%r.", existing_id, entry["category"])
            return "updated"
        else:
            conn.execute(
                "INSERT INTO condensed (topic, category, summary, priority, "
                "source_ids, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry["topic"],
                    entry["category"],
                    entry["summary"],
                    entry["priority"],
                    source_ids_json,
                    now,
                    now,
                ),
            )
            conn.commit()
            logger.debug("Created condensed entry cat=%r.", entry["category"])
            return "created"

    def _upsert_condensed_append(self, entry: dict[str, Any]) -> str:
        """Insert or APPEND condensed entry (algorithmic mode).

        In algorithmic mode, new summaries are appended to existing ones
        separated by ``"; "`` since the algorithm can't rewrite text.

        Args:
            entry: Dict with ``topic``, ``category``, ``summary``,
                ``priority``, ``source_ids``.

        Returns:
            str: ``'created'`` or ``'updated'``.
        """
        conn = self.store.get_connection()
        now = int(time.time())

        cursor = conn.execute(
            "SELECT id, source_ids FROM condensed "
            "WHERE topic = ? AND category = ?",
            (entry["topic"], entry["category"]),
        )
        row = cursor.fetchone()

        if row:
            existing_id, existing_source_ids = row
            merged_ids = _merge_source_ids(existing_source_ids, entry["source_ids"])
            cursor2 = conn.execute(
                "SELECT summary, priority FROM condensed WHERE id = ?",
                (existing_id,),
            )
            old_summary, old_priority = cursor2.fetchone()
            new_summary = (
                f"{old_summary}; {entry['summary']}" if old_summary else entry["summary"]
            )
            final_priority = max(old_priority or 0, entry["priority"])

            conn.execute(
                "UPDATE condensed SET summary = ?, priority = ?, "
                "source_ids = ?, updated_at = ? WHERE id = ?",
                (new_summary, final_priority, merged_ids, now, existing_id),
            )
            conn.commit()
            logger.debug("Appended to condensed entry id=%d.", existing_id)
            return "updated"
        else:
            source_ids_json = json.dumps(entry["source_ids"])
            conn.execute(
                "INSERT INTO condensed (topic, category, summary, priority, "
                "source_ids, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry["topic"],
                    entry["category"],
                    entry["summary"],
                    entry["priority"],
                    source_ids_json,
                    now,
                    now,
                ),
            )
            conn.commit()
            logger.debug("Created condensed entry topic=%r.", entry["topic"])
            return "created"

    def _mark_condensed(self, fact_ids: list[int]) -> None:
        """Set ``condensed = 1`` on the given raw_fact IDs.

        Args:
            fact_ids: List of ``raw_facts.id`` values.
        """
        if not fact_ids:
            return
        try:
            conn = self.store.get_connection()
            placeholders = ", ".join("?" for _ in fact_ids)
            conn.execute(
                f"UPDATE raw_facts SET condensed = 1 WHERE id IN ({placeholders})",
                fact_ids,
            )
            conn.commit()
            logger.debug("Marked %d raw facts as condensed.", len(fact_ids))
        except Exception:
            logger.exception("Failed to mark facts as condensed.")

    def _load_condensed_sorted(self) -> list[dict[str, Any]]:
        """Load all condensed entries ordered by priority descending.

        Returns:
            list[dict]: All condensed rows as dicts.
        """
        try:
            conn = self.store.get_connection()
            cursor = conn.execute(
                "SELECT id, topic, category, summary, priority, source_ids, "
                "created_at, updated_at FROM condensed "
                "ORDER BY priority DESC, updated_at DESC"
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            logger.exception("Failed to load condensed entries.")
            return []

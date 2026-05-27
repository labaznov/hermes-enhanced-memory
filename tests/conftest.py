"""Shared fixtures for the enhanced-memory-plugin test suite."""
from __future__ import annotations

import sys
import os

# Add project root and enhanced_memory package to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PACKAGE_DIR = os.path.join(_PROJECT_ROOT, "enhanced_memory")

for p in (_PACKAGE_DIR, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Also add hermes-agent if installed (for MemoryProvider ABC import in __init__.py)
_HERMES_ROOT = "/usr/local/lib/hermes-agent"
if os.path.isdir(_HERMES_ROOT) and _HERMES_ROOT not in sys.path:
    sys.path.insert(0, _HERMES_ROOT)

import pytest

from store import EnhancedMemoryStore
from condenser import FactCondenser


@pytest.fixture
def db_path(tmp_path):
    """Return a path to a temporary database file."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def store(db_path):
    """Create a fresh EnhancedMemoryStore backed by a temp database."""
    s = EnhancedMemoryStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def condenser(store):
    """Create a FactCondenser bound to the temp store."""
    return FactCondenser(store)


@pytest.fixture
def populated_store(store):
    """Store pre-populated with a handful of raw facts across categories."""
    facts = [
        {"content": "User prefers dark mode in all editors", "category": "user_pref"},
        {"content": "User always uses vim keybindings", "category": "user_pref"},
        {"content": "The project uses Python 3.12", "category": "project"},
        {"content": "Server runs Ubuntu 22.04 LTS", "category": "env"},
        {"content": "API key stored in vault", "category": "security"},
        {"content": "We decided to use PostgreSQL", "category": "decision"},
        {"content": "Linting uses ruff with strict config", "category": "tool"},
        {"content": "General note about meeting schedule", "category": "general"},
    ]
    store.add_raw_facts_batch(facts)
    return store

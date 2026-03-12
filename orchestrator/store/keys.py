"""
Keys store — read/write for data/setup/keys.json.

This is the ONLY place key values are persisted to disk.
All other store files reference models/servers by id — never by key value.

The scrub() function must be called before writing any log, transcript,
or result file to ensure key values are never captured in outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import setup_dir


def _keys_path() -> Path:
    return setup_dir() / "keys.json"


def _load() -> dict:
    path = _keys_path()
    if not path.exists():
        return {"models": {}, "search": {"provider": "brave", "api_key": "", "enabled": True}}
    return json.loads(path.read_text())


def _save(data: dict) -> None:
    path = _keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ── Model keys ────────────────────────────────────────────────────────────────

def get_model_key(model_id: str) -> str | None:
    """Return the API key for a model, or None if not set."""
    return _load()["models"].get(model_id)


def set_model_key(model_id: str, key: str) -> None:
    """Store the API key for a model."""
    data = _load()
    data["models"][model_id] = key
    _save(data)


def delete_model_key(model_id: str) -> None:
    """Remove the API key for a model (called when the model is deleted)."""
    data = _load()
    data["models"].pop(model_id, None)
    _save(data)


# ── Search config ─────────────────────────────────────────────────────────────

def get_search_config() -> dict:
    """Return search configuration: provider, api_key, enabled."""
    return _load().get("search", {"provider": "brave", "api_key": "", "enabled": True})


def set_search_config(provider: str, api_key: str, enabled: bool) -> None:
    """Store search configuration."""
    data = _load()
    data["search"] = {"provider": provider, "api_key": api_key, "enabled": enabled}
    _save(data)


# ── Scrubber ──────────────────────────────────────────────────────────────────

def scrub(text: str) -> str:
    """
    Replace all known key values in text with [REDACTED].
    Call this before writing any log, transcript, or result file.
    Keys shorter than 8 characters are skipped to avoid false positives.
    """
    data = _load()

    keys_to_scrub = []
    for key_value in data.get("models", {}).values():
        if key_value and len(key_value) >= 8:
            keys_to_scrub.append(key_value)

    search_key = data.get("search", {}).get("api_key", "")
    if search_key and len(search_key) >= 8:
        keys_to_scrub.append(search_key)

    for key_value in keys_to_scrub:
        text = text.replace(key_value, "[REDACTED]")

    return text

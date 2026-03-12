"""
Store layer tests — covers all CRUD operations and key safety rules.
"""

import json
import os
import pytest
import tempfile


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path):
    """Each test gets a fresh temporary data directory."""
    os.environ["DATA_DIR"] = str(tmp_path)
    # Re-import store modules so they pick up the new DATA_DIR
    import importlib
    import store.base
    import store.models
    import store.prompts
    import store.mcp
    import store.experiments
    import store.keys
    for mod in [store.base, store.models, store.prompts, store.mcp, store.experiments, store.keys]:
        importlib.reload(mod)
    import store
    importlib.reload(store)
    yield tmp_path
    del os.environ["DATA_DIR"]


import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Model store ───────────────────────────────────────────────────────────────

def test_model_crud():
    from store.models import create_model, get_model, update_model, delete_model, list_models

    entry = create_model(
        name="GLM-5",
        model_identifier="glm-5:cloud",
        endpoint_url="https://api.ollama.com",
        context_window=198000,
    )
    assert entry.name == "GLM-5"
    assert entry.model_identifier == "glm-5:cloud"
    assert entry.context_window == 198000

    fetched = get_model(entry.id)
    assert fetched is not None
    assert fetched.id == entry.id

    updated = update_model(entry.id, name="GLM-5 Updated")
    assert updated.name == "GLM-5 Updated"
    assert updated.updated_at >= entry.updated_at

    entries = list_models()
    assert len(entries) == 1

    assert delete_model(entry.id) is True
    assert get_model(entry.id) is None
    assert delete_model(entry.id) is False


def test_model_file_contains_no_key(isolated_data_dir):
    """Key values must never appear in model JSON files."""
    from store.models import create_model
    from store.keys import set_model_key

    entry = create_model(
        name="Test Model",
        model_identifier="qwen3-coder-next:cloud",
        endpoint_url="https://api.ollama.com",
        context_window=256000,
    )
    set_model_key(entry.id, "super-secret-api-key-12345")

    model_file = isolated_data_dir / "setup" / "models" / f"{entry.id}.json"
    content = model_file.read_text()
    assert "super-secret-api-key-12345" not in content


# ── Prompt store ──────────────────────────────────────────────────────────────

def test_prompt_crud():
    from store.prompts import create_prompt, get_prompt, update_prompt, delete_prompt, list_prompts

    entry = create_prompt(name="Default Interviewer", content="You are an interviewer...")
    assert entry.name == "Default Interviewer"
    assert "interviewer" in entry.content

    fetched = get_prompt(entry.id)
    assert fetched.content == entry.content

    updated = update_prompt(entry.id, content="Updated content.")
    assert updated.content == "Updated content."

    assert len(list_prompts()) == 1
    assert delete_prompt(entry.id) is True
    assert list_prompts() == []


# ── MCP store ─────────────────────────────────────────────────────────────────

def test_mcp_crud():
    from store.mcp import create_mcp_server, get_mcp_server, update_mcp_server, delete_mcp_server, list_mcp_servers

    entry = create_mcp_server(
        name="My MCP Server",
        endpoint_url="http://localhost:9000/sse",
        auth_type="bearer",
    )
    assert entry.auth_type == "bearer"
    assert entry.endpoint_url == "http://localhost:9000/sse"

    fetched = get_mcp_server(entry.id)
    assert fetched.id == entry.id

    updated = update_mcp_server(entry.id, name="Updated Name")
    assert updated.name == "Updated Name"

    assert len(list_mcp_servers()) == 1
    assert delete_mcp_server(entry.id) is True
    assert get_mcp_server(entry.id) is None


# ── Experiment store ──────────────────────────────────────────────────────────

def test_experiment_crud():
    from store.experiments import create_experiment, get_experiment, update_experiment, delete_experiment, list_experiments
    from store.models import create_model

    m1 = create_model("Pass1 Model", "qwen3-coder-next:cloud", "https://api.ollama.com", 256000)
    m2 = create_model("Interviewer", "glm-5:cloud", "https://api.ollama.com", 198000)

    exp = create_experiment(
        name="Baseline Test",
        pass1_model_id=m1.id,
        pass2_model_id=m1.id,
        interviewer_model_id=m2.id,
        turn_limit=100,
    )
    assert exp.name == "Baseline Test"
    assert exp.turn_limit == 100
    assert exp.pass1_model_id == m1.id

    fetched = get_experiment(exp.id)
    assert fetched.id == exp.id

    updated = update_experiment(exp.id, turn_limit=50)
    assert updated.turn_limit == 50

    assert len(list_experiments()) == 1
    assert delete_experiment(exp.id) is True
    assert get_experiment(exp.id) is None


def test_experiment_defaults_are_none():
    """Unset override fields must be None so global config.yaml values are used."""
    from store.experiments import create_experiment
    from store.models import create_model

    m = create_model("M", "m:cloud", "https://api.ollama.com", 256000)
    exp = create_experiment("Test", m.id, m.id, m.id)

    assert exp.turn_limit is None
    assert exp.context_window is None
    assert exp.compaction_threshold_fraction is None
    assert exp.search_enabled is None


# ── Keys store ────────────────────────────────────────────────────────────────

def test_keys_set_and_get():
    from store.keys import set_model_key, get_model_key

    assert get_model_key("nonexistent-id") is None
    set_model_key("model-abc", "my-api-key-xyz-99")
    assert get_model_key("model-abc") == "my-api-key-xyz-99"


def test_search_config():
    from store.keys import set_search_config, get_search_config

    set_search_config(provider="brave", api_key="brave-key-12345", enabled=True)
    cfg = get_search_config()
    assert cfg["provider"] == "brave"
    assert cfg["api_key"] == "brave-key-12345"
    assert cfg["enabled"] is True


def test_scrubber_replaces_model_key():
    from store.keys import set_model_key, scrub

    set_model_key("model-xyz", "top-secret-key-abc123")
    result = scrub("The api_key is top-secret-key-abc123 and should be hidden.")
    assert "top-secret-key-abc123" not in result
    assert "[REDACTED]" in result


def test_scrubber_replaces_search_key():
    from store.keys import set_search_config, scrub

    set_search_config("brave", "brave-secret-key-xyz", True)
    result = scrub("Authorization: Bearer brave-secret-key-xyz")
    assert "brave-secret-key-xyz" not in result
    assert "[REDACTED]" in result


def test_scrubber_skips_short_strings():
    """Keys shorter than 8 chars are not scrubbed to avoid false positives."""
    from store.keys import set_model_key, scrub

    set_model_key("model-short", "abc")
    result = scrub("This has abc in it.")
    assert "abc" in result  # Not scrubbed — too short

"""
Router tests — HTTP endpoints for all stores and config.
Key masking is verified throughout: key values must never appear in responses.
"""

import os
import sys
import importlib
import pytest
import yaml
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Each test gets isolated data dir and a temporary config file."""
    os.environ["DATA_DIR"] = str(tmp_path)

    # Write a minimal config.yaml for the config router tests
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text(yaml.dump({
        "experiment": {
            "turn_limit": 200,
            "context_window": 256000,
            "compaction_threshold_fraction": 0.80,
            "pass1_activation_fraction": 0.50,
            "turn_pause_seconds": 5,
        }
    }))
    os.environ["CONFIG_PATH"] = str(config_file)

    # Reload all modules so env vars take effect
    mods = [m for m in sys.modules if m.startswith("store") or m.startswith("routers")]
    for mod in mods:
        del sys.modules[mod]

    yield tmp_path

    del os.environ["DATA_DIR"]
    del os.environ["CONFIG_PATH"]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main
    importlib.reload(main)
    return TestClient(main.app)


# ── Models ────────────────────────────────────────────────────────────────────

def test_model_crud_http(client):
    # Create
    resp = client.post("/api/models", json={
        "name": "GLM-5",
        "model_identifier": "glm-5:cloud",
        "endpoint_url": "https://api.ollama.com",
        "context_window": 198000,
    })
    assert resp.status_code == 201
    model_id = resp.json()["id"]
    assert resp.json()["name"] == "GLM-5"

    # Read
    resp = client.get(f"/api/models/{model_id}")
    assert resp.status_code == 200

    # List
    resp = client.get("/api/models")
    assert len(resp.json()) == 1

    # Update
    resp = client.patch(f"/api/models/{model_id}", json={"name": "GLM-5 Updated"})
    assert resp.json()["name"] == "GLM-5 Updated"

    # Delete
    resp = client.delete(f"/api/models/{model_id}")
    assert resp.status_code == 204
    assert client.get(f"/api/models/{model_id}").status_code == 404


def test_model_key_never_returned(client):
    resp = client.post("/api/models", json={
        "name": "Secret Model",
        "model_identifier": "qwen3:cloud",
        "endpoint_url": "https://api.ollama.com",
        "context_window": 256000,
        "api_key": "ultra-secret-key-xyz-99999",
    })
    model_id = resp.json()["id"]

    # The model response must not contain the key
    assert "ultra-secret-key-xyz-99999" not in resp.text

    # The key status endpoint returns boolean only
    status_resp = client.get(f"/api/models/{model_id}/key/status")
    assert status_resp.json() == {"has_key": True}
    assert "ultra-secret-key-xyz-99999" not in status_resp.text

    # Setting key via PUT also must not echo it back
    put_resp = client.put(f"/api/models/{model_id}/key", json={"key": "another-secret-key-abc"})
    assert put_resp.status_code == 204


# ── Prompts ───────────────────────────────────────────────────────────────────

def test_prompt_crud_http(client):
    resp = client.post("/api/prompts", json={
        "name": "Custom Interviewer",
        "content": "You are an interviewer...",
    })
    assert resp.status_code == 201
    prompt_id = resp.json()["id"]

    assert client.get(f"/api/prompts/{prompt_id}").status_code == 200
    assert len(client.get("/api/prompts").json()) == 1

    resp = client.patch(f"/api/prompts/{prompt_id}", json={"content": "Updated."})
    assert resp.json()["content"] == "Updated."

    assert client.delete(f"/api/prompts/{prompt_id}").status_code == 204
    assert client.get(f"/api/prompts/{prompt_id}").status_code == 404


# ── MCP Servers ───────────────────────────────────────────────────────────────

def test_mcp_server_crud_http(client):
    resp = client.post("/api/mcp-servers", json={
        "name": "My MCP",
        "endpoint_url": "http://localhost:9000/sse",
        "auth_type": "bearer",
        "api_key": "mcp-secret-key-12345",
    })
    assert resp.status_code == 201
    server_id = resp.json()["id"]
    assert "mcp-secret-key-12345" not in resp.text

    status_resp = client.get(f"/api/mcp-servers/{server_id}/key/status")
    assert status_resp.json()["has_key"] is True
    assert "mcp-secret-key-12345" not in status_resp.text

    assert client.delete(f"/api/mcp-servers/{server_id}").status_code == 204


# ── Experiments ───────────────────────────────────────────────────────────────

def test_experiment_crud_http(client):
    # Create models first
    m1 = client.post("/api/models", json={
        "name": "Pass1", "model_identifier": "qwen3:cloud",
        "endpoint_url": "https://api.ollama.com", "context_window": 256000,
    }).json()
    m2 = client.post("/api/models", json={
        "name": "Interviewer", "model_identifier": "glm-5:cloud",
        "endpoint_url": "https://api.ollama.com", "context_window": 198000,
    }).json()

    resp = client.post("/api/experiments", json={
        "name": "Test Run Alpha",
        "pass1_model_id": m1["id"],
        "pass2_model_id": m1["id"],
        "interviewer_model_id": m2["id"],
        "turn_limit": 50,
    })
    assert resp.status_code == 201
    exp_id = resp.json()["id"]
    assert resp.json()["turn_limit"] == 50

    updated = client.patch(f"/api/experiments/{exp_id}", json={"turn_limit": 100})
    assert updated.json()["turn_limit"] == 100

    assert client.delete(f"/api/experiments/{exp_id}").status_code == 204


# ── Config ────────────────────────────────────────────────────────────────────

def test_global_config_read_and_update(client):
    resp = client.get("/api/config/globals")
    assert resp.status_code == 200
    assert resp.json()["turn_limit"] == 200

    resp = client.patch("/api/config/globals", json={"turn_limit": 100})
    assert resp.json()["turn_limit"] == 100

    # Persisted — re-read confirms the change
    resp = client.get("/api/config/globals")
    assert resp.json()["turn_limit"] == 100


def test_search_config_key_never_returned(client):
    resp = client.put("/api/config/search", json={
        "provider": "brave",
        "api_key": "brave-top-secret-key-xyz",
        "enabled": True,
    })
    assert resp.status_code == 200
    assert "brave-top-secret-key-xyz" not in resp.text
    assert resp.json()["key_configured"] is True

    # GET also must not return the key
    resp = client.get("/api/config/search")
    assert "brave-top-secret-key-xyz" not in resp.text
    assert resp.json()["key_configured"] is True
    assert "api_key" not in resp.json()

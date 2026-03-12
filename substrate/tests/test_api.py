"""
Substrate API tests for the testplatform substrate.
Verifies health, stream CRUD, search, topics, and admin reset.
"""

import pytest
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api
import db


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_driver():
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock(return_value=None)
    return driver


@pytest.fixture
def client(mock_driver):
    api.app.state.driver = mock_driver
    return TestClient(api.app, raise_server_exceptions=True)


SAMPLE_STREAM = {
    "id": "abc-123",
    "content": "Context windows limit reasoning continuity.",
    "summary": "Analysis of context window limitations.",
    "created_at": "2026-03-11T00:00:00+00:00",
    "source": "conversation",
    "source_id": "",
    "completion_state": "complete",
    "confidence": 0.9,
    "topics": ["ai-architecture"],
    "related_streams": [],
}


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_ok(client, mock_driver):
    with patch.object(db, "check_health", new=AsyncMock(return_value=True)):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "ok"


def test_health_neo4j_down(client, mock_driver):
    with patch.object(db, "check_health", new=AsyncMock(return_value=False)):
        response = client.get("/health")
    assert response.status_code == 503


# ── Streams ────────────────────────────────────────────────────────────────────

def test_create_stream(client):
    fake_embedding = [0.1] * 768
    with patch("embeddings.generate_embedding", new=AsyncMock(return_value=fake_embedding)), \
         patch.object(db, "create_thought_stream", new=AsyncMock(return_value=SAMPLE_STREAM)):
        response = client.post("/streams", json={
            "content": "Context windows limit reasoning continuity.",
            "summary": "Analysis of context window limitations.",
            "source": "conversation",
            "completion_state": "complete",
            "confidence": 0.9,
        })
    assert response.status_code == 201
    assert response.json()["data"]["id"] == "abc-123"


def test_get_stream_found(client):
    with patch.object(db, "get_thought_stream", new=AsyncMock(return_value=SAMPLE_STREAM)):
        response = client.get("/streams/abc-123")
    assert response.status_code == 200
    assert response.json()["data"]["id"] == "abc-123"


def test_get_stream_not_found(client):
    with patch.object(db, "get_thought_stream", new=AsyncMock(return_value=None)):
        response = client.get("/streams/missing-id")
    assert response.status_code == 404


def test_search_streams(client):
    fake_embedding = [0.1] * 768
    with patch("embeddings.generate_embedding", new=AsyncMock(return_value=fake_embedding)), \
         patch.object(db, "search_streams", new=AsyncMock(return_value=[SAMPLE_STREAM])):
        response = client.get("/streams/search?q=context+window")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_get_recent_streams(client):
    with patch.object(db, "get_recent_streams", new=AsyncMock(return_value=[SAMPLE_STREAM])):
        response = client.get("/streams/recent")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


# ── Admin Reset ────────────────────────────────────────────────────────────────

def test_admin_reset(client):
    with patch.object(db, "reset_database", new=AsyncMock(return_value=42)):
        response = client.post("/admin/reset")
    assert response.status_code == 200
    assert response.json()["data"]["deleted_nodes"] == 42
    assert response.json()["data"]["status"] == "reset complete"


# ── Embeddings ────────────────────────────────────────────────────────────────

def test_generate_embedding_calls_ollama():
    import asyncio
    import embeddings

    fake_response = {"data": [{"embedding": [0.5] * 768}]}
    with respx.mock:
        respx.post("http://ollama-embed:11434/v1/embeddings").mock(
            return_value=httpx.Response(200, json=fake_response)
        )
        result = asyncio.run(embeddings.generate_embedding("test text"))
    assert len(result) == 768
    assert result[0] == pytest.approx(0.5)


def test_generate_embedding_uses_correct_model():
    import asyncio
    import json
    import embeddings

    captured = {}
    fake_response = {"data": [{"embedding": [0.1] * 768}]}

    async def capture(request, route):
        captured["body"] = request.content
        return httpx.Response(200, json=fake_response)

    with respx.mock:
        respx.post("http://ollama-embed:11434/v1/embeddings").mock(side_effect=capture)
        asyncio.run(embeddings.generate_embedding("hello"))

    body = json.loads(captured["body"])
    assert body["model"] == "nomic-embed-text"
    assert body["input"] == "hello"

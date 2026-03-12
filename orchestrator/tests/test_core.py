"""
Tests for orchestrator core: Ollama client, tool handler, search, logger.
"""

import asyncio
import json
import os
import sys

import httpx
import pytest
import respx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["RESULTS_DIR"] = str(tmp_path / "results")
    import importlib, store.base, store.keys
    importlib.reload(store.base)
    importlib.reload(store.keys)
    yield tmp_path
    del os.environ["DATA_DIR"]
    del os.environ["RESULTS_DIR"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_completion(content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls or [],
            }
        }],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


# ── Ollama client ─────────────────────────────────────────────────────────────

def test_ollama_client_chat():
    from core.ollama_client import OllamaClient

    fake = _make_completion(content="Hello from the model.")
    with respx.mock:
        respx.post("https://api.ollama.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=fake)
        )
        client = OllamaClient("https://api.ollama.com", "glm-5:cloud", "test-key")
        result = asyncio.run(client.chat([{"role": "user", "content": "Hi"}]))

    assert result["content"] == "Hello from the model."
    assert result["tool_calls"] == []
    assert result["usage"]["prompt_tokens"] == 10


def test_ollama_client_sends_auth_header():
    from core.ollama_client import OllamaClient

    captured = {}
    fake = _make_completion(content="ok")

    async def capture(request, route):
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=fake)

    with respx.mock:
        respx.post("https://api.ollama.com/v1/chat/completions").mock(side_effect=capture)
        client = OllamaClient("https://api.ollama.com", "glm-5:cloud", "my-secret-key")
        asyncio.run(client.chat([{"role": "user", "content": "Hi"}]))

    assert captured["auth"] == "Bearer my-secret-key"


def test_ollama_client_tool_loop():
    """Client executes tool calls and returns final text response."""
    from core.ollama_client import OllamaClient

    tool_call_resp = _make_completion(tool_calls=[{
        "id": "call_1",
        "function": {"name": "web_search", "arguments": json.dumps({"query": "test query"})},
    }])
    final_resp = _make_completion(content="Based on the search results, here is my answer.")

    call_count = 0

    async def fake_post(request, route):
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=tool_call_resp if call_count == 1 else final_resp)

    async def fake_search(query: str) -> str:
        return f"Results for: {query}"

    with respx.mock:
        respx.post("https://api.ollama.com/v1/chat/completions").mock(side_effect=fake_post)
        client = OllamaClient("https://api.ollama.com", "qwen3:cloud", "key")
        text, events, usage = asyncio.run(client.run_with_tools(
            messages=[{"role": "user", "content": "Search for test"}],
            tools=[{"type": "function", "function": {"name": "web_search"}}],
            tool_dispatch={"web_search": fake_search},
        ))

    assert "Based on the search results" in text
    assert len(events) == 1
    assert events[0]["tool"] == "web_search"
    assert not events[0]["error"]


# ── Tool scoping ──────────────────────────────────────────────────────────────

def test_pass1_tools_substrate_only():
    from core.tool_handler import pass1_tools, SUBSTRATE_TOOLS

    tools = pass1_tools()
    names = {t["function"]["name"] for t in tools}
    assert names == {"search_streams", "get_stream", "list_topics", "get_recent"}
    assert "web_search" not in names


def test_pass2_tools_includes_search():
    from core.tool_handler import pass2_tools

    tools = pass2_tools()
    names = {t["function"]["name"] for t in tools}
    assert "web_search" in names
    # Substrate tools must NOT be in Pass 2
    assert "search_streams" not in names
    assert "get_stream" not in names


def test_pass2_tools_includes_extra_mcp():
    from core.tool_handler import pass2_tools

    extra = [{"type": "function", "function": {"name": "my_custom_tool"}}]
    tools = pass2_tools(extra_mcp_tools=extra)
    names = {t["function"]["name"] for t in tools}
    assert "my_custom_tool" in names
    assert "web_search" in names


# ── Search ────────────────────────────────────────────────────────────────────

def test_search_disabled_returns_message(isolated_env):
    from store.keys import set_search_config
    import importlib, core.search
    set_search_config(provider="brave", api_key="key", enabled=False)
    importlib.reload(core.search)

    result = asyncio.run(core.search.web_search("anything"))
    assert "disabled" in result.lower()


def test_search_no_key_returns_message(isolated_env):
    from store.keys import set_search_config
    import importlib, core.search
    set_search_config(provider="brave", api_key="", enabled=True)
    importlib.reload(core.search)

    result = asyncio.run(core.search.web_search("anything"))
    assert "no api key" in result.lower()


def test_search_brave_formats_results(isolated_env):
    from store.keys import set_search_config
    import importlib, core.search
    set_search_config(provider="brave", api_key="brave-key-abc123", enabled=True)
    importlib.reload(core.search)

    fake_response = {
        "web": {"results": [
            {"title": "AI Architecture", "url": "https://example.com", "description": "A paper on AI."},
        ]}
    }

    with respx.mock:
        respx.get("https://api.search.brave.com/res/v1/web/search").mock(
            return_value=httpx.Response(200, json=fake_response)
        )
        result = asyncio.run(core.search.web_search("AI architecture"))

    assert "AI Architecture" in result
    assert "https://example.com" in result


def test_search_unknown_provider(isolated_env):
    from store.keys import set_search_config
    import importlib, core.search
    set_search_config(provider="unknown_provider", api_key="key-xyz", enabled=True)
    importlib.reload(core.search)

    result = asyncio.run(core.search.web_search("query"))
    assert "unknown" in result.lower()


# ── Logger ────────────────────────────────────────────────────────────────────

def test_logger_creates_files(isolated_env):
    os.environ["RESULTS_DIR"] = str(isolated_env / "results")
    import importlib, core.logger
    importlib.reload(core.logger)

    log = core.logger.RunLogger("My Test", "baseline", timestamp="20260311_120000")
    assert log.raw_path.exists()
    assert log.transcript_path.exists()


def test_logger_raw_writes_jsonl(isolated_env):
    os.environ["RESULTS_DIR"] = str(isolated_env / "results")
    import importlib, core.logger
    importlib.reload(core.logger)

    log = core.logger.RunLogger("My Test", "proxy", timestamp="20260311_120001")
    log.log_turn_complete(1, "What is context rot?", "Context rot is...", {"prompt_tokens": 100, "completion_tokens": 50})

    lines = log.raw_path.read_text().strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "turn_complete"
    assert event["turn"] == 1


def test_logger_transcript_writes_markdown(isolated_env):
    os.environ["RESULTS_DIR"] = str(isolated_env / "results")
    import importlib, core.logger
    importlib.reload(core.logger)

    log = core.logger.RunLogger("My Test", "baseline", timestamp="20260311_120002")
    log.transcript_turn(1, "What do you mean by attention?", "Attention is the mechanism by which...")

    content = log.transcript_path.read_text()
    assert "## Turn 1" in content
    assert "**Interviewer:**" in content
    assert "**Model:**" in content


def test_logger_scrubs_keys(isolated_env):
    from store.keys import set_model_key
    from store.models import create_model
    os.environ["RESULTS_DIR"] = str(isolated_env / "results")
    import importlib, core.logger
    importlib.reload(core.logger)

    m = create_model("Test", "m:cloud", "https://api.ollama.com", 256000)
    set_model_key(m.id, "super-secret-key-to-scrub")

    log = core.logger.RunLogger("Scrub Test", "baseline", timestamp="20260311_120003")
    log.log_api_request(1, "pass2", "m:cloud",
                        [{"role": "user", "content": "super-secret-key-to-scrub is in here"}])

    raw_content = log.raw_path.read_text()
    assert "super-secret-key-to-scrub" not in raw_content
    assert "[REDACTED]" in raw_content

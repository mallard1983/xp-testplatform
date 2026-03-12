"""
Tests for proxy condition: substrate client, Pass 1, Pass 2, full proxy loop.
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["RESULTS_DIR"] = str(tmp_path / "results")
    yield tmp_path
    del os.environ["DATA_DIR"]
    del os.environ["RESULTS_DIR"]


def _make_client(responses):
    client = MagicMock()
    client.model_identifier = "qwen3:cloud"
    idx = 0

    async def fake_chat(messages, tools=None, **kwargs):
        nonlocal idx
        r = responses[min(idx, len(responses) - 1)]
        idx += 1
        return {"content": r, "tool_calls": [], "usage": {"prompt_tokens": 30, "completion_tokens": 15}, "raw": {}}

    async def fake_run_with_tools(messages, tools, tool_dispatch, **kwargs):
        nonlocal idx
        r = responses[min(idx, len(responses) - 1)]
        idx += 1
        return r, [], {"prompt_tokens": 30, "completion_tokens": 15}

    client.chat = fake_chat
    client.run_with_tools = fake_run_with_tools
    return client


def _make_logger(tmp_path):
    import importlib, core.logger
    importlib.reload(core.logger)
    return core.logger.RunLogger("proxy-test", "proxy", timestamp="20260311_test")


# ── Substrate client ──────────────────────────────────────────────────────────

def test_substrate_client_search_streams():
    from core.substrate_client import SubstrateClient

    fake_response = {"data": [
        {"id": "abc", "summary": "Context rot analysis", "confidence": 0.9,
         "relevance_score": 0.85, "topics": ["ai"]},
    ]}
    with respx.mock:
        respx.get("http://substrate-api:8000/streams/search").mock(
            return_value=httpx.Response(200, json=fake_response)
        )
        client = SubstrateClient("http://substrate-api:8000")
        result = asyncio.run(client.search_streams("context window"))

    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["id"] == "abc"
    assert "content" not in data[0]  # content excluded from search results


def test_substrate_client_empty_results():
    from core.substrate_client import SubstrateClient

    fake_response = {"data": []}
    with respx.mock:
        respx.get("http://substrate-api:8000/streams/search").mock(
            return_value=httpx.Response(200, json=fake_response)
        )
        client = SubstrateClient("http://substrate-api:8000")
        result = asyncio.run(client.search_streams("anything"))

    assert json.loads(result) == []


def test_substrate_client_get_stream_not_found():
    from core.substrate_client import SubstrateClient

    with respx.mock:
        respx.get("http://substrate-api:8000/streams/missing-id").mock(
            return_value=httpx.Response(404, json={"detail": "Not found"})
        )
        client = SubstrateClient("http://substrate-api:8000")
        result = asyncio.run(client.get_stream("missing-id"))

    data = json.loads(result)
    assert "error" in data


# ── Pass 1 ────────────────────────────────────────────────────────────────────

def test_pass1_returns_context_block():
    from core.pass1 import run_pass1, has_context

    context_output = (
        "CONTEXT_START\n"
        "[STREAM: abc-123] Context rot is the degradation of reasoning quality\n"
        "Earlier in this conversation the model established...\n"
        "CONTEXT_END\n"
        "RETRIEVAL_NOTES: Found 1 relevant stream. Good match."
    )
    model_client = _make_client([context_output])

    substrate = MagicMock()
    substrate.tool_dispatch.return_value = {
        "search_streams": AsyncMock(return_value="[]"),
        "get_stream": AsyncMock(return_value="{}"),
        "list_topics": AsyncMock(return_value="[]"),
        "get_recent": AsyncMock(return_value="[]"),
    }

    result = asyncio.run(run_pass1(
        messages=[{"role": "user", "content": "What is context rot?"}],
        client=model_client,
        substrate_client=substrate,
        pass1_system_prompt="You are the Context Manager.",
    ))

    assert "CONTEXT_START" in result
    assert has_context(result)


def test_pass1_handles_empty_substrate():
    from core.pass1 import run_pass1, has_context

    empty_output = (
        "CONTEXT_START\n"
        "CONTEXT_END\n"
        "RETRIEVAL_NOTES: Substrate returned no results. Pass 2 should proceed without context."
    )
    model_client = _make_client([empty_output])

    substrate = MagicMock()
    substrate.tool_dispatch.return_value = {
        "search_streams": AsyncMock(return_value="[]"),
        "get_stream": AsyncMock(return_value="{}"),
        "list_topics": AsyncMock(return_value="[]"),
        "get_recent": AsyncMock(return_value="[]"),
    }

    result = asyncio.run(run_pass1(
        messages=[{"role": "user", "content": "First turn question"}],
        client=model_client,
        substrate_client=substrate,
        pass1_system_prompt="You are the Context Manager.",
    ))

    assert "CONTEXT_START" in result
    assert not has_context(result)  # empty block


# ── Pass 2 ────────────────────────────────────────────────────────────────────

def test_pass2_injects_context_into_system():
    from core.pass2 import run_pass2

    context_output = (
        "CONTEXT_START\n"
        "[STREAM: abc] Relevant prior reasoning about attention.\n"
        "CONTEXT_END\n"
        "RETRIEVAL_NOTES: Good match."
    )

    captured_messages = []

    async def fake_run_with_tools(messages, tools, tool_dispatch, **kwargs):
        captured_messages.extend(messages)
        return "My answer building on prior context.", [], {"prompt_tokens": 40, "completion_tokens": 20}

    client = MagicMock()
    client.run_with_tools = fake_run_with_tools

    asyncio.run(run_pass2(
        current_question="How does attention work?",
        pass1_output=context_output,
        client=client,
        test_model_system_prompt="You are a participant.",
        tools=[],
        tool_dispatch={},
    ))

    system_msg = next(m for m in captured_messages if m["role"] == "system")
    assert "Prior Conversation Context" in system_msg["content"]
    assert "Relevant prior reasoning" in system_msg["content"]

    user_msg = next(m for m in captured_messages if m["role"] == "user")
    assert user_msg["content"] == "How does attention work?"


def test_pass2_no_context_uses_base_prompt():
    from core.pass2 import run_pass2

    empty_output = "CONTEXT_START\nCONTEXT_END\nRETRIEVAL_NOTES: Nothing found."

    captured_messages = []

    async def fake_run_with_tools(messages, tools, tool_dispatch, **kwargs):
        captured_messages.extend(messages)
        return "My answer.", [], {"prompt_tokens": 20, "completion_tokens": 10}

    client = MagicMock()
    client.run_with_tools = fake_run_with_tools

    asyncio.run(run_pass2(
        current_question="What do you think?",
        pass1_output=empty_output,
        client=client,
        test_model_system_prompt="You are a participant.",
        tools=[],
        tool_dispatch={},
    ))

    system_msg = next(m for m in captured_messages if m["role"] == "system")
    assert system_msg["content"] == "You are a participant."
    assert "Prior Conversation Context" not in system_msg["content"]


# ── Proxy condition turn loop ─────────────────────────────────────────────────

def test_proxy_below_threshold_skips_pass1(tmp_path):
    """Below activation threshold, Pass 1 should not be called."""
    from core.proxy_condition import run_proxy

    model_client = _make_client(["Model response."] * 20)
    iv_client = _make_client(["Interviewer question?"] * 20)
    logger = _make_logger(tmp_path)

    substrate = MagicMock()
    substrate.create_stream = AsyncMock(return_value={})
    substrate.tool_dispatch.return_value = {}

    with patch("core.search.web_search", new=AsyncMock(return_value="")):
        result = asyncio.run(run_proxy(
            pass1_client=model_client,
            pass2_client=model_client,
            interviewer_client=iv_client,
            substrate_client=substrate,
            test_model_system_prompt="You are a participant.",
            interviewer_system_prompt="You are an interviewer.",
            pass1_system_prompt="You are the Context Manager.",
            opening_question="What is context rot?",
            closing_prompt="Summarize.",
            context_window=256000,
            pass1_activation_fraction=0.5,  # threshold: 128K tokens — well above test messages
            turn_limit=2,
            turn_pause_seconds=0,
            checkpoint_turns=[],
            logger=logger,
        ))

    assert result["turns_completed"] == 2
    assert result["pass1_activations"] == 0  # never activated


def test_proxy_above_threshold_runs_pass1(tmp_path):
    """Above activation threshold, Pass 1 activates."""
    from core.proxy_condition import run_proxy

    pass1_output = "CONTEXT_START\n[STREAM: x] Info.\nCONTEXT_END\nRETRIEVAL_NOTES: Found 1."
    model_client = _make_client([pass1_output, "Model answer.", pass1_output, "Model answer."] * 5)
    iv_client = _make_client(["Question?"] * 20)
    logger = _make_logger(tmp_path)

    substrate = MagicMock()
    substrate.create_stream = AsyncMock(return_value={})
    substrate.tool_dispatch.return_value = {
        "search_streams": AsyncMock(return_value="[]"),
    }

    with patch("core.search.web_search", new=AsyncMock(return_value="")):
        result = asyncio.run(run_proxy(
            pass1_client=model_client,
            pass2_client=model_client,
            interviewer_client=iv_client,
            substrate_client=substrate,
            test_model_system_prompt="You are a participant.",
            interviewer_system_prompt="You are an interviewer.",
            pass1_system_prompt="You are the Context Manager.",
            opening_question="What is context rot?",
            closing_prompt="Summarize.",
            context_window=100,             # tiny context window
            pass1_activation_fraction=0.01, # fires immediately
            turn_limit=2,
            turn_pause_seconds=0,
            checkpoint_turns=[],
            logger=logger,
        ))

    assert result["pass1_activations"] > 0


def test_proxy_substrate_write_fires(tmp_path):
    """Substrate write-back is called after each turn."""
    from core.proxy_condition import run_proxy

    model_client = _make_client(["Response."] * 20)
    iv_client = _make_client(["Question?"] * 20)
    logger = _make_logger(tmp_path)

    substrate = MagicMock()
    substrate.create_stream = AsyncMock(return_value={})
    substrate.tool_dispatch.return_value = {}

    with patch("core.search.web_search", new=AsyncMock(return_value="")):
        asyncio.run(run_proxy(
            pass1_client=model_client,
            pass2_client=model_client,
            interviewer_client=iv_client,
            substrate_client=substrate,
            test_model_system_prompt="You are a participant.",
            interviewer_system_prompt="You are an interviewer.",
            pass1_system_prompt="You are the Context Manager.",
            opening_question="What is context rot?",
            closing_prompt="Summarize.",
            context_window=256000,
            pass1_activation_fraction=0.5,
            turn_limit=3,
            turn_pause_seconds=0,
            checkpoint_turns=[],
            logger=logger,
        ))

    # 4 writes: opening + 3 turns
    assert substrate.create_stream.call_count == 4

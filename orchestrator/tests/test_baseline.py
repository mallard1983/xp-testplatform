"""
Tests for compaction logic and baseline condition turn loop.
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["RESULTS_DIR"] = str(tmp_path / "results")
    yield tmp_path
    del os.environ["DATA_DIR"]
    del os.environ["RESULTS_DIR"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client(responses: list[str | dict]):
    """
    Build a mock OllamaClient. responses is a list of:
      - str: a plain text response
      - dict: {"tool": name, "args": dict} to simulate a tool call then a text response
    Each item in responses is consumed in order per chat() call.
    """
    client = MagicMock()
    client.model_identifier = "test-model:cloud"

    call_count = 0

    async def fake_chat(messages, tools=None, **kwargs):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return {
            "content": resp if isinstance(resp, str) else None,
            "tool_calls": [],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            "raw": {},
        }

    async def fake_run_with_tools(messages, tools, tool_dispatch, max_iterations=12):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return (
            resp if isinstance(resp, str) else "Final answer after tool use.",
            [],
            {"prompt_tokens": 50, "completion_tokens": 20},
        )

    client.chat = fake_chat
    client.run_with_tools = fake_run_with_tools
    return client


def _make_logger(tmp_path):
    import importlib, core.logger
    importlib.reload(core.logger)
    return core.logger.RunLogger("test", "baseline", timestamp="20260311_test")


# ── Compaction ────────────────────────────────────────────────────────────────

def test_count_tokens_returns_positive():
    from core.compaction import count_tokens
    messages = [
        {"role": "system", "content": "You are a test model."},
        {"role": "user", "content": "What is context rot?"},
        {"role": "assistant", "content": "Context rot refers to the degradation of..."},
    ]
    total = count_tokens(messages)
    assert total > 0


def test_needs_compaction_below_threshold():
    from core.compaction import needs_compaction
    messages = [{"role": "user", "content": "short message"}]
    assert needs_compaction(messages, context_window=256000, threshold_fraction=0.8) is False


def test_needs_compaction_at_threshold():
    from core.compaction import count_tokens, needs_compaction

    # Build messages large enough to exceed a tiny context window
    big_text = "word " * 5000
    messages = [{"role": "user", "content": big_text}]
    tokens = count_tokens(messages)
    # Use a context window just below the token count
    small_window = int(tokens / 0.5)
    assert needs_compaction(messages, context_window=small_window, threshold_fraction=0.5) is True


def test_run_compaction_returns_summary_and_new_messages():
    from core.compaction import run_compaction

    client = _make_client(["GOAL\nTest context.\n\nCONCLUSIONS\nNone yet."])

    summary, new_messages, usage = asyncio.run(run_compaction(
        messages=[
            {"role": "system", "content": "You are a test model."},
            {"role": "user", "content": "Turn 1 question"},
            {"role": "assistant", "content": "Turn 1 answer"},
        ],
        client=client,
        compaction_prompt="Summarize the conversation.",
        system_prompt="You are a test model.",
        turn=1,
    ))

    assert "GOAL" in summary
    assert new_messages[0]["role"] == "system"
    assert "SUMMARY" in new_messages[1]["content"]
    assert new_messages[2]["role"] == "assistant"
    assert usage["prompt_tokens"] > 0


# ── Baseline turn loop ────────────────────────────────────────────────────────

def test_baseline_opening_exchange(tmp_path):
    """Opening exchange runs and populates interviewer messages."""
    model_client = _make_client(["This is my opening response about context rot."] * 20)
    iv_client = _make_client(["What mechanism produces that?" ] * 20)
    logger = _make_logger(tmp_path)

    with patch("core.search.web_search", new=AsyncMock(return_value="search results")):
        result = asyncio.run(run_baseline_short(model_client, iv_client, logger, turn_limit=1))

    assert result["turns_completed"] == 1
    assert result["total_tokens"]["prompt"] > 0


def test_baseline_runs_correct_turn_count(tmp_path):
    model_client = _make_client(["Model answer."] * 210)
    iv_client = _make_client(["Interviewer question."] * 210)
    logger = _make_logger(tmp_path)

    with patch("core.search.web_search", new=AsyncMock(return_value="")):
        result = asyncio.run(run_baseline_short(model_client, iv_client, logger, turn_limit=3))

    assert result["turns_completed"] == 3


def test_baseline_compaction_fires(tmp_path):
    """Compaction fires when threshold is exceeded."""
    from core.compaction import count_tokens

    model_client = _make_client(["Long answer. " * 200] * 20)
    iv_client = _make_client(["Question?"] * 20)
    logger = _make_logger(tmp_path)

    # Use a very small context window so compaction fires immediately
    with patch("core.search.web_search", new=AsyncMock(return_value="")):
        result = asyncio.run(run_baseline_short(
            model_client, iv_client, logger,
            turn_limit=2,
            context_window=200,           # tiny window
            compaction_threshold_fraction=0.01,  # fires almost immediately
        ))

    assert len(result["compaction_events"]) > 0


def test_baseline_checkpoints_logged(tmp_path):
    model_client = _make_client(["Answer."] * 20)
    iv_client = _make_client(["Question?"] * 20)
    logger = _make_logger(tmp_path)

    with patch("core.search.web_search", new=AsyncMock(return_value="")):
        result = asyncio.run(run_baseline_short(
            model_client, iv_client, logger,
            turn_limit=3,
            checkpoint_turns=[1, 3],
        ))

    assert 1 in result["checkpoint_turns"]
    assert 3 in result["checkpoint_turns"]


def test_baseline_logs_are_written(tmp_path):
    model_client = _make_client(["Answer."] * 10)
    iv_client = _make_client(["Question?"] * 10)
    logger = _make_logger(tmp_path)

    with patch("core.search.web_search", new=AsyncMock(return_value="")):
        asyncio.run(run_baseline_short(model_client, iv_client, logger, turn_limit=2))

    raw_lines = logger.raw_path.read_text().strip().splitlines()
    assert len(raw_lines) > 0

    transcript = logger.transcript_path.read_text()
    assert "## Turn 1" in transcript
    assert "## Turn 2" in transcript


# ── Helper ────────────────────────────────────────────────────────────────────

async def run_baseline_short(
    model_client, iv_client, logger,
    turn_limit=2,
    context_window=256000,
    compaction_threshold_fraction=0.8,
    checkpoint_turns=None,
):
    from core.baseline import run_baseline
    return await run_baseline(
        pass2_client=model_client,
        interviewer_client=iv_client,
        test_model_system_prompt="You are a test model.",
        interviewer_system_prompt="You are an interviewer.",
        opening_question="What is context rot?",
        closing_prompt="Summarize your thinking.",
        compaction_prompt="Summarize the conversation.",
        context_window=context_window,
        compaction_threshold_fraction=compaction_threshold_fraction,
        turn_limit=turn_limit,
        turn_pause_seconds=0,  # no pause in tests
        checkpoint_turns=checkpoint_turns or [],
        logger=logger,
    )

"""
T8.1 Automated smoke test — 3-turn end-to-end with mocked LLM clients.

Runs both baseline and proxy conditions against the same experiment config,
verifies all expected output files, builds the evaluation package, and audits
every output file to confirm no API key values appear in the logs or transcripts.

This test is the gate for Phase 8.
"""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    config_dir = tmp_path / "config"
    prompts_dir = config_dir / "prompts"

    data_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)

    (config_dir / "config.yaml").write_text(yaml.dump({
        "experiment": {
            "turn_limit": 3,
            "context_window": 256000,
            "compaction_threshold_fraction": 0.80,
            "pass1_activation_fraction": 0.50,
            "turn_pause_seconds": 0,
            "checkpoint_turns": [3],
        }
    }))

    (prompts_dir / "opening_question.txt").write_text(
        "AI systems currently have a fundamental limitation with context. What is your current understanding of this problem?"
    )
    (prompts_dir / "test_model_system.txt").write_text(
        "You are a participant in a structured research conversation. Reason carefully."
    )
    (prompts_dir / "interviewer_system.txt").write_text(
        "You are an interviewer exploring a research topic. Ask focused follow-up questions."
    )
    (prompts_dir / "closing_prompt.txt").write_text(
        "We are at the end of our conversation. Describe where your thinking has arrived."
    )
    (prompts_dir / "compaction_prompt.txt").write_text(
        "Summarize: GOAL, KEY CONSTRAINTS, REASONING, CONCLUSIONS, OPEN QUESTIONS."
    )
    (prompts_dir / "pass1_system.txt").write_text(
        "You are the Context Manager. Return a CONTEXT_START...CONTEXT_END block."
    )

    os.environ["DATA_DIR"]    = str(data_dir)
    os.environ["RESULTS_DIR"] = str(results_dir)
    os.environ["CONFIG_PATH"] = str(config_dir / "config.yaml")

    yield tmp_path

    for key in ("DATA_DIR", "RESULTS_DIR", "CONFIG_PATH"):
        os.environ.pop(key, None)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_client(responses, model_id="test-model:cloud"):
    """Mock OllamaClient that walks through canned responses in order."""
    client = MagicMock()
    client.model_identifier = model_id
    idx = 0

    async def fake_chat(messages, tools=None, **kwargs):
        nonlocal idx
        r = responses[min(idx, len(responses) - 1)]
        idx += 1
        return {"content": r, "tool_calls": [], "usage": {"prompt_tokens": 20, "completion_tokens": 10}, "raw": {}}

    async def fake_run_with_tools(messages, tools, tool_dispatch, **kwargs):
        nonlocal idx
        r = responses[min(idx, len(responses) - 1)]
        idx += 1
        return r, [], {"prompt_tokens": 20, "completion_tokens": 10}

    client.chat = fake_chat
    client.run_with_tools = fake_run_with_tools
    return client


class _MockNeo4jManager:
    """No-op neo4j manager for tests that exercise the proxy condition."""
    async def prepare_fresh(self): pass
    async def prepare_from_snapshot(self, test_id, source_ts): pass
    def save_snapshot(self, test_id, timestamp): return Path("/tmp")
    def list_snapshots(self, test_id): return []


def _make_experiment(fake_key: str, **overrides):
    """Create model + experiment entries in store. Returns ExperimentConfig."""
    import core.logger
    importlib.reload(core.logger)

    from store.models import create_model
    from store.keys import set_model_key
    from store.experiments import create_experiment

    m = create_model(
        name="Smoke Model",
        model_identifier="smoke-model:test",
        endpoint_url="http://localhost:11434",
        context_window=overrides.pop("model_context_window", 1000),
    )
    set_model_key(m.id, fake_key)

    exp = create_experiment(
        name="smoke test",
        pass1_model_id=m.id,
        pass2_model_id=m.id,
        interviewer_model_id=m.id,
        turn_limit=3,
        turn_pause_seconds=0,
        **overrides,
    )
    return exp


def _mock_substrate():
    s = MagicMock()
    s.create_stream = AsyncMock(return_value={})
    s.tool_dispatch.return_value = {
        "search_streams": AsyncMock(return_value="[]"),
        "get_stream":     AsyncMock(return_value="{}"),
        "list_topics":    AsyncMock(return_value="[]"),
        "get_recent":     AsyncMock(return_value="[]"),
    }
    return s


# ── T8.1: Baseline 3-turn smoke ────────────────────────────────────────────────

def test_baseline_3turn_produces_all_expected_files():
    """Baseline: 3 turns → transcript, raw log, summary, checkpoint artifact, closing artifact."""
    from core.experiment import run_condition

    FAKE_KEY = "SMOKEKEY_BASELINE_001"
    exp = _make_experiment(FAKE_KEY)

    # Opening(0) + 3 × [IV question, model answer] + closing
    responses = [
        "Opening answer about context limitations.",          # turn 0
        "Interviewer Q1?",  "Model answer turn 1.",
        "Interviewer Q2?",  "Model answer turn 2.",
        "Interviewer Q3?",  "Model answer turn 3.",           # also logged as checkpoint
        "Closing reflection on the full conversation.",
    ]
    mock_client = _make_client(responses)

    with patch("core.experiment._resolve_client", side_effect=lambda _: mock_client):
        with patch("core.search.web_search", new=AsyncMock(return_value="")):
            result = asyncio.run(run_condition(
                experiment=exp,
                condition="baseline",
                timestamp="20260311_smoke01",
                run_id="smoke-baseline-001",
            ))

    results_dir = Path(os.environ["RESULTS_DIR"])

    # Summary
    summaries = list(results_dir.rglob("summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert summary["condition"] == "baseline"
    assert summary["turns_completed"] == 3
    assert summary["run_id"] == "smoke-baseline-001"
    assert summary["parameters"]["turn_limit"] == 3
    assert summary["pass2_model"] == "smoke-model:test"
    assert summary["database_source"] is None

    # Raw JSONL
    raw_logs = list(results_dir.rglob("raw_*.jsonl"))
    assert len(raw_logs) == 1
    events = [json.loads(l) for l in raw_logs[0].read_text().splitlines() if l.strip()]
    event_types = {e.get("event") for e in events}
    assert "turn_complete" in event_types
    assert "closing" in event_types
    assert "checkpoint" in event_types  # turn 3 is a checkpoint

    # Transcript
    transcripts = list(results_dir.rglob("transcript_*.md"))
    assert len(transcripts) == 1
    transcript = transcripts[0].read_text()
    assert "## Turn 1" in transcript
    assert "## Turn 2" in transcript
    assert "## Turn 3" in transcript
    assert "## Closing" in transcript
    assert "Interviewer" in transcript
    assert "Model" in transcript

    # Checkpoint artifact
    ckpt = list(results_dir.rglob("turn_3_baseline.md"))
    assert len(ckpt) == 1
    assert "turn 3" in ckpt[0].read_text().lower()

    # Closing artifact
    closing = list(results_dir.rglob("closing_baseline.md"))
    assert len(closing) == 1
    assert "Closing" in closing[0].read_text()


# ── T8.1: Proxy 3-turn smoke ───────────────────────────────────────────────────

def test_proxy_3turn_produces_all_expected_files():
    """
    Proxy: 3 turns with pass1 firing every turn.
    context_window=100, pass1_activation_fraction=0.01 → threshold = 1 token, fires immediately.
    """
    from core.experiment import run_condition

    FAKE_KEY = "SMOKEKEY_PROXY_002"
    # Tiny context window so Pass 1 fires on every turn
    exp = _make_experiment(FAKE_KEY, context_window=100, pass1_activation_fraction=0.01)

    pass1_ctx = (
        "CONTEXT_START\n"
        "[STREAM: abc] Relevant prior reasoning about context windows.\n"
        "CONTEXT_END\n"
        "RETRIEVAL_NOTES: Found 1 relevant stream."
    )
    # Index order: turn0 run_with_tools, then per turn: iv.chat, pass1.run_with_tools, pass2.run_with_tools
    responses = [
        "Opening answer.",                               # 0: turn 0 pass2.run_with_tools
        "Interviewer Q1?",                               # 1: iv.chat turn 1
        pass1_ctx,                                       # 2: pass1.run_with_tools turn 1
        "Model answer turn 1 with injected context.",    # 3: pass2.run_with_tools turn 1
        "Interviewer Q2?",                               # 4
        pass1_ctx,                                       # 5
        "Model answer turn 2.",                          # 6
        "Interviewer Q3?",                               # 7
        pass1_ctx,                                       # 8
        "Model answer turn 3.",                          # 9
        "Closing synthesis of the conversation.",        # 10: pass2.chat closing
    ]
    mock_client = _make_client(responses)

    with patch("core.experiment._resolve_client", side_effect=lambda _: mock_client):
        with patch("core.experiment.SubstrateClient", return_value=_mock_substrate()):
            with patch("core.search.web_search", new=AsyncMock(return_value="")):
                result = asyncio.run(run_condition(
                    experiment=exp,
                    condition="proxy",
                    timestamp="20260311_smoke02",
                    run_id="smoke-proxy-001",
                    neo4j_manager=_MockNeo4jManager(),
                ))

    results_dir = Path(os.environ["RESULTS_DIR"])

    summary = json.loads(list(results_dir.rglob("summary.json"))[0].read_text())
    assert summary["condition"] == "proxy"
    assert summary["turns_completed"] == 3
    assert summary["pass1_activations"] == 3        # fired every turn
    assert summary["database_source"] == "new"
    assert summary["pass1_model"] == "smoke-model:test"
    assert summary["pass2_model"] == "smoke-model:test"

    transcript = list(results_dir.rglob("transcript_*.md"))[0].read_text()
    assert "## Turn 1" in transcript
    assert "## Turn 3" in transcript
    assert "## Closing" in transcript

    assert len(list(results_dir.rglob("closing_proxy.md"))) == 1
    assert len(list(results_dir.rglob("turn_3_proxy.md"))) == 1


# ── T8.2: Key scrubber audit ───────────────────────────────────────────────────

def test_key_scrubber_no_key_in_any_output_file():
    """
    Key scrubber audit: API key value must not appear in ANY output file,
    even when the mock model echoes the key back in its responses.
    """
    from core.experiment import run_condition

    FAKE_KEY = "SUPERSECRETKEY_AUDIT_XYZ"
    exp = _make_experiment(FAKE_KEY)

    # Model deliberately echoes the key
    responses = [
        f"The answer involves {FAKE_KEY} as a token.",
        "Follow-up question?",
        f"Turn 1: credential is {FAKE_KEY}.",
        "Another question?",
        "Turn 2: normal response.",
        "Final question?",
        "Turn 3: reasoning complete.",
        f"Closing: key was {FAKE_KEY} all along.",
    ]
    mock_client = _make_client(responses)

    with patch("core.experiment._resolve_client", side_effect=lambda _: mock_client):
        with patch("core.search.web_search", new=AsyncMock(return_value="")):
            asyncio.run(run_condition(
                experiment=exp,
                condition="baseline",
                timestamp="20260311_scrub",
                run_id="smoke-scrub-001",
            ))

    results_dir = Path(os.environ["RESULTS_DIR"])

    violations = []
    for output_file in results_dir.rglob("*"):
        if output_file.is_file():
            content = output_file.read_text(errors="replace")
            if FAKE_KEY in content:
                violations.append(str(output_file.relative_to(results_dir)))

    assert violations == [], (
        f"Key '{FAKE_KEY}' found unredacted in: {violations}"
    )


# ── SSE callback fires on every turn ──────────────────────────────────────────

def test_sse_callback_receives_turn_events():
    """event_callback receives a turn_complete event for every turn."""
    from core.experiment import run_condition

    FAKE_KEY = "CALLBACKKEY_004"
    exp = _make_experiment(FAKE_KEY)

    responses = ["Opening.", "IV Q1?", "A1.", "IV Q2?", "A2.", "IV Q3?", "A3.", "Closing."]
    mock_client = _make_client(responses)

    received = []

    async def capture(event_type, data):
        received.append({"type": event_type, **data})

    with patch("core.experiment._resolve_client", side_effect=lambda _: mock_client):
        with patch("core.search.web_search", new=AsyncMock(return_value="")):
            asyncio.run(run_condition(
                experiment=exp,
                condition="baseline",
                timestamp="20260311_cb",
                run_id="smoke-cb-001",
                event_callback=capture,
            ))

    turn_events = [e for e in received if e["type"] == "turn_complete"]
    assert len(turn_events) == 3
    assert [e["turn"] for e in turn_events] == [1, 2, 3]
    assert all("total_tokens" in e for e in turn_events)
    assert all(e["turn_limit"] == 3 for e in turn_events)


# ── Evaluation package from paired runs ───────────────────────────────────────

def test_evaluation_package_from_paired_runs():
    """Build a blinded A/B evaluation package from a paired baseline + proxy run."""
    from core.experiment import run_condition
    from core.extractor import build_evaluation_package

    FAKE_KEY = "EVALPACKAGEKEY_005"

    baseline_responses = [
        "Opening baseline.", "IV Q1?", "B Turn1.", "IV Q2?", "B Turn2.", "IV Q3?", "B Turn3.", "B Closing.",
    ]
    proxy_ctx = "CONTEXT_START\n[S:x] Prior context.\nCONTEXT_END\nRETRIEVAL_NOTES: found."
    proxy_responses = [
        "Opening proxy.",
        "IV Q1?", proxy_ctx, "P Turn1.",
        "IV Q2?", proxy_ctx, "P Turn2.",
        "IV Q3?", proxy_ctx, "P Turn3.",
        "P Closing.",
    ]

    # Baseline run
    exp_b = _make_experiment(FAKE_KEY)
    with patch("core.experiment._resolve_client", side_effect=lambda _: _make_client(baseline_responses)):
        with patch("core.search.web_search", new=AsyncMock(return_value="")):
            asyncio.run(run_condition(
                experiment=exp_b, condition="baseline",
                timestamp="20260311_eval_b", run_id="smoke-eval-baseline",
            ))

    # Proxy run (must reload logger so file timestamps differ)
    importlib.reload(importlib.import_module("core.logger"))
    exp_p = _make_experiment(FAKE_KEY, context_window=100, pass1_activation_fraction=0.01)
    with patch("core.experiment._resolve_client", side_effect=lambda _: _make_client(proxy_responses)):
        with patch("core.experiment.SubstrateClient", return_value=_mock_substrate()):
            with patch("core.search.web_search", new=AsyncMock(return_value="")):
                asyncio.run(run_condition(
                    experiment=exp_p, condition="proxy",
                    timestamp="20260311_eval_p", run_id="smoke-eval-proxy",
                    neo4j_manager=_MockNeo4jManager(),
                ))

    results_dir = Path(os.environ["RESULTS_DIR"])
    baseline_dir = next(d.parent for d in results_dir.rglob("summary.json")
                        if json.loads(d.read_text()).get("condition") == "baseline")
    proxy_dir    = next(d.parent for d in results_dir.rglob("summary.json")
                        if json.loads(d.read_text()).get("condition") == "proxy")

    output_dir = results_dir / "evaluation"
    result = build_evaluation_package(baseline_dir, proxy_dir, output_dir)

    package = json.loads(Path(result["package_path"]).read_text())
    key     = json.loads(Path(result["key_path"]).read_text())

    assert set(package.keys()) == {"A", "B"}
    assert set(key.values()) == {"baseline", "proxy"}
    # Closing artifacts present in both
    assert any("closing" in k.lower() for k in package["A"])
    assert any("closing" in k.lower() for k in package["B"])
    # Key not leaked into package
    assert FAKE_KEY not in Path(result["package_path"]).read_text()

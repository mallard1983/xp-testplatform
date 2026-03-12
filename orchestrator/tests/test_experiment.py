"""
Tests for Phase 6: Neo4j manager, extractor, and end-to-end experiment runner.
"""

import asyncio
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

    # Minimal config.yaml
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump({
        "experiment": {
            "turn_limit": 2,
            "context_window": 1000,
            "compaction_threshold_fraction": 0.80,
            "pass1_activation_fraction": 0.50,
            "turn_pause_seconds": 0,
            "checkpoint_turns": [2],
        }
    }))

    # Prompt files
    (prompts_dir / "opening_question.txt").write_text("What is context rot?")
    (prompts_dir / "test_model_system.txt").write_text("You are a participant.")
    (prompts_dir / "interviewer_system.txt").write_text("You are an interviewer.")
    (prompts_dir / "closing_prompt.txt").write_text("Summarize where your thinking has arrived.")
    (prompts_dir / "compaction_prompt.txt").write_text("Summarize the conversation.")
    (prompts_dir / "pass1_system.txt").write_text("You are the Context Manager.")

    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["RESULTS_DIR"] = str(results_dir)
    os.environ["CONFIG_PATH"] = str(config_path)

    yield tmp_path

    for key in ("DATA_DIR", "RESULTS_DIR", "CONFIG_PATH"):
        os.environ.pop(key, None)


def _make_client(responses):
    """Shared mock client (mirrors test_proxy_condition.py helper)."""
    client = MagicMock()
    client.model_identifier = "test-model:cloud"
    idx = 0

    async def fake_chat(messages, tools=None, **kwargs):
        nonlocal idx
        r = responses[min(idx, len(responses) - 1)]
        idx += 1
        return {"content": r, "tool_calls": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "raw": {}}

    async def fake_run_with_tools(messages, tools, tool_dispatch, **kwargs):
        nonlocal idx
        r = responses[min(idx, len(responses) - 1)]
        idx += 1
        return r, [], {"prompt_tokens": 10, "completion_tokens": 5}

    client.chat = fake_chat
    client.run_with_tools = fake_run_with_tools
    return client


# ── Neo4j manager ─────────────────────────────────────────────────────────────

def test_neo4j_manager_prepare_fresh_clears_active(tmp_path):
    from core.neo4j_manager import Neo4jManager

    mgr = Neo4jManager(data_dir=tmp_path / "data", container_name="test-neo4j")

    # Pre-populate active_neo4j to confirm it gets cleared
    active = tmp_path / "data" / "active_neo4j"
    (active / "data").mkdir(parents=True)
    (active / "data" / "old_file.db").write_text("stale")
    (active / "logs").mkdir()

    async def fake_stop(*args, **kwargs):
        pass

    async def fake_start(*args, **kwargs):
        pass

    async def fake_healthy(*args, **kwargs):
        pass

    mgr._stop = fake_stop
    mgr._start = fake_start
    mgr._wait_healthy = fake_healthy

    asyncio.run(mgr.prepare_fresh())

    # active_neo4j/data should be empty (old_file.db removed)
    assert not (active / "data" / "old_file.db").exists()
    assert (active / "data").is_dir()
    assert (active / "logs").is_dir()


def test_neo4j_manager_save_snapshot(tmp_path):
    from core.neo4j_manager import Neo4jManager

    mgr = Neo4jManager(data_dir=tmp_path / "data", container_name="test-neo4j")

    # Put something in active_neo4j
    active = tmp_path / "data" / "active_neo4j"
    (active / "data").mkdir(parents=True)
    (active / "data" / "neo4j.db").write_text("graph data")
    (active / "logs").mkdir()
    (active / "logs" / "neo4j.log").write_text("log line")

    dest = mgr.save_snapshot("exp-1", "20260311_120000")

    assert (dest / "data" / "neo4j.db").read_text() == "graph data"
    assert (dest / "logs" / "neo4j.log").read_text() == "log line"


def test_neo4j_manager_prepare_from_snapshot(tmp_path):
    from core.neo4j_manager import Neo4jManager

    mgr = Neo4jManager(data_dir=tmp_path / "data", container_name="test-neo4j")

    # Create a snapshot
    snapshot = tmp_path / "data" / "tests" / "exp-1" / "neo4j" / "20260310_120000"
    (snapshot / "data").mkdir(parents=True)
    (snapshot / "data" / "restored.db").write_text("prior state")
    (snapshot / "logs").mkdir()

    async def fake_stop(*args, **kwargs):
        pass

    async def fake_start(*args, **kwargs):
        pass

    async def fake_healthy(*args, **kwargs):
        pass

    mgr._stop = fake_stop
    mgr._start = fake_start
    mgr._wait_healthy = fake_healthy

    asyncio.run(mgr.prepare_from_snapshot("exp-1", "20260310_120000"))

    active = tmp_path / "data" / "active_neo4j"
    assert (active / "data" / "restored.db").read_text() == "prior state"


def test_neo4j_manager_list_snapshots(tmp_path):
    from core.neo4j_manager import Neo4jManager

    mgr = Neo4jManager(data_dir=tmp_path / "data", container_name="test-neo4j")

    base = tmp_path / "data" / "tests" / "exp-1" / "neo4j"
    for ts in ("20260310_100000", "20260311_120000", "20260312_090000"):
        (base / ts).mkdir(parents=True)

    snapshots = mgr.list_snapshots("exp-1")
    assert snapshots == ["20260310_100000", "20260311_120000", "20260312_090000"]


def test_neo4j_manager_list_snapshots_empty(tmp_path):
    from core.neo4j_manager import Neo4jManager

    mgr = Neo4jManager(data_dir=tmp_path / "data", container_name="test-neo4j")
    assert mgr.list_snapshots("nonexistent") == []


# ── Extractor ─────────────────────────────────────────────────────────────────

def _write_fake_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def test_extractor_writes_checkpoint_artifact(tmp_path):
    from core.extractor import extract_artifacts

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw_path = run_dir / "raw_test_baseline_ts.jsonl"

    _write_fake_jsonl(raw_path, [
        {"event": "turn_complete", "turn": 1, "model_response": "Some response."},
        {"event": "checkpoint", "turn": 2, "response": "Deep analysis at turn 2."},
    ])

    artifacts = extract_artifacts(raw_path, "baseline", run_dir, checkpoint_turns=[2])

    assert "turn_2" in artifacts
    artifact_path = run_dir / artifacts["turn_2"]
    assert artifact_path.exists()
    assert "Deep analysis at turn 2." in artifact_path.read_text()


def test_extractor_writes_closing_artifact(tmp_path):
    from core.extractor import extract_artifacts

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw_path = run_dir / "raw_test_proxy_ts.jsonl"

    _write_fake_jsonl(raw_path, [
        {"event": "closing", "prompt": "Summarize.", "response": "Final synthesis."},
    ])

    artifacts = extract_artifacts(raw_path, "proxy", run_dir, checkpoint_turns=[])

    assert "closing" in artifacts
    closing_path = run_dir / artifacts["closing"]
    assert "Final synthesis." in closing_path.read_text()


def test_extractor_empty_log_returns_no_artifacts(tmp_path):
    from core.extractor import extract_artifacts

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw_path = run_dir / "raw_empty.jsonl"
    raw_path.touch()

    artifacts = extract_artifacts(raw_path, "baseline", run_dir, checkpoint_turns=[2])
    assert artifacts == {}


def test_extractor_missing_log_returns_no_artifacts(tmp_path):
    from core.extractor import extract_artifacts

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw_path = run_dir / "does_not_exist.jsonl"

    artifacts = extract_artifacts(raw_path, "baseline", run_dir, checkpoint_turns=[2])
    assert artifacts == {}


# ── Evaluation package ────────────────────────────────────────────────────────

def test_evaluation_package_ab_structure(tmp_path):
    from core.extractor import build_evaluation_package

    # Build two fake run dirs with artifacts
    for condition in ("baseline", "proxy"):
        run_dir = tmp_path / condition
        artifact_dir = run_dir / "artifacts"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / f"turn_2_{condition}.md").write_text(f"Turn 2 {condition} response")
        (artifact_dir / f"closing_{condition}.md").write_text(f"Closing {condition} response")

    output_dir = tmp_path / "eval"
    result = build_evaluation_package(
        baseline_run_dir=tmp_path / "baseline",
        proxy_run_dir=tmp_path / "proxy",
        output_dir=output_dir,
    )

    package_path = Path(result["package_path"])
    key_path = Path(result["key_path"])

    assert package_path.exists()
    assert key_path.exists()

    package = json.loads(package_path.read_text())
    key = json.loads(key_path.read_text())

    # Package has A and B, both with artifact content
    assert set(package.keys()) == {"A", "B"}
    assert len(package["A"]) > 0
    assert len(package["B"]) > 0

    # Key maps to the two valid conditions
    assert set(key.values()) == {"baseline", "proxy"}
    assert set(key.keys()) == {"A", "B"}


def test_evaluation_package_ab_labels_are_random(tmp_path):
    from core.extractor import build_evaluation_package

    for condition in ("baseline", "proxy"):
        run_dir = tmp_path / condition
        artifact_dir = run_dir / "artifacts"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / f"closing_{condition}.md").write_text(f"Closing {condition}")

    # Run many times — both A=baseline and A=proxy should appear
    a_labels = set()
    for i in range(20):
        output_dir = tmp_path / f"eval_{i}"
        result = build_evaluation_package(
            baseline_run_dir=tmp_path / "baseline",
            proxy_run_dir=tmp_path / "proxy",
            output_dir=output_dir,
        )
        key = json.loads(Path(result["key_path"]).read_text())
        a_labels.add(key["A"])

    # With 20 runs and p=0.5, probability of all same label is (0.5^19) ≈ 0 — both should appear
    assert a_labels == {"baseline", "proxy"}


# ── End-to-end: run_condition (baseline, 2 turns) ─────────────────────────────

def test_run_condition_baseline_produces_logs_and_summary(tmp_path):
    """
    2-turn baseline run with mocked clients — verifies logs, summary, artifacts.
    """
    import importlib
    import core.logger
    importlib.reload(core.logger)

    from core.experiment import run_condition
    from store.models import create_model
    from store.keys import set_model_key
    from store.experiments import create_experiment

    # Create model entry in store
    m = create_model(
        name="Test Model",
        model_identifier="test-model:cloud",
        endpoint_url="http://test-api:11434",
        context_window=1000,
    )
    set_model_key(m.id, "fake-api-key")

    # Create experiment config
    exp = create_experiment(
        name="e2e test",
        pass1_model_id=m.id,
        pass2_model_id=m.id,
        interviewer_model_id=m.id,
        turn_limit=2,
        turn_pause_seconds=0,
        context_window=1000,
    )

    # Responses: turn0, turn1_iv, turn1_model, turn2_iv, turn2_model, closing
    responses = [
        "Opening answer about context rot.",
        "Follow-up question from interviewer?",
        "Model answer turn 1.",
        "Another interviewer question?",
        "Model answer turn 2.",
        "Closing summary of thinking.",
    ]
    mock_client = _make_client(responses)

    with patch("core.experiment._resolve_client", side_effect=lambda _: mock_client):
        with patch("core.search.web_search", new=AsyncMock(return_value="")):
            result = asyncio.run(run_condition(
                experiment=exp,
                condition="baseline",
                timestamp="20260311_test",
            ))

    # Verify summary structure
    assert result["condition"] == "baseline"
    assert result["turns_completed"] == 2
    assert "total_tokens" in result
    assert result["database_source"] is None  # baseline has no substrate

    # Verify summary.json on disk
    results_dir = Path(os.environ["RESULTS_DIR"])
    run_dirs = list(results_dir.rglob("summary.json"))
    assert len(run_dirs) == 1
    summary = json.loads(run_dirs[0].read_text())
    assert summary["condition"] == "baseline"
    assert summary["experiment_name"] == "e2e test"
    assert "parameters" in summary

    # Verify transcript exists and has content
    transcripts = list(results_dir.rglob("transcript_*.md"))
    assert len(transcripts) == 1
    transcript_text = transcripts[0].read_text()
    assert "## Turn 1" in transcript_text
    assert "## Turn 2" in transcript_text
    assert "## Closing" in transcript_text

    # Verify raw JSONL exists
    raw_logs = list(results_dir.rglob("raw_*.jsonl"))
    assert len(raw_logs) == 1
    events = [json.loads(line) for line in raw_logs[0].read_text().splitlines() if line.strip()]
    event_types = {e["event"] for e in events}
    assert "turn_complete" in event_types
    assert "closing" in event_types


def test_run_condition_produces_checkpoint_artifact(tmp_path):
    """Checkpoint turns produce artifact files in the artifacts/ directory."""
    import importlib
    import core.logger
    importlib.reload(core.logger)

    from core.experiment import run_condition
    from store.models import create_model
    from store.keys import set_model_key
    from store.experiments import create_experiment

    m = create_model(
        name="Test Model",
        model_identifier="test-model:cloud",
        endpoint_url="http://test-api:11434",
        context_window=1000,
    )
    set_model_key(m.id, "fake-api-key")

    exp = create_experiment(
        name="checkpoint test",
        pass1_model_id=m.id,
        pass2_model_id=m.id,
        interviewer_model_id=m.id,
        turn_limit=2,
        turn_pause_seconds=0,
        context_window=1000,
    )

    responses = [
        "Opening.",
        "IV Q1?", "Model A1.",
        "IV Q2?", "Model A2 — this is the checkpoint response.",
        "Closing summary.",
    ]
    mock_client = _make_client(responses)

    with patch("core.experiment._resolve_client", side_effect=lambda _: mock_client):
        with patch("core.search.web_search", new=AsyncMock(return_value="")):
            result = asyncio.run(run_condition(
                experiment=exp,
                condition="baseline",
                timestamp="20260311_ckpt",
            ))

    # config.yaml has checkpoint_turns: [2], so turn 2 artifact should exist
    results_dir = Path(os.environ["RESULTS_DIR"])
    artifacts = list(results_dir.rglob("turn_2_baseline.md"))
    assert len(artifacts) == 1
    assert "checkpoint response" in artifacts[0].read_text()

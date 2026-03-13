"""
Experiment runner — top-level sequencing for a single test condition run.

Responsibilities:
  1. Load global config (config.yaml) and merge per-test overrides.
  2. Resolve prompts from the prompt store (or fall back to config/prompts/*.txt).
  3. Load model entries + their API keys from the model store.
  4. Instantiate OllamaClient for each role (pass1, pass2, interviewer).
  5. For proxy: set up neo4j via Neo4jManager before the run.
  6. Run the condition loop (run_baseline or run_proxy).
  7. Deliver the closing prompt to the test model and log the response.
  8. Run the artifact extractor to write checkpoint and closing artifacts.
  9. Save the neo4j snapshot (proxy runs only).
 10. Write summary.json and return the summary dict.

The closing prompt is delivered by this runner (not inside the condition loops)
so that it can be applied identically regardless of condition.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Callable, Awaitable

import yaml

from core.baseline import run_baseline
from core.extractor import extract_artifacts
from core.logger import RunLogger
from core.neo4j_manager import Neo4jManager
from core.ollama_client import OllamaClient
from core.proxy_condition import run_proxy
from core.substrate_client import SubstrateClient
from store.experiments import ExperimentConfig
from store.keys import get_model_key
from store.models import get_model
from store.prompts import get_prompt


def load_global_config() -> dict:
    """Load and return the experiment section of config.yaml."""
    path = Path(os.environ.get("CONFIG_PATH", "/app/config/config.yaml"))
    with path.open() as f:
        data = yaml.safe_load(f)
    return data.get("experiment", {})


def _prompts_dir() -> Path:
    config_path = Path(os.environ.get("CONFIG_PATH", "/app/config/config.yaml"))
    return config_path.parent / "prompts"


def _resolve_prompt(prompt_id: str | None, fallback_file: str) -> str:
    """Load prompt from store (by id) or fall back to the default .txt file."""
    if prompt_id:
        entry = get_prompt(prompt_id)
        if entry:
            return entry.content
    return (_prompts_dir() / fallback_file).read_text().strip()


def _resolve_client(model_id: str) -> OllamaClient:
    """Look up a model entry + its API key and return a configured OllamaClient."""
    model = get_model(model_id)
    if model is None:
        raise ValueError(f"Model not found in store: {model_id}")
    api_key = get_model_key(model_id) or ""
    return OllamaClient(
        endpoint_url=model.endpoint_url,
        model_identifier=model.model_identifier,
        api_key=api_key,
    )


async def run_condition(
    experiment: ExperimentConfig,
    condition: str,
    timestamp: str,
    run_id: str | None = None,
    neo4j_manager: Neo4jManager | None = None,
    db_source: str = "new",
    substrate_base_url: str = "http://substrate-api:8000",
    event_callback: Callable[[str, dict], Awaitable[None]] | None = None,
    should_finish: Callable[[], bool] | None = None,
) -> dict:
    """
    Run one condition (baseline or proxy) for the given experiment.

    condition      : "baseline" or "proxy"
    timestamp      : run identifier used for log file names, e.g. "20260311_142530"
    neo4j_manager  : optional Neo4jManager; defaults to Neo4jManager() if None
    db_source      : "new" (fresh substrate) or a prior proxy run timestamp to restore
    substrate_base_url : internal URL of the substrate-api service

    Returns the summary dict (also written to summary.json in the run directory).
    """
    # ── Load + merge config ───────────────────────────────────────────────────
    cfg = load_global_config()

    turn_limit = (
        experiment.turn_limit
        if experiment.turn_limit is not None
        else cfg.get("turn_limit", 200)
    )
    context_window = (
        experiment.context_window
        if experiment.context_window is not None
        else cfg.get("context_window", 256000)
    )
    compaction_threshold = (
        experiment.compaction_threshold_fraction
        if experiment.compaction_threshold_fraction is not None
        else cfg.get("compaction_threshold_fraction", 0.80)
    )
    pass1_activation = (
        experiment.pass1_activation_fraction
        if experiment.pass1_activation_fraction is not None
        else cfg.get("pass1_activation_fraction", 0.50)
    )
    turn_pause_min = (
        experiment.turn_pause_min_seconds
        if experiment.turn_pause_min_seconds is not None
        else cfg.get("turn_pause_min_seconds", 10)
    )
    turn_pause_max = (
        experiment.turn_pause_max_seconds
        if experiment.turn_pause_max_seconds is not None
        else cfg.get("turn_pause_max_seconds", 20)
    )
    # Ensure max >= min in case of misconfiguration
    turn_pause_max = max(turn_pause_min, turn_pause_max)
    checkpoint_turns = cfg.get("checkpoint_turns", [50, 100, 150, 200])

    # ── Resolve prompts ───────────────────────────────────────────────────────
    opening_question = _resolve_prompt(
        experiment.opening_question_prompt_id, "opening_question.txt"
    )
    test_model_system = _resolve_prompt(
        experiment.test_model_system_prompt_id, "test_model_system.txt"
    )
    interviewer_system = _resolve_prompt(
        experiment.interviewer_system_prompt_id, "interviewer_system.txt"
    )
    closing_prompt = _resolve_prompt(
        experiment.closing_prompt_id, "closing_prompt.txt"
    )
    compaction_prompt = _resolve_prompt(
        experiment.compaction_prompt_id, "compaction_prompt.txt"
    )
    pass1_system = _resolve_prompt(None, "pass1_system.txt")

    # ── Create clients ────────────────────────────────────────────────────────
    pass2_client = _resolve_client(experiment.pass2_model_id)
    interviewer_client = _resolve_client(experiment.interviewer_model_id)

    # ── Create logger ─────────────────────────────────────────────────────────
    logger = RunLogger(experiment.name, condition, timestamp=timestamp)

    # ── Neo4j setup (proxy only) ──────────────────────────────────────────────
    if condition == "proxy":
        mgr = neo4j_manager or Neo4jManager()
        if db_source == "new":
            await mgr.prepare_fresh()
        else:
            await mgr.prepare_from_snapshot(experiment.id, db_source)

    substrate_client = SubstrateClient(base_url=substrate_base_url)

    # ── Build streaming turn callback ─────────────────────────────────────────
    async def _on_turn(turn: int, question: str, response: str, stats: dict) -> None:
        if event_callback:
            event_type = "opening_complete" if turn == 0 else "turn_complete"
            await event_callback(event_type, {
                "turn": turn,
                "turn_limit": turn_limit,
                "question": question,
                "response": response,
                **stats,
            })

    # ── Run the condition loop ────────────────────────────────────────────────
    _cancelled_result: dict | None = None

    def _on_cancel(partial: dict) -> None:
        nonlocal _cancelled_result
        _cancelled_result = partial

    try:
        if condition == "baseline":
            run_result = await run_baseline(
                pass2_client=pass2_client,
                interviewer_client=interviewer_client,
                test_model_system_prompt=test_model_system,
                interviewer_system_prompt=interviewer_system,
                opening_question=opening_question,
                closing_prompt=closing_prompt,
                compaction_prompt=compaction_prompt,
                context_window=context_window,
                compaction_threshold_fraction=compaction_threshold,
                turn_limit=turn_limit,
                turn_pause_min_seconds=turn_pause_min,
                turn_pause_max_seconds=turn_pause_max,
                checkpoint_turns=checkpoint_turns,
                logger=logger,
                on_turn_complete=_on_turn,
                on_cancel=_on_cancel,
                should_finish=should_finish,
            )
        else:
            pass1_client = _resolve_client(experiment.pass1_model_id)
            run_result = await run_proxy(
                pass1_client=pass1_client,
                pass2_client=pass2_client,
                interviewer_client=interviewer_client,
                substrate_client=substrate_client,
                test_model_system_prompt=test_model_system,
                interviewer_system_prompt=interviewer_system,
                pass1_system_prompt=pass1_system,
                opening_question=opening_question,
                closing_prompt=closing_prompt,
                context_window=context_window,
                pass1_activation_fraction=pass1_activation,
                turn_limit=turn_limit,
                turn_pause_min_seconds=turn_pause_min,
                turn_pause_max_seconds=turn_pause_max,
                checkpoint_turns=checkpoint_turns,
                logger=logger,
                on_turn_complete=_on_turn,
                on_cancel=_on_cancel,
                should_finish=should_finish,
            )
    except asyncio.CancelledError:
        # Write a partial summary so the run appears in the sidebar after restart
        partial = _cancelled_result or {"turns_completed": 0}
        pass2_model_entry = get_model(experiment.pass2_model_id)
        interviewer_model_entry = get_model(experiment.interviewer_model_id)
        pass1_model_entry = (
            get_model(experiment.pass1_model_id) if condition == "proxy" else None
        )
        cancelled_summary = {
            "run_id": run_id or timestamp,
            "experiment_id": experiment.id,
            "experiment_name": experiment.name,
            "condition": condition,
            "timestamp": timestamp,
            "status": "cancelled",
            "pass1_model": pass1_model_entry.model_identifier if pass1_model_entry else None,
            "pass2_model": pass2_model_entry.model_identifier if pass2_model_entry else None,
            "interviewer_model": interviewer_model_entry.model_identifier if interviewer_model_entry else None,
            "parameters": {
                "turn_limit": turn_limit,
                "context_window": context_window,
                "compaction_threshold_fraction": compaction_threshold,
                "pass1_activation_fraction": pass1_activation,
                "turn_pause_min_seconds": turn_pause_min,
                "turn_pause_max_seconds": turn_pause_max,
                "checkpoint_turns": checkpoint_turns,
            },
            "database_source": db_source if condition == "proxy" else None,
            **partial,
            "artifacts": {},
        }
        summary_path = logger.run_dir / "summary.json"
        summary_path.write_text(json.dumps(cancelled_summary, indent=2))
        logger.transcript_cancelled(partial.get("turns_completed", 0))
        raise

    # ── Deliver closing prompt ────────────────────────────────────────────────
    # Appended to the full conversation history so the model can reflect on
    # the actual exchange that took place.
    closing_messages = run_result.get("final_history", [
        {"role": "system", "content": test_model_system},
    ]) + [{"role": "user", "content": closing_prompt}]
    logger.log_api_request(
        turn_limit + 1, "closing", pass2_client.model_identifier, closing_messages
    )
    closing_raw = await pass2_client.chat(closing_messages)
    closing_text = (closing_raw.get("content") or "").strip()
    closing_usage = closing_raw.get("usage", {})
    logger.log_api_response(
        turn_limit + 1, "closing", pass2_client.model_identifier,
        closing_text, [], closing_usage,
    )
    logger.log_closing(closing_prompt, closing_text)
    logger.transcript_closing(closing_prompt, closing_text)

    # ── Extract artifacts ─────────────────────────────────────────────────────
    artifacts = extract_artifacts(
        raw_path=logger.raw_path,
        condition=condition,
        run_dir=logger.run_dir,
        checkpoint_turns=checkpoint_turns,
    )

    # ── Save neo4j snapshot (proxy only) ──────────────────────────────────────
    if condition == "proxy":
        mgr = neo4j_manager or Neo4jManager()
        mgr.save_snapshot(experiment.id, timestamp)

    # ── Resolve model names for the audit record ───────────────────────────────
    pass2_model_entry = get_model(experiment.pass2_model_id)
    interviewer_model_entry = get_model(experiment.interviewer_model_id)
    pass1_model_entry = (
        get_model(experiment.pass1_model_id) if condition == "proxy" else None
    )

    # ── Write summary.json ────────────────────────────────────────────────────
    summary = {
        "run_id": run_id or timestamp,
        "experiment_id": experiment.id,
        "experiment_name": experiment.name,
        "condition": condition,
        "timestamp": timestamp,
        "pass1_model": pass1_model_entry.model_identifier if pass1_model_entry else None,
        "pass2_model": pass2_model_entry.model_identifier if pass2_model_entry else None,
        "interviewer_model": interviewer_model_entry.model_identifier if interviewer_model_entry else None,
        "parameters": {
            "turn_limit": turn_limit,
            "context_window": context_window,
            "compaction_threshold_fraction": compaction_threshold,
            "pass1_activation_fraction": pass1_activation,
            "turn_pause_min_seconds": turn_pause_min,
            "turn_pause_max_seconds": turn_pause_max,
            "checkpoint_turns": checkpoint_turns,
        },
        "database_source": db_source if condition == "proxy" else None,
        **run_result,
        "artifacts": artifacts,
    }

    summary_path = logger.run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary

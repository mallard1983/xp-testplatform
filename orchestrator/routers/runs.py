"""
Runs router — start/stop runs, SSE live stream, completed run data.

Run lifecycle:
  POST   /api/runs                       enqueue a run (starts immediately if nothing active)
  GET    /api/runs                       list all runs (active + queued + completed from disk)
  GET    /api/runs/{run_id}              get run status
  DELETE /api/runs/{run_id}              cancel an active run, or dequeue a queued run
  GET    /api/runs/{run_id}/stream       SSE event stream (connects immediately; events flow when run starts)
  GET    /api/runs/{run_id}/transcript   transcript .md for a completed run
  GET    /api/runs/{run_id}/events       all logged events from raw JSONL

Run IDs are UUIDs for active/queued runs, written into summary.json so they
survive server restarts when the disk is scanned for completed runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from core.run_queue import run_queue

router = APIRouter(prefix="/api/runs", tags=["runs"])

# In-memory registry: run_id → run record (active + queued)
_active_runs: dict[str, dict] = {}


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "/app/results"))


# ── Request models ─────────────────────────────────────────────────────────────

class StartRunRequest(BaseModel):
    experiment_id: str
    condition: str          # "baseline" or "proxy"
    db_source: str = "new"  # "new" or prior proxy run timestamp


# ── Completed run scanning ─────────────────────────────────────────────────────

def _scan_completed_runs() -> list[dict]:
    """Scan results directory for completed runs (summary.json files)."""
    completed = []
    results = _results_dir()
    if not results.exists():
        return completed
    for summary_path in sorted(results.rglob("summary.json"), reverse=True):
        try:
            data = json.loads(summary_path.read_text())
            data.setdefault("status", "complete")
            data["_run_dir"] = str(summary_path.parent)
            completed.append(data)
        except Exception:
            continue
    return completed


# ── Run execution (called by the queue) ────────────────────────────────────────

async def _execute_run(run_id: str) -> None:
    """Start the asyncio task for a run that has been dequeued."""
    run_record = _active_runs.get(run_id)
    if not run_record:
        run_queue.notify_complete()
        return

    events_queue = run_record["events_queue"]

    async def _event_callback(event_type: str, data: dict) -> None:
        await events_queue.put({"type": event_type, **data})

    async def _run_task() -> None:
        from core.experiment import run_condition
        try:
            run_record["status"] = "running"
            run_record["finish_requested"] = False
            await events_queue.put({
                "type": "run_started",
                "run_id": run_id,
                "condition": run_record["condition"],
                "timestamp": run_record["timestamp"],
            })
            summary = await run_condition(
                experiment=run_record["_exp"],
                condition=run_record["condition"],
                timestamp=run_record["timestamp"],
                run_id=run_id,
                db_source=run_record["db_source"],
                event_callback=_event_callback,
                should_finish=lambda: run_record.get("finish_requested", False),
            )
            run_record["status"] = "complete"
            run_record["summary"] = summary
            await events_queue.put({"type": "run_complete", "summary": summary})
        except asyncio.CancelledError:
            run_record["status"] = "cancelled"
            await events_queue.put({"type": "cancelled"})
        except Exception as exc:
            logging.exception("Run %s failed", run_id)
            run_record["status"] = "error"
            run_record["error"] = str(exc)
            await events_queue.put({"type": "error", "message": str(exc)})
        finally:
            await events_queue.put(None)  # sentinel — closes the SSE stream
            run_queue.notify_complete()

    task = asyncio.create_task(_run_task())
    run_record["task"] = task


# Wire the queue to the execution function
run_queue.set_execute_fn(_execute_run)


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("")
def list_runs():
    """Return all runs: queued + active (in memory) + completed (from disk)."""
    active_ids = set()
    active = []
    for run_id, r in _active_runs.items():
        active_ids.add(run_id)
        active.append({
            "run_id": run_id,
            "experiment_id": r["experiment_id"],
            "experiment_name": r["experiment_name"],
            "condition": r["condition"],
            "timestamp": r["timestamp"],
            "status": r["status"],
        })

    for r in _scan_completed_runs():
        if r.get("run_id") not in active_ids:
            active.append({
                "run_id": r.get("run_id"),
                "experiment_id": r.get("experiment_id"),
                "experiment_name": r.get("experiment_name"),
                "condition": r.get("condition"),
                "timestamp": r.get("timestamp"),
                "status": r.get("status", "complete"),
            })

    return active


@router.post("", status_code=201)
async def start_run(body: StartRunRequest):
    """Enqueue a run. Starts immediately if nothing is active, otherwise waits."""
    from store.experiments import get_experiment

    exp = get_experiment(body.experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    events_queue: asyncio.Queue = asyncio.Queue()

    run_record: dict = {
        "run_id": run_id,
        "experiment_id": body.experiment_id,
        "experiment_name": exp.name,
        "condition": body.condition,
        "timestamp": timestamp,
        "db_source": body.db_source,
        "status": "queued",
        "events_queue": events_queue,
        "_exp": exp,            # kept for _execute_run; not exposed in API responses
        "task": None,
        "summary": None,
        "error": None,
    }
    _active_runs[run_id] = run_record
    run_queue.enqueue(run_id)

    return {"run_id": run_id, "timestamp": timestamp, "status": "queued"}


@router.get("/{run_id}")
def get_run(run_id: str):
    """Get status for an active or queued run."""
    if run_id in _active_runs:
        r = _active_runs[run_id]
        return {k: v for k, v in r.items() if k not in ("events_queue", "task", "_exp")}
    raise HTTPException(status_code=404, detail="Run not found in active registry")


@router.delete("/{run_id}", status_code=204)
def cancel_run(run_id: str):
    """Cancel an active run, or remove a queued run before it starts."""
    if run_id not in _active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    run = _active_runs[run_id]
    if run["status"] == "queued":
        run_queue.remove(run_id)
        del _active_runs[run_id]
    else:
        task = run.get("task")
        if task and not task.done():
            task.cancel()
            run["status"] = "cancelled"
    return None


@router.patch("/{run_id}", status_code=204)
def finish_run(run_id: str):
    """Signal a running run to stop after the current turn and deliver the closing prompt."""
    if run_id not in _active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    run = _active_runs[run_id]
    if run["status"] == "running":
        run["finish_requested"] = True
    return None


@router.get("/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE stream for a run. Safe to connect before the run starts — events flow when it does."""
    if run_id not in _active_runs:
        raise HTTPException(status_code=404, detail="Run not in active registry")

    queue = _active_runs[run_id]["events_queue"]

    async def _generator() -> AsyncGenerator[dict, None]:
        while True:
            event = await queue.get()
            if event is None:  # sentinel — run finished
                break
            yield {"data": json.dumps(event)}

    return EventSourceResponse(_generator())


@router.get("/{run_id}/transcript")
def get_transcript(run_id: str):
    """Return the transcript .md for a completed run."""
    path = _find_file(run_id, "transcript_*.md")
    if path is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return PlainTextResponse(path.read_text())


@router.get("/{run_id}/events")
def get_events(run_id: str):
    """Return all raw events from the JSONL log for a completed run."""
    path = _find_file(run_id, "raw_*.jsonl")
    if path is None:
        raise HTTPException(status_code=404, detail="Raw log not found")
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_file(run_id: str, pattern: str) -> Path | None:
    """Locate a file in any completed run whose summary.json contains run_id."""
    for summary_path in _results_dir().rglob("summary.json"):
        try:
            data = json.loads(summary_path.read_text())
            if data.get("run_id") == run_id:
                matches = list(summary_path.parent.glob(pattern))
                return matches[0] if matches else None
        except Exception:
            continue
    return None

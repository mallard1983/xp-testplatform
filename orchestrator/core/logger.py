"""
Dual-log system — raw JSONL and human-readable transcript (.md).

Raw log: every event that passes through the orchestrator, one JSON object per line.
  - api_request, api_response, tool_call, search_result
  - compaction_event, turn_complete, checkpoint, closing, error

Transcript log: turn number, speaker, message only — clean and readable.
  Formatted as Markdown for direct use as evaluator input.

The key scrubber from store.keys.scrub() is applied to ALL writes.
File naming: raw_{test_name}_{condition}_{timestamp}.jsonl
             transcript_{test_name}_{condition}_{timestamp}.md
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from store.keys import scrub


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "/app/results"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_json(obj: dict) -> str:
    """Serialize to JSON then scrub all key values."""
    return scrub(json.dumps(obj, ensure_ascii=False))


class RunLogger:
    """
    Logger for a single run (one test, one condition).
    Instantiate at the start of a run; call close() when done.
    """

    def __init__(self, test_name: str, condition: str, timestamp: str | None = None):
        self.test_name = test_name
        self.condition = condition  # "baseline" or "proxy"
        self.ts = timestamp or _timestamp()

        slug = f"{_slugify(test_name)}_{condition}_{self.ts}"
        run_dir = _results_dir() / _slugify(test_name) / self.ts
        run_dir.mkdir(parents=True, exist_ok=True)

        self._raw_path = run_dir / f"raw_{slug}.jsonl"
        self._transcript_path = run_dir / f"transcript_{slug}.md"

        # Create raw log file immediately so it exists for auditors even before the first event
        self._raw_path.touch()

        # Write transcript header
        header = (
            f"# Transcript — {test_name} — {condition.capitalize()}\n\n"
            f"**Run:** {self.ts}  \n"
            f"**Condition:** {condition}  \n\n"
            f"---\n\n"
        )
        self._transcript_path.write_text(scrub(header))

    # ── Raw log events ────────────────────────────────────────────────────────

    def log_api_request(self, turn: int, pass_label: str, model: str, messages: list[dict], tools: list[dict] | None = None):
        self._raw({"event": "api_request", "turn": turn, "pass": pass_label,
                   "model": model, "messages": messages, "tools": tools})

    def log_api_response(self, turn: int, pass_label: str, model: str, content: str | None,
                         tool_calls: list, usage: dict):
        self._raw({"event": "api_response", "turn": turn, "pass": pass_label,
                   "model": model, "content": content, "tool_calls": tool_calls, "usage": usage})

    def log_tool_call(self, turn: int, pass_label: str, tool_name: str, args: dict,
                      result: str, error: bool = False):
        self._raw({"event": "tool_call", "turn": turn, "pass": pass_label,
                   "tool": tool_name, "args": args, "result": result, "error": error})

    def log_search(self, turn: int, pass_label: str, query: str, results: str):
        self._raw({"event": "search_result", "turn": turn, "pass": pass_label,
                   "query": query, "results": results})

    def log_compaction(self, turn: int, summary: str, tokens_before: int):
        self._raw({"event": "compaction_event", "turn": turn,
                   "tokens_before": tokens_before, "summary": summary})

    def log_opening_complete(self, question: str, response: str, usage: dict,
                             stats: dict | None = None):
        self._raw({"event": "opening_complete", "question": question,
                   "response": response, "usage": usage, **(stats or {})})

    def log_turn_complete(self, turn: int, interviewer_question: str,
                          model_response: str, usage: dict,
                          stats: dict | None = None):
        self._raw({"event": "turn_complete", "turn": turn,
                   "question": interviewer_question,
                   "response": model_response, "usage": usage, **(stats or {})})

    def log_checkpoint(self, turn: int, response: str):
        self._raw({"event": "checkpoint", "turn": turn, "response": response})

    def log_closing(self, prompt: str, response: str):
        self._raw({"event": "closing", "prompt": prompt, "response": response})

    def log_error(self, turn: int | None, message: str, detail: str = ""):
        self._raw({"event": "error", "turn": turn, "message": message, "detail": detail})

    # ── Transcript ────────────────────────────────────────────────────────────

    def transcript_turn(self, turn: int, interviewer_question: str, model_response: str):
        block = (
            f"## Turn {turn}\n\n"
            f"**Interviewer:** {interviewer_question.strip()}\n\n"
            f"**Model:** {model_response.strip()}\n\n"
            f"---\n\n"
        )
        self._append_transcript(block)

    def transcript_closing(self, prompt: str, response: str):
        block = (
            f"## Closing\n\n"
            f"**Orchestrator:** {prompt.strip()}\n\n"
            f"**Model:** {response.strip()}\n\n"
        )
        self._append_transcript(block)

    def transcript_compaction_note(self, turn: int):
        note = f"*[Compaction event at turn {turn} — context window condensed]*\n\n"
        self._append_transcript(note)

    # ── Artifact paths ────────────────────────────────────────────────────────

    @property
    def raw_path(self) -> Path:
        return self._raw_path

    @property
    def transcript_path(self) -> Path:
        return self._transcript_path

    @property
    def run_dir(self) -> Path:
        return self._raw_path.parent

    # ── Internal ──────────────────────────────────────────────────────────────

    def _raw(self, event: dict) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        event["_ts"] = ts
        line = _safe_json(event) + "\n"
        with self._raw_path.open("a") as f:
            f.write(line)

    def _append_transcript(self, text: str) -> None:
        with self._transcript_path.open("a") as f:
            f.write(scrub(text))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")

# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — 2026-03-12

### Fixed

- **Token accounting now uses API-reported counts throughout.** Compaction threshold checks (baseline) and Pass 1 activation threshold checks (proxy) previously used a local tiktoken estimate of the message array, which did not include tool schema tokens. Both now use the `prompt_tokens` value returned by the API on the previous turn — the same ground-truth number the model actually processed.
- `needs_compaction()` signature changed to accept a pre-computed token count (`int`) rather than a message array, removing the misleading implication that a local estimate was authoritative.

### Changed

- Per-turn stats now emit `current_context` (API-reported `prompt_tokens` from the last Pass 2 call) in place of `turn_tokens` (local estimate). This is the value used for compaction/activation decisions and is what a real production system such as Claude Code or Open Code would track.
- **Context stat in the header bar now shows for both baseline and proxy conditions.** For baseline it reflects context after any compaction; for proxy it reflects the Pass 2 prompt size (compressed context when Pass 1 is active, full history when below threshold).
- `pass1_tokens` and `pass2_tokens` in per-turn stats continue to accumulate API-reported prompt + completion tokens for their respective roles. `total_tokens` remains the sum of all non-interviewer calls.

---

## [0.1.1] — 2026-03-12

### Added

- Run queue: multiple runs can now be queued and execute sequentially without manual intervention. A queue panel appears in the sidebar footer when runs are pending, with per-item remove and a Clear All button. The **+ Both** button queues baseline followed by proxy in one action, enabling unattended overnight runs.
- Graceful shutdown (**Finish** button): signals a running run to stop cleanly after the current turn completes. The closing prompt is delivered and all artifacts are written before the run exits — no partial output, no lost data. This is distinct from the **Stop** button, which cancels immediately. Available via `PATCH /api/runs/{run_id}`.

### Fixed

- Header title colour corrected.

### Docs

- Added contribution guidelines to the results publishing section of the README.

---

## [0.1.0] — 2026-03-11

Initial release.

- Two-condition experiment framework: baseline (full history + compaction) and proxy (Pass 1 substrate retrieval + Pass 2 with injected context).
- Configurable models, prompts, and experiment parameters via the UI — no config file editing required for normal use.
- Live SSE streaming of turn events to the browser; per-turn stats in the header bar.
- Replay mode: step through any completed run turn by turn with stats updating as you scrub.
- Neo4j substrate isolation between proxy runs: each run gets a fresh or copied database.
- Run output: raw JSONL event log, Markdown transcript, `summary.json`, checkpoint and closing artifacts.
- Blinded evaluation package builder (`core.extractor.build_evaluation_package`).
- Smoke test script (`tools/smoke_test.py`) for end-to-end validation against a live stack.

# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — 2026-03-13

### Added

- **Retry logic for transient API failures.** The HTTP client now retries up to 3 times on 429 (rate limited), 5xx server errors, timeouts, and connection resets. On 429, the `Retry-After` response header is respected (defaulting to 30s if absent). On 5xx, timeout, or connection error, retries back off at 5s, 10s, 15s. 4xx client errors (other than 429) fail immediately. A run only fails if all retries are exhausted.
- **`User-Agent: XP-TestPlatform/1.0` header** on all model API requests. Avoids being treated as raw script traffic by cloud endpoints that inspect headers.
- **Randomised turn pause.** The fixed `turn_pause_seconds` parameter has been replaced with `turn_pause_min_seconds` (default 10) and `turn_pause_max_seconds` (default 20). Each inter-turn sleep picks a random value in that range, making the request cadence less mechanical and less likely to trigger rate limiting. Both values are configurable as global defaults and as per-experiment overrides.
- **Structured error logging to JSONL.** Unhandled exceptions in the baseline and proxy turn loops now write an `error` event (with turn number and message) to the raw log before propagating. Previously errors were only visible in docker logs and the UI; now they are part of the permanent run record.
- **Context fill bar with threshold marker in the header.** The raw "Context" token number has been replaced with a labelled fill bar showing current context as a fraction and percentage of the configured context window (e.g. `42K / 256K (16%)`). An amber tick mark sits at the compaction threshold (baseline) or Pass 1 activation threshold (proxy), so you can see at a glance how far the conversation is from triggering each mechanism. The bar updates live during a run and is fully accurate during replay scrubbing.
- **`context_window`, `activation_threshold` (proxy), and `compaction_threshold` (baseline) are now emitted in every `turn_complete` event**, both to the JSONL log and the SSE stream. This makes the threshold values available to any downstream consumer without having to reconstruct them from config.

### Fixed

- **Token accounting now uses API-reported counts throughout.** Compaction threshold checks (baseline) and Pass 1 activation threshold checks (proxy) previously used a local tiktoken estimate of the message array, which did not include tool schema tokens. Both now use the `prompt_tokens` value returned by the API on the previous turn — the same ground-truth number the model actually processed.
- `needs_compaction()` signature changed to accept a pre-computed token count (`int`) rather than a message array, removing the misleading implication that a local estimate was authoritative.
- **`current_context` now correctly tracks the actual context window size when tool calls are made.** `run_with_tools` loops through multiple API calls per turn when tools are invoked. Previously, `prompt_tokens` was accumulated across all iterations, inflating `current_context` by roughly 2× on any turn with at least one tool call and causing false threshold comparisons. The fix: `run_with_tools` now tracks `last_prompt_tokens` (the final iteration's prompt count only) separately from the billing total, and all threshold checks use `last_prompt_tokens`. The accumulated total continues to be used for cost tracking. Both baseline and proxy were affected.
- **Closing prompt now receives the full conversation history.** The closing exchange was being sent with only the system prompt and closing question — no conversation history — so the model had no context for its closing reflection. The fix passes `final_history` back from both condition runners and appends the closing prompt to the real message array. Prompt tokens at closing are now ~42K (full conversation) rather than ~180 (system + question only).

### Changed

- Fixed turn pause in proxy condition below-threshold path firing before tool/response logging rather than after — logging now always completes before the sleep.
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

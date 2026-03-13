# Notes — Qwen3.5 Cloud 25-Turn Test

This run was conducted during early platform development and **did not complete correctly**. It is preserved as a development record but should not be used for evaluation or comparison purposes.

## Run inventory

| Timestamp | Condition | Turns | Status | Notes |
|-----------|-----------|-------|--------|-------|
| 20260313_023709 | baseline | unknown | crashed | No summary.json — run terminated before completion. Baseline transcript is partial; no proxy run was attempted. |

## Why this run is invalid

- The run crashed before producing a `summary.json`, most likely due to the context overflow bug present at the time: compaction threshold was evaluated using a local token estimate that did not account for tool schema tokens, causing the true context usage to exceed the model's window before compaction fired.
- Only one condition (baseline) was attempted; without a matching proxy run the comparison is impossible.
- This run predates the fix to API-reported token counting (see CHANGELOG [Unreleased]).

## Model

- **Model:** Qwen3.5 (cloud API)
- **Intended turn limit:** 25

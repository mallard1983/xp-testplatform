# Notes — Qwen3-Coder-Next (LM Studio) - Baseline

These runs were conducted during early platform development and **did not complete correctly**. They are preserved as a development record but should not be used for evaluation or comparison purposes.

## Run inventory

| Timestamp | Condition | Turns | Status | Notes |
|-----------|-----------|-------|--------|-------|
| 20260312_051019 | baseline + proxy | unknown | crashed | No summary.json — run terminated before completion. Both baseline and proxy transcripts are partial. |
| 20260312_111902 | baseline + proxy | unknown | crashed | No summary.json — run terminated before completion. Both baseline and proxy transcripts are partial. |
| 20260312_145743 | baseline only | unknown | crashed | No summary.json — run terminated before completion. Baseline transcript is partial; no proxy output. |
| 20260312_181710 | baseline | 5 | cancelled | Manually cancelled after 5 turns. |
| 20260312_194711 | baseline | 38 | incomplete | Run terminated at 38 of 100 turns. Model used as its own interviewer (same model for both roles — not the intended configuration). No compaction events recorded despite 1.75M prompt tokens consumed, which indicates the context overflow bug that has since been fixed: compaction threshold was evaluated using a local token estimate that did not account for tool schema tokens, causing the true context usage to exceed the model's window before compaction fired. |

## Why these runs are invalid

- Runs 051019 through 145743 crashed before producing a `summary.json`, most likely due to the same context overflow issue as 194711.
- Run 194711 used the test model as its own interviewer, which is not the intended experimental configuration and produces a different conversation dynamic.
- All runs predate the fix to API-reported token counting (see CHANGELOG [Unreleased]).

## Model

- **Model:** qwen/qwen3-coder-next via LM Studio
- **Context window configured:** 256,000 tokens
- **Compaction threshold:** 80%

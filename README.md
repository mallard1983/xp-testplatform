# XP Framework Test Platform

A self-contained research testbed for controlled LLM experiments. Originally built to evaluate a graph-substrate memory management system, the platform is general enough to support prompt engineering studies, context management research, and any experiment that benefits from structured, repeatable multi-turn model conversations.

---

## Background

This platform was built to answer a specific question in the development of the **XP Framework** — a memory management system for large language models that maintains a curated, semantically-indexed substrate of prior exchanges rather than relying on the model's raw context window.

The core hypothesis: *a model with curated rolling context reasons better across a long conversation than a model managing context through autocompaction*.  Autocompaction preserves surface content but discards the semantic relationships and accumulated reasoning structure that make prior exchanges useful — what gets summarized away is often exactly what a later turn needs.

To test this, the platform runs two conditions against the same opening question, same interviewer, same turn count, and produces blinded output artifacts for side-by-side evaluation. But the scaffolding that makes this possible — configurable models, prompt stores, live streaming, replay, per-turn stats — turns out to be useful for a much broader class of experiments.

**Other uses this platform is well-suited for:**
- Prompt engineering: compare system prompt variants across controlled multi-turn conversations
- Context window research: observe how different context management strategies affect response quality over time
- Model comparison: run the same experiment against different model backends and compare transcripts
- Interviewer design: experiment with interviewer prompt styles and observe downstream effects on the conversation

---

## How It Works

Every experiment pits two conditions against each other:

### Baseline condition

The test model receives the full conversation history every turn. When the accumulated context approaches the configured threshold (default 80% of the context window), the orchestrator compacts the history in-place — asking the model to summarise prior turns — and replaces the message array with the compressed version. The number of compaction events and current context size are tracked and displayed live.

```
Interviewer ──► Orchestrator ──► Test model (full history)
                │                    │
                └─ tokens ≥ 80%?      └─► compaction ──► compressed history
```

### Proxy condition

The test model never sees the raw history. Instead, the orchestrator manages a graph-substrate (Neo4j + semantic embeddings) that accumulates exchanges as ThoughtStreams. Each turn, if the conversation has grown past the activation threshold (default 50% of context window), Pass 1 runs first — it queries the substrate via MCP tools and returns a curated `CONTEXT_START...CONTEXT_END` block. Pass 2 then runs with only that curated context injected, not the full history. Below the threshold, Pass 2 runs directly with the full (still-short) history.

After each turn, the exchange is written back to the substrate asynchronously so it is available for future Pass 1 retrievals.

```
Interviewer ──► Orchestrator
                │
                ├─ tokens < 50% context window?
                │    └─► Pass 2 directly (full history)
                │
                └─ tokens ≥ 50% context window?
                     ├─► Pass 1 (substrate MCP tools → curated context block)
                     └─► Pass 2 (curated context injected into system prompt)
                              └─► async write-back to substrate
```

Token accounting is intentional: **the interviewer is not counted** — it stands in for a human user and its tokens are not a cost of the system under test. Baseline counts test model tokens only. Proxy counts Pass 1 + Pass 2 tokens separately, since Pass 1 at ~100K tokens per activation adds up quickly and needs to be visible alongside Pass 2.

All token counts use the `prompt_tokens` and `completion_tokens` values returned directly by the API — not local estimates. When tool calls occur within a turn, multiple API calls are made; the **Context** stat reflects only the final call's `prompt_tokens` (the actual context window fill), while the billing totals (`total_tokens`, `pass1_tokens`, `pass2_tokens`) accumulate across all iterations. This is also the value used to decide when compaction or Pass 1 activation fires.

The closing prompt is delivered at the end of every run with the full conversation history attached, so the model can reflect accurately on what was discussed.

### API reliability

All model calls include a `User-Agent: XP-TestPlatform/1.0` header and are retried automatically on transient failures:

| Failure type | Behaviour |
|---|---|
| HTTP 429 (rate limited) | Waits for the `Retry-After` header value (default 30s), retries up to 3 times |
| HTTP 5xx (server error) | Backs off 5s / 10s / 15s, retries up to 3 times |
| Timeout | Backs off 5s / 10s / 15s, retries up to 3 times |
| Connection reset | Same backoff as timeout |
| HTTP 4xx (other) | Fails immediately — client errors won't resolve on retry |

If all retries are exhausted the run fails with a descriptive error written to both the JSONL log and the UI. The turn number and model identifier are included in every error message.

---

## Architecture

```
neo4j          — graph substrate (per-test isolation: stop / remount / restart between runs)
ollama-embed   — local nomic-embed-text embeddings (768-dim, never hits external APIs)
substrate-api  — FastAPI wrapper exposing Neo4j with semantic search + CRUD
mcp-server     — MCP protocol server exposing substrate tools to Pass 1
orchestrator   — FastAPI backend + React frontend (served together on port 3000)
```

All services communicate over an internal Docker network. Only the orchestrator port is exposed to the host. The Neo4j substrate is isolated between proxy test runs by stopping the container, swapping the bind-mount data directory, and restarting — so two proxy runs never share a substrate unless you explicitly choose to copy one.

---

## Quick Start

### Prerequisites

- Docker + Docker Compose v2
- An OpenAI-compatible LLM endpoint (Ollama, LM Studio, any compatible API)

### 1. Clone and configure

```bash
git clone <repo-url>
cd testplatform
cp .env.example .env
```

Edit `.env` and set a Neo4j password:

```bash
NEO4J_AUTH=neo4j/your-password-here
```

Everything else — models, prompts, experiment parameters — is configured through the UI after startup.

### 2. Start the stack

```bash
docker compose up -d
```

First run takes about two minutes: `ollama-embed` pulls and initialises the `nomic-embed-text` embedding model. Watch readiness with:

```bash
docker compose ps
```

All services should reach `healthy` before you start a run.

### 3. Open the UI

[http://localhost:3000](http://localhost:3000)

The port is configurable via `UI_PORT` in `.env`.

### 4. Add your model(s)

Open **⚙ Config → Models**. Each model entry needs:

| Field | Example |
|-------|---------|
| Display name | `Qwen3 32B Local` |
| Model identifier | `qwen3-32b` |
| Endpoint URL | `http://host.docker.internal:12345` |
| API key | Set via the **Key** button |

For LM Studio running on the host machine, use `http://host.docker.internal:12345` as the endpoint URL. For Ollama cloud, use your cloud endpoint. API keys are stored in `data/setup/keys.json` and are never committed.

Models can also be added inline while creating a test — click **+ New** next to any model selector in the test form.

### 5. Create a test

Open **Config → Tests → + New Test**. Select:
- **Pass 1 model** — the context retrieval model (proxy only; use the same as Pass 2 for the canonical experiment)
- **Pass 2 model** — the test model under evaluation
- **Interviewer model** — drives the questions; use any capable model, it doesn't affect the results being tested

Parameter overrides are optional. Leave them blank to inherit global defaults (shown as placeholders). Set them to override just for this experiment.

### 6. Run

Select your experiment in the sidebar and click **+ Baseline**, **+ Proxy**, or **+ Both**. The conversation streams live in the main view. Runs are queued automatically — if you start a second run while one is active it will wait and start when the first completes.

---

## The UI

### Sidebar

Lists all experiments and their completed runs. Click a run to load it.

Three buttons appear under each experiment:

| Button | What it does |
|--------|-------------|
| **+ Baseline** | Queues a baseline run. Starts immediately if nothing is running. |
| **+ Proxy** | Opens the substrate dialog, then queues a proxy run. |
| **+ Both** | Opens the substrate dialog, then queues baseline followed by proxy — useful for unattended overnight runs. |

Proxy runs can optionally copy the substrate from a prior completed proxy run (the source is never modified).

A **Queue panel** appears at the bottom of the sidebar whenever runs are pending, showing each item with a remove button and a Clear All option. Queued runs also appear in the run list with a slow-pulsing dot — clicking one selects it, and the live stream connection is already open waiting for it to start.

### Live run view

While a run is active, two controls appear in the run item:

| Button | What it does |
|--------|-------------|
| **Finish** | Stops after the current turn completes, delivers the closing prompt, and writes all artifacts. Use this for a clean early exit. |
| **Stop** | Cancels immediately and saves a partial run. No closing prompt is delivered. |

The main panel shows two tabs:

- **Chat** — the conversation as it unfolds, turn by turn, starting from the opening exchange (Turn 0). The interviewer's questions appear in green, the model's responses in blue, compaction events as inline notes.
- **Detail** — the raw event stream: every API request, response, tool call, and timing event as it arrives.

The header bar shows live stats updated after each turn:

| Stat | Baseline | Proxy |
|------|----------|-------|
| Turn / turn limit | ✓ | ✓ |
| Total tokens | ✓ | ✓ |
| Context fill bar (`current / window (%)` with threshold marker) | ✓ | ✓ |
| Compactions | ✓ (when > 0) | — |
| P1 Activations | — | ✓ |
| P1 Tokens | — | ✓ |
| P2 Tokens | — | ✓ |

The context fill bar shows how much of the configured context window is occupied, with an amber tick mark at the threshold where compaction (baseline) or Pass 1 (proxy) will fire. A progress bar tracks completion against the turn limit.

While scrubbing, the header bar updates to show the stats *as they were at that turn* — total tokens, context fill, compaction count, P1/P2 tokens — so you can watch exactly when costs accumulated and when each mechanism triggered.

### Replay

When you select a completed run, the Chat tab loads all events and shows the full conversation. A **Replay** bar appears at the bottom:

- **↩ Replay** enters step-through mode
- **←** / **→** step one turn at a time
- Click anywhere on the progress bar to jump to any turn
- **▶ Show All** exits replay and returns to the full conversation

While scrubbing, the header bar updates to show the stats as they were at that turn — context fill, total tokens, compaction count, P1/P2 tokens — so you can watch exactly how costs accumulated and when each mechanism triggered.

---

## Configuration Reference

Global defaults live in `config/config.yaml`. Per-experiment overrides are set in the UI and recorded in each run's `summary.json`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `turn_limit` | 200 | Conversation turns per run (opening exchange is Turn 0 and not counted) |
| `context_window` | 256000 | Token budget — set this to match your model's actual context window |
| `compaction_threshold_fraction` | 0.80 | Baseline: compaction fires when context exceeds this fraction of the window |
| `pass1_activation_fraction` | 0.50 | Proxy: Pass 1 activates when context exceeds this fraction of the window |
| `turn_pause_min_seconds` | 10 | Minimum pause between turns (seconds) |
| `turn_pause_max_seconds` | 20 | Maximum pause between turns (seconds) — actual pause is randomised between min and max each turn to avoid mechanical request patterns that can trigger rate limiting |
| `checkpoint_turns` | [50,100,150,200] | Turns where the model's response is extracted as a standalone artifact |

### Prompts

Six prompts drive the experiment. Built-in defaults live in `config/prompts/` and are visible (read-only) in **Config → Prompts → Built-in Defaults**. To override one, add an entry to the Prompt Store — the override applies to all runs until removed.

Prompts can also be added to the store inline from within the test form, without navigating away.

| Prompt file | Role |
|-------------|------|
| `opening_question.txt` | The question that opens every run |
| `test_model_system.txt` | System prompt for the test model (baseline and Pass 2) |
| `interviewer_system.txt` | System prompt for the interviewer model |
| `pass1_system.txt` | System prompt for Pass 1 (context retrieval) |
| `compaction_prompt.txt` | Instruction used during baseline compaction |
| `closing_prompt.txt` | Delivered to the test model after the final turn |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_AUTH` | `neo4j/changeme` | Neo4j credentials — change before any run |
| `UI_PORT` | `3000` | Host port for the UI and API |
| `LLM_TIMEOUT_SECONDS` | `600` | Per-request timeout for model calls — increase for slow local inference |
| `NEO4J_CONTAINER_NAME` | `xp-testplatform-neo4j-1` | Container name used by the orchestrator to stop/start Neo4j between proxy runs |

---

## Output Files

Every run produces a timestamped directory under `results/`:

```
results/{experiment-name}/{timestamp}/
  raw_{name}_{condition}_{timestamp}.jsonl     — every API call, tool call, and event
  transcript_{name}_{condition}_{timestamp}.md — clean turn-by-turn transcript
  summary.json                                  — parameters, token totals, run metadata
  artifacts/
    turn_50_{condition}.md                      — checkpoint response at turn 50
    turn_100_{condition}.md
    turn_150_{condition}.md
    turn_200_{condition}.md
    closing_{condition}.md                      — closing prompt response
```

The `results/` directory contains no sensitive data and can be committed or shared as-is. API keys are scrubbed from every write. The `summary.json` in each run records the exact model identifiers, parameters, and (for proxy runs) which substrate was used, making the audit trail unambiguous.

### Building a blinded evaluation package

```python
from core.extractor import build_evaluation_package
build_evaluation_package(baseline_run_dir, proxy_run_dir, output_dir)
# Produces evaluation_package.json (A/B blinded) and evaluation_key.json
```

---

## Proxy Run Database Options

When starting a proxy run the sidebar offers two options:

- **New** — start with a clean substrate (no prior context)
- **From existing run** — copy a prior proxy run's substrate (the source is never modified)

This supports two meaningful comparison types:

1. **Two clean runs** — isolates model variance from context quality variation
2. **Mature substrate vs. clean start** — measures the effect of accumulated context on response quality

The `database_source` field in `summary.json` records which option was used.

---

## Running Tests

```bash
cd orchestrator
python3 -m pytest tests/ -v
```

Expected: **69 tests passing.**

---

## Smoke Test (end-to-end, requires running stack)

```bash
# Against LM Studio on the host
python3 tools/smoke_test.py --model-id "your-model-name"

# Against Ollama cloud
python3 tools/smoke_test.py \
  --lm-studio-url https://api.ollama.com \
  --model-id "qwen3.5:cloud"

# Adjust turn count and timeout for slow local models
python3 tools/smoke_test.py --model-id "local-model" --turns 3 --timeout 1200
```

The script creates a temporary model and experiment, runs both conditions, reports pass/fail, and cleans up. Exit code 0 means both conditions completed successfully.

Local inference can be slow — a 3-turn run against a large local model may take 15–20 minutes. This is normal and not a timeout; set `LLM_TIMEOUT_SECONDS` appropriately.

---

## Port Reference

| Port | Service | Exposed |
|------|---------|---------|
| 3000 | UI + API (configurable via `UI_PORT`) | Yes |
| 8000 | Substrate API | Internal only |
| 8080 | MCP Server | Internal only |
| 7474 | Neo4j Browser | Not exposed by default |
| 7687 | Neo4j Bolt | Internal only |

---

## Sharing Results

The `results/` directory is safe to publish — no API keys, no credentials. To keep your model configurations and keys private while sharing results:

```bash
# Add to .gitignore
data/setup/
```

Or to share configurations but not keys:

```bash
data/setup/keys.json
```

---

## Publishing Your Results

Results from this platform are welcome in this repository. If you run the experiment — or any variant of it — and want to share your findings, open a pull request adding your results to the `results/` directory under a clearly named subdirectory:

```
results/{your-experiment-name}/
```

There is no formal review process. The goal is to accumulate a public record of how different models and configurations perform under this kind of extended, structured conversation. Positive results, negative results, and inconclusive results are all equally useful.

### To keep results comparable

A few lightweight conventions make the data useful rather than just numerous:

**Required:**
- `summary.json` — generated automatically by every run; do not edit it. This is the primary record of what was actually run.
- At least one transcript file (`transcript_*.md`) for each condition you are submitting.

**In your PR description, include:**
- Model name and parameter count (e.g. `Qwen3 32B`)
- Quantization if applicable (e.g. `Q4_K_M`, `fp16`, `bf16`)
- Hardware (e.g. `RTX 4090 24GB`, `M3 Max 96GB`, `cloud API`)
- Any parameters you changed from the defaults, and why
- A brief qualitative note on what you observed — even one sentence is enough

**Please don't:**
- Edit transcript files after generation — unmodified transcripts are the point
- Submit partial runs (both conditions must be complete for the comparison to be valid)
- Change the directory structure inside your result folder

Raw JSONL logs and checkpoint artifacts are optional but appreciated for reproducibility. If you ran a variant — different prompts, different turn limit, different interviewer — note that clearly so it can be interpreted in context rather than treated as a straight replication.

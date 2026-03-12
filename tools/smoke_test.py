#!/usr/bin/env python3
"""
Live smoke test — runs a real 3-turn baseline + proxy experiment against a
deployed XP Test Platform stack.

By default points at LM Studio on host.docker.internal:12345, which is how
the orchestrator container reaches the host machine. Run this script from
the host machine (not inside Docker).

Usage:
  python3 tools/smoke_test.py [options]

Options:
  --base-url        Orchestrator API base (default: http://localhost:3000)
  --lm-studio-url   LM Studio endpoint as seen FROM the orchestrator container
                    (default: http://host.docker.internal:12345)
  --model-id        Model identifier string in LM Studio (default: local-model)
  --turns           Turns per run (default: 3)
  --timeout         Max seconds to wait per run (default: 300)
  --no-cleanup      Keep the test model/experiment in the store after the run
  --help            Show this help

Example:
  python3 tools/smoke_test.py --model-id "qwen3-coder-next" --turns 3
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

API = "http://localhost:3000"


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _request(method: str, path: str, body=None, base=None):
    base = base or API
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            if resp.status == 204 or not raw:
                return None
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {body_text}") from e


def get(path):    return _request("GET",    path)
def post(path, body): return _request("POST",   path, body)
def delete(path): return _request("DELETE", path)
def put(path, body):  return _request("PUT",    path, body)
def patch(path, body): return _request("PATCH",  path, body)


# ── Polling ────────────────────────────────────────────────────────────────────

def poll_run(run_id: str, timeout: int, label: str) -> dict:
    """
    Poll GET /api/runs until the run is complete, error, or cancelled.
    Returns the summary dict from GET /api/runs (list scan).
    """
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        runs = get("/api/runs")
        match = next((r for r in runs if r.get("run_id") == run_id), None)
        if match:
            status = match.get("status", "")
            if status == "complete":
                print(f"\r  {label}: complete ({int(time.time() - (deadline - timeout))}s)")
                return match
            if status in ("error", "cancelled"):
                print(f"\r  {label}: {status}")
                return match
            # Still running — show a dot
            print(f"\r  {label}: {status} {'.' * (dots % 4 + 1)}   ", end="", flush=True)
        time.sleep(3)
        dots += 1
    raise TimeoutError(f"{label} did not complete within {timeout}s")


# ── Smoke test ─────────────────────────────────────────────────────────────────

def run_smoke(args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"smoke-{ts}"
    exp_name   = f"smoke-{ts}"
    created_model_id = None
    created_exp_id   = None

    print(f"\n{'='*60}")
    print(f"  XP Framework Test Platform — Live Smoke Test")
    print(f"  API:     {args.base_url}")
    print(f"  Model:   {args.model_id} @ {args.lm_studio_url}")
    print(f"  Turns:   {args.turns}")
    print(f"{'='*60}\n")

    # ── 1. Health check ───────────────────────────────────────────────────────
    print("[ 1/6 ] Health check…", end=" ")
    health = get("/api/health")
    assert health.get("status") == "ok", f"Health check failed: {health}"
    print("ok")

    # ── 2. Create model ───────────────────────────────────────────────────────
    print(f"[ 2/6 ] Creating model store entry '{model_name}'…", end=" ")
    model = post("/api/models", {
        "name":             model_name,
        "model_identifier": args.model_id,
        "endpoint_url":     args.lm_studio_url,
        "context_window":   131072,
    })
    created_model_id = model["id"]
    print(f"ok (id={created_model_id[:8]})")

    # Set API key — LM Studio doesn't require one, but the field must be set
    put(f"/api/models/{created_model_id}/key", {"key": "lm-studio-local"})

    # ── 3. Create experiment ──────────────────────────────────────────────────
    print(f"[ 3/6 ] Creating experiment '{exp_name}'…", end=" ")
    exp = post("/api/experiments", {
        "name":                 exp_name,
        "pass1_model_id":       created_model_id,
        "pass2_model_id":       created_model_id,
        "interviewer_model_id": created_model_id,
        "turn_limit":           args.turns,
        "turn_pause_seconds":   2,
        "context_window":       131072,
        "pass1_activation_fraction": 0.50,
    })
    created_exp_id = exp["id"]
    print(f"ok (id={created_exp_id[:8]})")

    results = {}

    # ── 4. Baseline run ───────────────────────────────────────────────────────
    print(f"[ 4/6 ] Starting baseline run ({args.turns} turns)…")
    run_b = post("/api/runs", {
        "experiment_id": created_exp_id,
        "condition":     "baseline",
        "db_source":     "new",
    })
    baseline_run_id = run_b["run_id"]
    print(f"        run_id: {baseline_run_id}")

    run_b_result = poll_run(baseline_run_id, args.timeout, "  baseline")

    if run_b_result.get("status") != "complete":
        print(f"  ERROR: baseline run ended with status '{run_b_result.get('status')}'")
        results["baseline"] = "FAILED"
    else:
        results["baseline"] = "PASSED"

    # ── 5. Proxy run ──────────────────────────────────────────────────────────
    print(f"[ 5/6 ] Starting proxy run ({args.turns} turns, fresh substrate)…")
    run_p = post("/api/runs", {
        "experiment_id": created_exp_id,
        "condition":     "proxy",
        "db_source":     "new",
    })
    proxy_run_id = run_p["run_id"]
    print(f"        run_id: {proxy_run_id}")

    run_p_result = poll_run(proxy_run_id, args.timeout, "  proxy")

    if run_p_result.get("status") != "complete":
        print(f"  ERROR: proxy run ended with status '{run_p_result.get('status')}'")
        results["proxy"] = "FAILED"
    else:
        results["proxy"] = "PASSED"

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print(f"\n[ 6/6 ] Results")
    print(f"  baseline : {results.get('baseline', '?')}")
    print(f"  proxy    : {results.get('proxy',    '?')}")

    if "FAILED" not in results.values():
        print(f"\n  All runs completed successfully.")
        print(f"  Output files are in: testplatform/results/")
        print(f"  Open http://localhost:{args.base_url.split(':')[-1]} to review in the UI.")
    else:
        print(f"\n  One or more runs failed — check the orchestrator logs for details.")
        print(f"  docker compose logs orchestrator")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not args.no_cleanup:
        print(f"\n  Cleaning up smoke test model and experiment from store…", end=" ")
        try:
            delete(f"/api/experiments/{created_exp_id}")
            delete(f"/api/models/{created_model_id}")
            print("done")
        except Exception as e:
            print(f"(warning: cleanup failed: {e})")
    else:
        print(f"\n  --no-cleanup: model '{model_name}' and experiment '{exp_name}' left in store.")

    print()
    failed = [k for k, v in results.items() if v != "PASSED"]
    sys.exit(1 if failed else 0)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live smoke test for the XP Test Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[0].strip(),
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:3000",
        help="Orchestrator API base URL (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--lm-studio-url",
        default="http://host.docker.internal:12345",
        help="LM Studio endpoint as seen from the orchestrator container "
             "(default: http://host.docker.internal:12345)",
    )
    parser.add_argument(
        "--model-id",
        default="local-model",
        help="Model identifier string in LM Studio (default: local-model). "
             "Must match the name shown in LM Studio's loaded model.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Number of turns per run (default: 3). Keep small for smoke testing.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Max seconds to wait for each run to complete (default: 300). "
             "Increase if the model is slow to respond.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep the smoke test model/experiment in the store after the run.",
    )

    args = parser.parse_args()
    global API
    API = args.base_url.rstrip("/")

    run_smoke(args)


if __name__ == "__main__":
    main()

"""
Results extractor — post-run artifact extraction.

Reads the raw JSONL log for a completed run and writes:
  - artifacts/turn_{N}_{condition}.md  for each checkpoint turn
  - artifacts/closing_{condition}.md

Also provides build_evaluation_package() to produce a blinded A/B evaluation
bundle from a paired baseline + proxy run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def extract_artifacts(
    raw_path: Path,
    condition: str,
    run_dir: Path,
    checkpoint_turns: list[int],
) -> dict[str, str]:
    """
    Read raw JSONL and write checkpoint + closing artifacts.

    Returns a dict mapping artifact keys to paths relative to run_dir:
      {"turn_50": "artifacts/turn_50_baseline.md", "closing": "artifacts/closing_baseline.md", ...}
    """
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    artifact_paths: dict[str, str] = {}

    if not raw_path.exists():
        return artifact_paths

    with raw_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("event") == "checkpoint":
                turn = event.get("turn")
                response = event.get("response", "")
                filename = f"turn_{turn}_{condition}.md"
                path = artifacts_dir / filename
                path.write_text(
                    f"# Turn {turn} — {condition.capitalize()}\n\n{response}\n"
                )
                artifact_paths[f"turn_{turn}"] = str(path.relative_to(run_dir))

            elif event.get("event") == "closing":
                response = event.get("response", "")
                filename = f"closing_{condition}.md"
                path = artifacts_dir / filename
                path.write_text(
                    f"# Closing — {condition.capitalize()}\n\n{response}\n"
                )
                artifact_paths["closing"] = str(path.relative_to(run_dir))

    return artifact_paths


def build_evaluation_package(
    baseline_run_dir: Path,
    proxy_run_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    """
    Build a blinded A/B evaluation package from a paired baseline + proxy run.

    Randomly assigns A/B labels so evaluators cannot identify the condition.
    Writes two files to output_dir:
      evaluation_package.json — artifacts labeled A and B (no condition names)
      evaluation_key.json     — which label maps to which condition

    Returns {"package_path": str, "key_path": str}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_artifacts = _load_artifacts(baseline_run_dir)
    proxy_artifacts = _load_artifacts(proxy_run_dir)

    # Randomly assign A/B
    if random.random() < 0.5:
        label_a, label_b = "baseline", "proxy"
        artifacts_a, artifacts_b = baseline_artifacts, proxy_artifacts
    else:
        label_a, label_b = "proxy", "baseline"
        artifacts_a, artifacts_b = proxy_artifacts, baseline_artifacts

    package = {"A": artifacts_a, "B": artifacts_b}
    key = {"A": label_a, "B": label_b}

    package_path = output_dir / "evaluation_package.json"
    key_path = output_dir / "evaluation_key.json"

    package_path.write_text(json.dumps(package, indent=2))
    key_path.write_text(json.dumps(key, indent=2))

    return {
        "package_path": str(package_path),
        "key_path": str(key_path),
    }


def _load_artifacts(run_dir: Path) -> dict[str, str]:
    """Load all artifact .md files from a run's artifacts/ directory."""
    artifacts: dict[str, str] = {}
    artifact_dir = run_dir / "artifacts"
    if not artifact_dir.exists():
        return artifacts
    for path in sorted(artifact_dir.glob("*.md")):
        artifacts[path.stem] = path.read_text()
    return artifacts

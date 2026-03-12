"""
Experiment store — CRUD for experiment configurations.

An ExperimentConfig defines a named test: which models run each pass,
which prompts to use, which MCP servers are attached, and any overrides
to global defaults. All overrides are recorded in run outputs for audit traceability.
"""

from __future__ import annotations

from pydantic import BaseModel

from .base import setup_dir, read_json, write_json, new_id, now_iso


def _experiments_dir():
    d = setup_dir() / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ExperimentConfig(BaseModel):
    id: str
    name: str

    # Model ids (references into the model store)
    pass1_model_id: str
    pass2_model_id: str
    interviewer_model_id: str

    # Prompt ids (None = use the global default file from config/prompts/)
    opening_question_prompt_id: str | None = None
    test_model_system_prompt_id: str | None = None
    interviewer_system_prompt_id: str | None = None
    closing_prompt_id: str | None = None
    compaction_prompt_id: str | None = None

    # MCP server ids attached to this test (available to Pass 2 and baseline)
    mcp_server_ids: list[str] = []

    # Parameter overrides (None = use global default from config.yaml)
    turn_limit: int | None = None
    context_window: int | None = None
    compaction_threshold_fraction: float | None = None
    pass1_activation_fraction: float | None = None
    turn_pause_seconds: int | None = None
    search_enabled: bool | None = None

    created_at: str
    updated_at: str


def list_experiments() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(**read_json(p))
        for p in sorted(_experiments_dir().glob("*.json"))
    ]


def get_experiment(experiment_id: str) -> ExperimentConfig | None:
    path = _experiments_dir() / f"{experiment_id}.json"
    if not path.exists():
        return None
    return ExperimentConfig(**read_json(path))


def create_experiment(
    name: str,
    pass1_model_id: str,
    pass2_model_id: str,
    interviewer_model_id: str,
    **kwargs,
) -> ExperimentConfig:
    entry = ExperimentConfig(
        id=new_id(),
        name=name,
        pass1_model_id=pass1_model_id,
        pass2_model_id=pass2_model_id,
        interviewer_model_id=interviewer_model_id,
        created_at=now_iso(),
        updated_at=now_iso(),
        **kwargs,
    )
    write_json(_experiments_dir() / f"{entry.id}.json", entry.model_dump())
    return entry


def update_experiment(experiment_id: str, **kwargs) -> ExperimentConfig | None:
    entry = get_experiment(experiment_id)
    if entry is None:
        return None
    data = entry.model_dump()
    data.update(kwargs)
    data["updated_at"] = now_iso()
    updated = ExperimentConfig(**data)
    write_json(_experiments_dir() / f"{experiment_id}.json", updated.model_dump())
    return updated


def delete_experiment(experiment_id: str) -> bool:
    path = _experiments_dir() / f"{experiment_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True

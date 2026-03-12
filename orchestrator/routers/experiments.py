from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from store.experiments import (
    ExperimentConfig, list_experiments, get_experiment,
    create_experiment, update_experiment, delete_experiment,
)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class CreateExperimentRequest(BaseModel):
    name: str
    pass1_model_id: str
    pass2_model_id: str
    interviewer_model_id: str
    opening_question_prompt_id: str | None = None
    test_model_system_prompt_id: str | None = None
    interviewer_system_prompt_id: str | None = None
    closing_prompt_id: str | None = None
    compaction_prompt_id: str | None = None
    mcp_server_ids: list[str] = []
    turn_limit: int | None = None
    context_window: int | None = None
    compaction_threshold_fraction: float | None = None
    pass1_activation_fraction: float | None = None
    turn_pause_seconds: int | None = None
    search_enabled: bool | None = None


class UpdateExperimentRequest(BaseModel):
    name: str | None = None
    pass1_model_id: str | None = None
    pass2_model_id: str | None = None
    interviewer_model_id: str | None = None
    opening_question_prompt_id: str | None = None
    test_model_system_prompt_id: str | None = None
    interviewer_system_prompt_id: str | None = None
    closing_prompt_id: str | None = None
    compaction_prompt_id: str | None = None
    mcp_server_ids: list[str] | None = None
    turn_limit: int | None = None
    context_window: int | None = None
    compaction_threshold_fraction: float | None = None
    pass1_activation_fraction: float | None = None
    turn_pause_seconds: int | None = None
    search_enabled: bool | None = None


@router.get("", response_model=list[ExperimentConfig])
def list_experiments_route():
    return list_experiments()


@router.post("", response_model=ExperimentConfig, status_code=status.HTTP_201_CREATED)
def create_experiment_route(body: CreateExperimentRequest):
    return create_experiment(**body.model_dump())


@router.get("/{experiment_id}", response_model=ExperimentConfig)
def get_experiment_route(experiment_id: str):
    exp = get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.patch("/{experiment_id}", response_model=ExperimentConfig)
def update_experiment_route(experiment_id: str, body: UpdateExperimentRequest):
    kwargs = {k: v for k, v in body.model_dump().items() if v is not None}
    exp = update_experiment(experiment_id, **kwargs)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment_route(experiment_id: str):
    if not delete_experiment(experiment_id):
        raise HTTPException(status_code=404, detail="Experiment not found")

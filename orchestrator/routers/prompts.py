import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from store.prompts import (
    PromptEntry, list_prompts, get_prompt, create_prompt, update_prompt, delete_prompt,
)

router = APIRouter(prefix="/api/prompts", tags=["prompts"])

_PROMPT_LABELS = {
    "opening_question":   "Opening Question",
    "test_model_system":  "Test Model System Prompt",
    "interviewer_system": "Interviewer System Prompt",
    "pass1_system":       "Pass 1 System Prompt",
    "compaction_prompt":  "Compaction Prompt",
    "closing_prompt":     "Closing Prompt",
}


class CreatePromptRequest(BaseModel):
    name: str
    content: str


class UpdatePromptRequest(BaseModel):
    name: str | None = None
    content: str | None = None


@router.get("/defaults")
def list_default_prompts():
    """Return the built-in default prompts from config/prompts/ as read-only entries."""
    config_dir = Path(os.environ.get("CONFIG_DIR", "/app/config")) / "prompts"
    defaults = []
    for key, label in _PROMPT_LABELS.items():
        path = config_dir / f"{key}.txt"
        if path.exists():
            defaults.append({"key": key, "name": label, "content": path.read_text()})
    return defaults


@router.get("", response_model=list[PromptEntry])
def list_prompts_route():
    return list_prompts()


@router.post("", response_model=PromptEntry, status_code=status.HTTP_201_CREATED)
def create_prompt_route(body: CreatePromptRequest):
    return create_prompt(name=body.name, content=body.content)


@router.get("/{prompt_id}", response_model=PromptEntry)
def get_prompt_route(prompt_id: str):
    entry = get_prompt(prompt_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return entry


@router.patch("/{prompt_id}", response_model=PromptEntry)
def update_prompt_route(prompt_id: str, body: UpdatePromptRequest):
    kwargs = {k: v for k, v in body.model_dump().items() if v is not None}
    entry = update_prompt(prompt_id, **kwargs)
    if entry is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return entry


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prompt_route(prompt_id: str):
    if not delete_prompt(prompt_id):
        raise HTTPException(status_code=404, detail="Prompt not found")

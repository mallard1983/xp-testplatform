"""
Model store — CRUD for model definitions.

A ModelEntry describes how to reach a model API. The actual API key is stored
separately in keys.json and referenced only by model id. Key values never
appear in ModelEntry files.
"""

from __future__ import annotations

from pydantic import BaseModel

from .base import setup_dir, read_json, write_json, new_id, now_iso


def _models_dir():
    d = setup_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ModelEntry(BaseModel):
    id: str
    name: str                   # Human-readable label shown in the UI
    model_identifier: str       # String passed to the API as the model name
    endpoint_url: str           # Base URL of the API (e.g. https://api.ollama.com)
    context_window: int         # Max tokens this model supports
    created_at: str
    updated_at: str


def list_models() -> list[ModelEntry]:
    return [
        ModelEntry(**read_json(p))
        for p in sorted(_models_dir().glob("*.json"))
    ]


def get_model(model_id: str) -> ModelEntry | None:
    path = _models_dir() / f"{model_id}.json"
    if not path.exists():
        return None
    return ModelEntry(**read_json(path))


def create_model(
    name: str,
    model_identifier: str,
    endpoint_url: str,
    context_window: int,
) -> ModelEntry:
    entry = ModelEntry(
        id=new_id(),
        name=name,
        model_identifier=model_identifier,
        endpoint_url=endpoint_url,
        context_window=context_window,
        created_at=now_iso(),
        updated_at=now_iso(),
    )
    write_json(_models_dir() / f"{entry.id}.json", entry.model_dump())
    return entry


def update_model(model_id: str, **kwargs) -> ModelEntry | None:
    entry = get_model(model_id)
    if entry is None:
        return None
    data = entry.model_dump()
    data.update(kwargs)
    data["updated_at"] = now_iso()
    updated = ModelEntry(**data)
    write_json(_models_dir() / f"{model_id}.json", updated.model_dump())
    return updated


def delete_model(model_id: str) -> bool:
    path = _models_dir() / f"{model_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True

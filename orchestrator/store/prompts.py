"""
Prompt store — CRUD for named prompts.

Prompts are versioned by name — you can create multiple prompts with different
names and select between them per-test. The content field holds the full prompt text.
"""

from __future__ import annotations

from pydantic import BaseModel

from .base import setup_dir, read_json, write_json, new_id, now_iso


def _prompts_dir():
    d = setup_dir() / "prompts"
    d.mkdir(parents=True, exist_ok=True)
    return d


class PromptEntry(BaseModel):
    id: str
    name: str       # Human-readable label, shown in dropdowns
    content: str    # Full prompt text
    created_at: str
    updated_at: str


def list_prompts() -> list[PromptEntry]:
    return [
        PromptEntry(**read_json(p))
        for p in sorted(_prompts_dir().glob("*.json"))
    ]


def get_prompt(prompt_id: str) -> PromptEntry | None:
    path = _prompts_dir() / f"{prompt_id}.json"
    if not path.exists():
        return None
    return PromptEntry(**read_json(path))


def create_prompt(name: str, content: str) -> PromptEntry:
    entry = PromptEntry(
        id=new_id(),
        name=name,
        content=content,
        created_at=now_iso(),
        updated_at=now_iso(),
    )
    write_json(_prompts_dir() / f"{entry.id}.json", entry.model_dump())
    return entry


def update_prompt(prompt_id: str, **kwargs) -> PromptEntry | None:
    entry = get_prompt(prompt_id)
    if entry is None:
        return None
    data = entry.model_dump()
    data.update(kwargs)
    data["updated_at"] = now_iso()
    updated = PromptEntry(**data)
    write_json(_prompts_dir() / f"{prompt_id}.json", updated.model_dump())
    return updated


def delete_prompt(prompt_id: str) -> bool:
    path = _prompts_dir() / f"{prompt_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True

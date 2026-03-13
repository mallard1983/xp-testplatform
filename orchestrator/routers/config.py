"""
Config router — global experiment defaults and search/key configuration.

Key values are NEVER returned in any response. Search config responses
include a `key_configured` boolean only.
"""

import os
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from store.keys import get_search_config, set_search_config

router = APIRouter(prefix="/api/config", tags=["config"])

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/app/config/config.yaml"))


def _read_config() -> dict:
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=503, detail="config.yaml not found")
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def _write_config(data: dict) -> None:
    with CONFIG_PATH.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


class GlobalsUpdateRequest(BaseModel):
    turn_limit: int | None = None
    context_window: int | None = None
    compaction_threshold_fraction: float | None = None
    pass1_activation_fraction: float | None = None
    turn_pause_min_seconds: float | None = None
    turn_pause_max_seconds: float | None = None


class SearchConfigRequest(BaseModel):
    provider: str
    api_key: str | None = None   # Optional — omit to leave existing key unchanged
    enabled: bool


@router.get("/globals")
def get_globals():
    cfg = _read_config()
    return cfg.get("experiment", {})


@router.patch("/globals")
def update_globals(body: GlobalsUpdateRequest):
    cfg = _read_config()
    exp = cfg.get("experiment", {})
    for field, value in body.model_dump().items():
        if value is not None:
            exp[field] = value
    cfg["experiment"] = exp
    _write_config(cfg)
    return exp


@router.get("/search")
def get_search():
    cfg = get_search_config()
    return {
        "provider": cfg.get("provider", "brave"),
        "enabled": cfg.get("enabled", True),
        "key_configured": bool(cfg.get("api_key")),
        # api_key is intentionally excluded
    }


@router.put("/search")
def update_search(body: SearchConfigRequest):
    existing = get_search_config()
    key = body.api_key if body.api_key is not None else existing.get("api_key", "")
    set_search_config(provider=body.provider, api_key=key, enabled=body.enabled)
    return {
        "provider": body.provider,
        "enabled": body.enabled,
        "key_configured": bool(key),
    }

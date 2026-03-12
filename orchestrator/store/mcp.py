"""
MCP server store — CRUD for MCP server definitions.

API key for the MCP server (if required) is stored in keys.json under
the server's id, not in the McpServerEntry file itself.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .base import setup_dir, read_json, write_json, new_id, now_iso


def _mcp_dir():
    d = setup_dir() / "mcp"
    d.mkdir(parents=True, exist_ok=True)
    return d


class McpServerEntry(BaseModel):
    id: str
    name: str                           # Human-readable label
    endpoint_url: str                   # SSE endpoint URL
    auth_type: Literal["none", "bearer", "api_key"] = "none"
    # Actual key stored in keys.json under this server's id — not here
    created_at: str
    updated_at: str


def list_mcp_servers() -> list[McpServerEntry]:
    return [
        McpServerEntry(**read_json(p))
        for p in sorted(_mcp_dir().glob("*.json"))
    ]


def get_mcp_server(server_id: str) -> McpServerEntry | None:
    path = _mcp_dir() / f"{server_id}.json"
    if not path.exists():
        return None
    return McpServerEntry(**read_json(path))


def create_mcp_server(
    name: str,
    endpoint_url: str,
    auth_type: str = "none",
) -> McpServerEntry:
    entry = McpServerEntry(
        id=new_id(),
        name=name,
        endpoint_url=endpoint_url,
        auth_type=auth_type,
        created_at=now_iso(),
        updated_at=now_iso(),
    )
    write_json(_mcp_dir() / f"{entry.id}.json", entry.model_dump())
    return entry


def update_mcp_server(server_id: str, **kwargs) -> McpServerEntry | None:
    entry = get_mcp_server(server_id)
    if entry is None:
        return None
    data = entry.model_dump()
    data.update(kwargs)
    data["updated_at"] = now_iso()
    updated = McpServerEntry(**data)
    write_json(_mcp_dir() / f"{server_id}.json", updated.model_dump())
    return updated


def delete_mcp_server(server_id: str) -> bool:
    path = _mcp_dir() / f"{server_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True

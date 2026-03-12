from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Literal

from store.mcp import (
    McpServerEntry, list_mcp_servers, get_mcp_server,
    create_mcp_server, update_mcp_server, delete_mcp_server,
)
from store.keys import get_model_key, set_model_key, delete_model_key

router = APIRouter(prefix="/api/mcp-servers", tags=["mcp-servers"])


class CreateMcpServerRequest(BaseModel):
    name: str
    endpoint_url: str
    auth_type: Literal["none", "bearer", "api_key"] = "none"
    api_key: str | None = None  # Written to keys.json if provided


class UpdateMcpServerRequest(BaseModel):
    name: str | None = None
    endpoint_url: str | None = None
    auth_type: Literal["none", "bearer", "api_key"] | None = None


class SetKeyRequest(BaseModel):
    key: str


@router.get("", response_model=list[McpServerEntry])
def list_mcp_servers_route():
    return list_mcp_servers()


@router.post("", response_model=McpServerEntry, status_code=status.HTTP_201_CREATED)
def create_mcp_server_route(body: CreateMcpServerRequest):
    entry = create_mcp_server(
        name=body.name,
        endpoint_url=body.endpoint_url,
        auth_type=body.auth_type,
    )
    if body.api_key:
        set_model_key(entry.id, body.api_key)
    return entry


@router.get("/{server_id}", response_model=McpServerEntry)
def get_mcp_server_route(server_id: str):
    entry = get_mcp_server(server_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return entry


@router.patch("/{server_id}", response_model=McpServerEntry)
def update_mcp_server_route(server_id: str, body: UpdateMcpServerRequest):
    kwargs = {k: v for k, v in body.model_dump().items() if v is not None}
    entry = update_mcp_server(server_id, **kwargs)
    if entry is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return entry


@router.delete("/{server_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_mcp_server_route(server_id: str):
    if not delete_mcp_server(server_id):
        raise HTTPException(status_code=404, detail="MCP server not found")
    delete_model_key(server_id)


@router.put("/{server_id}/key", status_code=status.HTTP_204_NO_CONTENT)
def set_mcp_server_key_route(server_id: str, body: SetKeyRequest):
    if get_mcp_server(server_id) is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    set_model_key(server_id, body.key)


@router.get("/{server_id}/key/status")
def get_mcp_server_key_status(server_id: str):
    if get_mcp_server(server_id) is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return {"has_key": get_model_key(server_id) is not None}

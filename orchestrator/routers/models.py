from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from store.models import (
    ModelEntry, list_models, get_model, create_model, update_model, delete_model,
)
from store.keys import get_model_key, set_model_key, delete_model_key

router = APIRouter(prefix="/api/models", tags=["models"])


class CreateModelRequest(BaseModel):
    name: str
    model_identifier: str
    endpoint_url: str
    context_window: int
    api_key: str | None = None  # Written to keys.json if provided; never stored in model file


class UpdateModelRequest(BaseModel):
    name: str | None = None
    model_identifier: str | None = None
    endpoint_url: str | None = None
    context_window: int | None = None


class SetKeyRequest(BaseModel):
    key: str


@router.get("", response_model=list[ModelEntry])
def list_models_route():
    return list_models()


@router.post("", response_model=ModelEntry, status_code=status.HTTP_201_CREATED)
def create_model_route(body: CreateModelRequest):
    entry = create_model(
        name=body.name,
        model_identifier=body.model_identifier,
        endpoint_url=body.endpoint_url,
        context_window=body.context_window,
    )
    if body.api_key:
        set_model_key(entry.id, body.api_key)
    return entry


@router.get("/{model_id}", response_model=ModelEntry)
def get_model_route(model_id: str):
    entry = get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return entry


@router.patch("/{model_id}", response_model=ModelEntry)
def update_model_route(model_id: str, body: UpdateModelRequest):
    kwargs = {k: v for k, v in body.model_dump().items() if v is not None}
    entry = update_model(model_id, **kwargs)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return entry


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model_route(model_id: str):
    if not delete_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    delete_model_key(model_id)


@router.put("/{model_id}/key", status_code=status.HTTP_204_NO_CONTENT)
def set_model_key_route(model_id: str, body: SetKeyRequest):
    if get_model(model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    set_model_key(model_id, body.key)


@router.get("/{model_id}/key/status")
def get_model_key_status(model_id: str):
    if get_model(model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"has_key": get_model_key(model_id) is not None}

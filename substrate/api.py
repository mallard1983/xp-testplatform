"""
Substrate API — FastAPI application.
Route handlers are thin wrappers. All logic lives in db.py and embeddings.py.
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ConstraintError
from pydantic import BaseModel, Field

import db
import embeddings


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
    auth_raw = os.environ.get("NEO4J_AUTH", "neo4j/changeme")
    user, password = auth_raw.split("/", 1)
    app.state.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    yield
    await app.state.driver.close()


app = FastAPI(title="XP Framework Substrate API", lifespan=lifespan)


# ── Response envelope ─────────────────────────────────────────────────────────

def ok(data: Any) -> dict:
    return {"data": data, "error": None}


def err(message: str) -> dict:
    return {"data": None, "error": message}


# ── Request models ─────────────────────────────────────────────────────────────

class CreateStreamRequest(BaseModel):
    content: str
    summary: str
    source: str
    source_id: str = ""
    completion_state: str = Field(pattern="^(complete|partial|dead_end)$")
    confidence: float = Field(ge=0.0, le=1.0)
    topic: str | None = None


class CreateTopicRequest(BaseModel):
    name: str
    description: str = ""


# ── Routes — Health ────────────────────────────────────────────────────────────

@app.get("/health", status_code=status.HTTP_200_OK)
async def health(request: Request):
    healthy = await db.check_health(request.app.state.driver)
    if not healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection failed",
        )
    return ok({"status": "ok", "neo4j": "connected"})


# ── Routes — Streams ───────────────────────────────────────────────────────────

@app.post("/streams", status_code=status.HTTP_201_CREATED)
async def create_stream(body: CreateStreamRequest, request: Request):
    embedding = await embeddings.generate_embedding(body.content)
    data = body.model_dump()
    data["embedding"] = embedding
    stream = await db.create_thought_stream(request.app.state.driver, data)
    return ok(stream)


@app.get("/streams/search", status_code=status.HTTP_200_OK)
async def search_streams(
    request: Request,
    q: str,
    limit: int = 5,
    topic: str | None = None,
):
    embedding = await embeddings.generate_embedding(q)
    results = await db.search_streams(request.app.state.driver, embedding, limit, topic)
    return ok(results)


@app.get("/streams/recent", status_code=status.HTTP_200_OK)
async def get_recent(
    request: Request,
    n: int = 10,
    source: str | None = None,
):
    results = await db.get_recent_streams(request.app.state.driver, n, source)
    return ok(results)


@app.get("/streams/{stream_id}", status_code=status.HTTP_200_OK)
async def get_stream(stream_id: str, request: Request):
    stream = await db.get_thought_stream(request.app.state.driver, stream_id)
    if stream is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stream not found")
    return ok(stream)


# ── Routes — Topics ────────────────────────────────────────────────────────────

@app.post("/topics", status_code=status.HTTP_201_CREATED)
async def create_topic(body: CreateTopicRequest, request: Request):
    try:
        topic = await db.create_topic(request.app.state.driver, body.name, body.description)
    except ConstraintError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Topic '{body.name}' already exists",
        )
    return ok(topic)


@app.get("/topics", status_code=status.HTTP_200_OK)
async def list_topics(request: Request):
    topics = await db.get_topics(request.app.state.driver)
    return ok(topics)


# ── Routes — Admin ────────────────────────────────────────────────────────────

@app.post("/admin/reset", status_code=status.HTTP_200_OK)
async def admin_reset(request: Request):
    """
    Delete all nodes and relationships. Schema is preserved.
    Called by the orchestrator's neo4j_manager between test runs to produce
    a clean substrate for each new test. Not exposed outside the internal network.
    """
    deleted = await db.reset_database(request.app.state.driver)
    return ok({"deleted_nodes": deleted, "status": "reset complete"})

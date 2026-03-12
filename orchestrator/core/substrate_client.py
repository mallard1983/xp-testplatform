"""
HTTP client for the substrate API.

Tool dispatch methods return JSON-formatted strings suitable for injection
into tool result messages. The create_stream method is used for async
write-back after each proxy condition turn.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class SubstrateClient:
    def __init__(self, base_url: str = "http://substrate-api:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── Tool dispatch methods (return strings for tool result injection) ───────

    async def search_streams(self, query: str, limit: int = 5, topic: str | None = None) -> str:
        params: dict[str, Any] = {"q": query, "limit": limit}
        if topic:
            params["topic"] = topic
        data = await self._get("/streams/search", params=params)
        results = data.get("data", [])
        if not results:
            return json.dumps([])
        # Return only the fields Pass 1 needs (exclude content — use get_stream for that)
        slim = [
            {
                "id": r["id"],
                "summary": r["summary"],
                "confidence": r["confidence"],
                "relevance_score": r.get("relevance_score", 0.0),
                "topics": r.get("topics", []),
            }
            for r in results
        ]
        return json.dumps(slim, indent=2)

    async def get_stream(self, stream_id: str) -> str:
        try:
            data = await self._get(f"/streams/{stream_id}")
            return json.dumps(data.get("data", {}), indent=2)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return json.dumps({"error": f"Stream {stream_id} not found"})
            raise

    async def list_topics(self) -> str:
        data = await self._get("/topics")
        topics = data.get("data", [])
        return json.dumps(topics, indent=2)

    async def get_recent(self, n: int = 10, source: str | None = None) -> str:
        params: dict[str, Any] = {"n": n}
        if source:
            params["source"] = source
        data = await self._get("/streams/recent", params=params)
        results = data.get("data", [])
        slim = [
            {
                "id": r["id"],
                "summary": r["summary"],
                "confidence": r["confidence"],
                "relevance_score": 0.0,
                "topics": r.get("topics", []),
            }
            for r in results
        ]
        return json.dumps(slim, indent=2)

    # ── Write-back ─────────────────────────────────────────────────────────────

    async def create_stream(
        self,
        content: str,
        summary: str,
        source: str = "conversation",
        source_id: str = "",
        completion_state: str = "complete",
        confidence: float = 0.7,
        topic: str | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "content": content,
            "summary": summary,
            "source": source,
            "source_id": source_id,
            "completion_state": completion_state,
            "confidence": confidence,
        }
        if topic:
            payload["topic"] = topic
        return await self._post("/streams", payload)

    # ── Admin ─────────────────────────────────────────────────────────────────

    async def reset(self) -> dict:
        return await self._post("/admin/reset", {})

    async def health(self) -> bool:
        try:
            data = await self._get("/health")
            return data.get("data", {}).get("status") == "ok"
        except Exception:
            return False

    # ── Tool dispatch dict ─────────────────────────────────────────────────────

    def tool_dispatch(self) -> dict:
        """Return a tool dispatch dict for use with OllamaClient.run_with_tools()."""
        return {
            "search_streams": self.search_streams,
            "get_stream": self.get_stream,
            "list_topics": self.list_topics,
            "get_recent": self.get_recent,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

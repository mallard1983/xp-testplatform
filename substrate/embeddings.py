"""
Embedding generation via local Ollama container (ollama-embed service).
Caller provides text; returns a 768-dim float list.

Uses the OpenAI-compatible /v1/embeddings endpoint that Ollama exposes,
so the payload format is identical to the LM Studio version in ../substrate/.
"""

import os

import httpx


OLLAMA_BASE_URL = os.environ.get("OLLAMA_EMBED_URL", "http://ollama-embed:11434")
EMBEDDING_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


async def generate_embedding(text: str) -> list[float]:
    """Call Ollama embeddings endpoint and return the embedding vector."""
    url = f"{OLLAMA_BASE_URL}/v1/embeddings"
    payload = {"model": EMBEDDING_MODEL, "input": text}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)

    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]

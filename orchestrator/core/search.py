"""
Pluggable search integration.

The search provider and API key are read from keys.json at call time.
Currently supported: Brave Search API.
Adding a new provider: implement a function matching the signature
  async def _search_<provider>(query, api_key) -> str
and register it in PROVIDERS.

The return value is a formatted string injected into the model's tool result.
"""

from __future__ import annotations

import httpx

from store.keys import get_search_config


# ── Brave ─────────────────────────────────────────────────────────────────────

async def _search_brave(query: str, api_key: str) -> str:
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": 5}

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("web", {}).get("results", [])
    if not results:
        return "No results found."

    lines = []
    for r in results[:5]:
        title = r.get("title", "")
        url_ = r.get("url", "")
        desc = r.get("description", "")
        lines.append(f"**{title}**\n{url_}\n{desc}")

    return "\n\n".join(lines)


# ── Provider registry ─────────────────────────────────────────────────────────

PROVIDERS: dict[str, callable] = {
    "brave": _search_brave,
}


# ── Public interface ──────────────────────────────────────────────────────────

async def web_search(query: str) -> str:
    """
    Execute a web search using the configured provider.
    Returns a formatted string of results for injection into tool results.
    Raises RuntimeError if search is disabled or provider is unknown.
    """
    cfg = get_search_config()

    if not cfg.get("enabled", True):
        return "Search is disabled for this experiment."

    provider = cfg.get("provider", "brave")
    api_key = cfg.get("api_key", "")

    if not api_key:
        return "Search is configured but no API key is set."

    handler = PROVIDERS.get(provider)
    if handler is None:
        return f"Unknown search provider: '{provider}'. Supported: {list(PROVIDERS.keys())}"

    return await handler(query=query, api_key=api_key)

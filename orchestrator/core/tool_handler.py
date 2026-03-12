"""
Tool definitions and scoping for Pass 1, Pass 2, and baseline.

Tool scoping is enforced at the API call level — each pass receives only
the tool list appropriate for its role:

  Pass 1:   SUBSTRATE_TOOLS only (substrate MCP tools — no search, no external MCP)
  Pass 2:   SEARCH_TOOLS + any external MCP tools configured for the test
  Baseline: SEARCH_TOOLS + any external MCP tools configured for the test
"""

from __future__ import annotations


# ── Substrate tool definitions (Pass 1 only) ──────────────────────────────────

SUBSTRATE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_streams",
            "description": (
                "Search the experience substrate semantically for relevant ThoughtStreams. "
                "Use a concise phrase capturing the concept you are looking for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 5, max 20)",
                        "default": 5,
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional topic filter — must exactly match a topic name from list_topics",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stream",
            "description": (
                "Retrieve the full content of a specific ThoughtStream by ID. "
                "Use when a search result summary is promising but ambiguous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stream_id": {
                        "type": "string",
                        "description": "ThoughtStream UUID from a search result",
                    },
                },
                "required": ["stream_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_topics",
            "description": (
                "List all topic domains currently in the substrate. "
                "Use when searches return nothing and you need to recalibrate."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent",
            "description": (
                "Retrieve the most recently added ThoughtStreams regardless of topic. "
                "Useful for continuity when the question implies ongoing reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of streams to return (default 10)",
                        "default": 10,
                    },
                    "source": {
                        "type": "string",
                        "description": "Filter by source: 'conversation'",
                    },
                },
            },
        },
    },
]

# ── Search tool definition (Pass 2 + baseline) ────────────────────────────────

SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web to verify an empirical claim or check whether an idea "
            "exists in the literature before continuing your reasoning."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["query"],
        },
    },
}


# ── Tool list builders ─────────────────────────────────────────────────────────

def pass1_tools() -> list[dict]:
    """Tool list for Pass 1 — substrate tools only."""
    return list(SUBSTRATE_TOOLS)


def pass2_tools(extra_mcp_tools: list[dict] | None = None) -> list[dict]:
    """Tool list for Pass 2 — search + any configured external MCP tools."""
    tools = [SEARCH_TOOL]
    if extra_mcp_tools:
        tools.extend(extra_mcp_tools)
    return tools


def baseline_tools(extra_mcp_tools: list[dict] | None = None) -> list[dict]:
    """Tool list for baseline condition — same as Pass 2."""
    return pass2_tools(extra_mcp_tools)

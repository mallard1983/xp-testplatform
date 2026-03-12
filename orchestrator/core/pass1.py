"""
Pass 1 — Context retrieval.

Receives the full conversation history, queries the substrate via MCP-style
tool calls, and returns the CONTEXT_START...CONTEXT_END block for injection
into Pass 2.

Pass 1 is only activated above the token threshold (see proxy_condition.py).
Below the threshold the full history is passed straight to Pass 2.

Tool budget is enforced via max_iterations in run_with_tools().
Pass 1 never receives search tools — substrate tools only.
"""

from __future__ import annotations

from core.tool_handler import pass1_tools


_CONTEXT_START = "CONTEXT_START"
_CONTEXT_END = "CONTEXT_END"


async def run_pass1(
    messages: list[dict],
    client,              # OllamaClient
    substrate_client,    # SubstrateClient
    pass1_system_prompt: str,
    tool_budget: int = 8,
) -> tuple[str, dict]:
    """
    Run Pass 1 context retrieval.

    Strips any existing system message from the input and prepends the
    Pass 1 system prompt. Calls the model with substrate tools only.

    Returns a tuple of (context_block_str, usage_dict). Returns an empty
    block string and zero usage on failure.
    """
    # Build Pass 1 message array: Pass 1 system prompt + conversation history
    conversation = [m for m in messages if m.get("role") != "system"]
    pass1_messages = [
        {"role": "system", "content": pass1_system_prompt},
        *conversation,
    ]

    tools = pass1_tools()
    dispatch = substrate_client.tool_dispatch()

    try:
        text, _, usage = await client.run_with_tools(
            messages=pass1_messages,
            tools=tools,
            tool_dispatch=dispatch,
            max_iterations=tool_budget + 2,  # budget + buffer for final response
        )
        return text or _empty_block("Pass 1 returned no output."), usage
    except Exception as exc:
        return _empty_block(f"Pass 1 error: {exc}"), {"prompt_tokens": 0, "completion_tokens": 0}


def extract_context_block(pass1_output: str) -> str:
    """
    Extract the content between CONTEXT_START and CONTEXT_END.
    Returns an empty string if the block is empty or malformed.
    """
    if _CONTEXT_START not in pass1_output:
        return ""
    start = pass1_output.find(_CONTEXT_START) + len(_CONTEXT_START)
    end = pass1_output.find(_CONTEXT_END)
    if end == -1:
        return pass1_output[start:].strip()
    content = pass1_output[start:end].strip()
    return content


def has_context(pass1_output: str) -> bool:
    """Return True if Pass 1 produced any retrievable context (non-empty block)."""
    return bool(extract_context_block(pass1_output).strip())


def _empty_block(note: str) -> str:
    return (
        f"{_CONTEXT_START}\n"
        f"{_CONTEXT_END}\n"
        f"RETRIEVAL_NOTES: {note} Pass 2 should proceed without substrate context."
    )

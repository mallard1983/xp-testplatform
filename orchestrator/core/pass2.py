"""
Pass 2 — Inference with curated context.

Receives the current question and the context block from Pass 1.
Builds a tight message array: system prompt (with context injected) + current question.
Calls the model with search + MCP tools available.

If Pass 1 produced no context (empty substrate or failed retrieval), Pass 2
receives only the base system prompt — it proceeds without substrate context.
"""

from __future__ import annotations

from core.pass1 import extract_context_block, has_context


_CONTEXT_HEADER = """

## Prior Conversation Context

The following context was retrieved from prior exchanges and is relevant \
to the current question. Use it to reason with continuity and specificity. \
Do not re-derive what has already been established.

"""

_CONTEXT_FOOTER = "\n\n---\n"


async def run_pass2(
    current_question: str,
    pass1_output: str,
    client,              # OllamaClient
    test_model_system_prompt: str,
    tools: list[dict],
    tool_dispatch: dict,
) -> tuple[str, list[dict], dict]:
    """
    Run Pass 2 inference.

    Builds the Pass 2 message array:
      - System: base system prompt + context block (if any)
      - User: current question only

    Returns (response_text, tool_events, usage).
    """
    system_content = _build_system(test_model_system_prompt, pass1_output)

    pass2_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": current_question},
    ]

    text, tool_events, usage = await client.run_with_tools(
        messages=pass2_messages,
        tools=tools,
        tool_dispatch=tool_dispatch,
    )

    return text or "", tool_events, usage


def _build_system(base_prompt: str, pass1_output: str) -> str:
    """Inject the context block into the system prompt if Pass 1 found anything."""
    if not has_context(pass1_output):
        return base_prompt

    context_content = extract_context_block(pass1_output)
    return base_prompt + _CONTEXT_HEADER + context_content + _CONTEXT_FOOTER

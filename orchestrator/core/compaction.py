"""
Compaction logic for the baseline condition.

Token counting uses tiktoken when available (inside the container) and falls
back to a character-based approximation (~4 chars/token) for environments
without it. The approximation is sufficient for threshold decisions.

Compaction fires when the message array token count exceeds the threshold.
The model is asked to produce a structured summary; the history is then
replaced with that summary so the conversation can continue within the
context window.
"""

from __future__ import annotations

import json


# ── Token counting ────────────────────────────────────────────────────────────

def _encode_length(text: str) -> int:
    """Token count for a string. Uses tiktoken if available, falls back to char estimate."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except (ImportError, Exception):
        # Fallback: ~4 characters per token (reasonable for English text)
        return max(1, len(text) // 4)


def count_tokens(messages: list[dict]) -> int:
    """
    Estimate total token count for a message array.
    Accounts for message structure overhead (~4 tokens per message for role/framing).
    """
    total = 0
    for msg in messages:
        total += 4  # per-message overhead (role, framing)
        for key, value in msg.items():
            if isinstance(value, str):
                total += _encode_length(value)
            elif isinstance(value, list):
                # tool_calls or similar — serialize to estimate
                total += _encode_length(json.dumps(value))
    total += 2  # reply primer
    return total


def needs_compaction(token_count: int, context_window: int, threshold_fraction: float) -> bool:
    """Return True if the API-reported prompt token count has exceeded the compaction threshold."""
    threshold = int(context_window * threshold_fraction)
    return token_count >= threshold


# ── Compaction execution ──────────────────────────────────────────────────────

async def run_compaction(
    messages: list[dict],
    client,  # OllamaClient — not typed here to avoid circular import
    compaction_prompt: str,
    system_prompt: str,
    turn: int,
) -> tuple[str, list[dict], dict]:
    """
    Ask the model to summarize the conversation to date.

    Sends the full message history plus the compaction prompt as a final
    user message. The model produces a structured summary. The history is
    then replaced with:
      - The original system message
      - A user message containing the labeled summary
      - A brief assistant acknowledgment (to maintain alternating structure)

    Returns:
      (summary_text, new_message_array, usage)
    """
    compaction_messages = list(messages) + [
        {"role": "user", "content": compaction_prompt}
    ]

    response = await client.chat(compaction_messages)
    summary = response["content"] or ""
    usage = response["usage"]

    new_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"[CONVERSATION SUMMARY — turns 1 through {turn}]\n\n"
                f"{summary}\n\n"
                f"[END SUMMARY — conversation continues below]"
            ),
        },
        {
            "role": "assistant",
            "content": "Understood. Continuing from the established context.",
        },
    ]

    return summary, new_messages, usage

"""
Proxy condition turn loop.

Flow per turn:
  1. Ask interviewer for its next question.
  2. Build current message array (proxy_history + current question).
  3. Check Pass 1 activation threshold.
     - Below: pass full history straight to Pass 2 (no substrate query).
     - Above: run Pass 1 → get context block → run Pass 2 with injected context.
  4. Log all events under the current turn number.
  5. Async write-back: fire-and-forget substrate write (does not block next turn).
  6. Update proxy_history and interviewer_messages.
  7. Checkpoint if applicable. Pause for rate limiting.

Turn 0 (opening exchange) runs before the loop and is not counted.
The closing prompt is delivered after turn 200 by the caller (experiment.py).
"""

from __future__ import annotations

import asyncio
from typing import Callable, Awaitable

from core.compaction import count_tokens
from core.logger import RunLogger
from core.pass1 import run_pass1, extract_context_block, has_context
from core.pass2 import run_pass2
from core.search import web_search
from core.tool_handler import pass2_tools


async def run_proxy(
    *,
    pass1_client,          # OllamaClient — same model as Pass 2 for this experiment
    pass2_client,          # OllamaClient — test model
    interviewer_client,    # OllamaClient — GLM-5
    substrate_client,      # SubstrateClient
    test_model_system_prompt: str,
    interviewer_system_prompt: str,
    pass1_system_prompt: str,
    opening_question: str,
    closing_prompt: str,
    context_window: int,
    pass1_activation_fraction: float,
    turn_limit: int,
    turn_pause_seconds: float,
    checkpoint_turns: list[int],
    logger: RunLogger,
    pass1_tool_budget: int = 8,
    extra_mcp_tools: list[dict] | None = None,
    extra_tool_dispatch: dict[str, Callable[..., Awaitable[str]]] | None = None,
    on_turn_complete: Callable[..., Awaitable[None]] | None = None,
) -> dict:
    """
    Run the full proxy condition.

    Returns a summary dict:
      {
        "turns_completed": int,
        "pass1_activations": int,
        "total_tokens": {"prompt": int, "completion": int},
        "checkpoint_turns": list[int],
      }
    """
    activation_threshold = int(context_window * pass1_activation_fraction)

    tools = pass2_tools(extra_mcp_tools)
    tool_dispatch: dict[str, Callable[..., Awaitable[str]]] = {"web_search": web_search}
    if extra_tool_dispatch:
        tool_dispatch.update(extra_tool_dispatch)

    pass1_activations = 0
    total_tokens  = {"prompt": 0, "completion": 0}
    pass1_tokens  = {"prompt": 0, "completion": 0}
    pass2_tokens  = {"prompt": 0, "completion": 0}
    completed_checkpoints: list[int] = []
    pending_writes: list[asyncio.Task] = []

    # ── Initialize proxy history ──────────────────────────────────────────────
    proxy_history: list[dict] = [
        {"role": "system", "content": test_model_system_prompt},
        {"role": "user", "content": opening_question},
    ]

    # ── Turn 0: opening exchange ──────────────────────────────────────────────
    logger.log_api_request(0, "pass2", pass2_client.model_identifier, proxy_history, tools)

    opening_text, opening_tool_events, opening_usage = await pass2_client.run_with_tools(
        messages=proxy_history,
        tools=tools,
        tool_dispatch=tool_dispatch,
    )
    _accumulate(total_tokens, opening_usage)
    _accumulate(pass2_tokens, opening_usage)

    for ev in opening_tool_events:
        logger.log_tool_call(0, "pass2", ev["tool"], ev["args"], ev["result"], ev["error"])
    logger.log_api_response(0, "pass2", pass2_client.model_identifier,
                            opening_text, [], opening_usage)

    proxy_history.append({"role": "assistant", "content": opening_text})

    opening_stats = {
        "total_tokens": dict(total_tokens),
        "pass1_tokens": dict(pass1_tokens),
        "pass2_tokens": dict(pass2_tokens),
        "pass1_activations": 0,
        "turn_limit": turn_limit,
    }
    logger.log_opening_complete(opening_question, opening_text, opening_usage, opening_stats)
    if on_turn_complete:
        await on_turn_complete(0, opening_question, opening_text, opening_stats)

    # Async write-back for opening exchange
    pending_writes.append(asyncio.create_task(
        _write_exchange(substrate_client, 0, opening_question, opening_text)
    ))

    # ── Initialize interviewer with opening response ───────────────────────────
    interviewer_messages: list[dict] = [
        {"role": "system", "content": interviewer_system_prompt},
        {"role": "user", "content": opening_text},
    ]

    # ── Main turn loop ────────────────────────────────────────────────────────
    for turn in range(1, turn_limit + 1):

        # 1. Get interviewer question
        logger.log_api_request(turn, "interviewer", interviewer_client.model_identifier,
                               interviewer_messages)
        iv_response = await interviewer_client.chat(interviewer_messages)
        iv_question = (iv_response["content"] or "").strip()
        logger.log_api_response(turn, "interviewer", interviewer_client.model_identifier,
                                iv_question, [], iv_response["usage"])

        # 2. Build current message array for this turn
        current_messages = proxy_history + [{"role": "user", "content": iv_question}]
        token_count = count_tokens(current_messages)

        # 3. Pass 1 activation check
        if token_count < activation_threshold:
            # Below threshold: pass straight through to Pass 2 with full history
            logger.log_api_request(turn, "pass2", pass2_client.model_identifier,
                                   current_messages, tools)
            model_text, tool_events, model_usage = await pass2_client.run_with_tools(
                messages=current_messages,
                tools=tools,
                tool_dispatch=tool_dispatch,
            )
            _accumulate(total_tokens, model_usage)
            _accumulate(pass2_tokens, model_usage)
            for ev in tool_events:
                logger.log_tool_call(turn, "pass2", ev["tool"], ev["args"],
                                     ev["result"], ev["error"])
            logger.log_api_response(turn, "pass2", pass2_client.model_identifier,
                                    model_text, [], model_usage)

        else:
            # Above threshold: activate Pass 1
            pass1_activations += 1
            logger.log_api_request(turn, "pass1", pass1_client.model_identifier,
                                   current_messages, None)

            pass1_output, pass1_usage = await run_pass1(
                messages=current_messages,
                client=pass1_client,
                substrate_client=substrate_client,
                pass1_system_prompt=pass1_system_prompt,
                tool_budget=pass1_tool_budget,
            )
            _accumulate(total_tokens, pass1_usage)
            _accumulate(pass1_tokens, pass1_usage)

            context_block = extract_context_block(pass1_output)
            logger.log_api_response(turn, "pass1", pass1_client.model_identifier,
                                    pass1_output, [], pass1_usage)

            # Pass 2 with injected context
            logger.log_api_request(turn, "pass2", pass2_client.model_identifier, [], tools)
            model_text, tool_events, model_usage = await run_pass2(
                current_question=iv_question,
                pass1_output=pass1_output,
                client=pass2_client,
                test_model_system_prompt=test_model_system_prompt,
                tools=tools,
                tool_dispatch=tool_dispatch,
            )
            _accumulate(total_tokens, model_usage)
            _accumulate(pass2_tokens, model_usage)
            for ev in tool_events:
                logger.log_tool_call(turn, "pass2", ev["tool"], ev["args"],
                                     ev["result"], ev["error"])
            logger.log_api_response(turn, "pass2", pass2_client.model_identifier,
                                    model_text, [], model_usage)

        # 4. Log turn
        turn_stats = {
            "total_tokens": dict(total_tokens),
            "pass1_tokens": dict(pass1_tokens),
            "pass2_tokens": dict(pass2_tokens),
            "pass1_activations": pass1_activations,
            "turn_limit": turn_limit,
        }
        logger.log_turn_complete(turn, iv_question, model_text, model_usage, turn_stats)
        logger.transcript_turn(turn, iv_question, model_text)

        # 5. Update histories
        proxy_history.append({"role": "user", "content": iv_question})
        proxy_history.append({"role": "assistant", "content": model_text})

        interviewer_messages.append({"role": "assistant", "content": iv_question})
        interviewer_messages.append({"role": "user", "content": model_text})

        # 6. Notify streaming callback
        if on_turn_complete:
            await on_turn_complete(turn, iv_question, model_text, turn_stats)

        # 7. Async write-back (fire and forget — does not block next turn)
        pending_writes.append(asyncio.create_task(
            _write_exchange(substrate_client, turn, iv_question, model_text)
        ))

        # 8. Checkpoint
        if turn in checkpoint_turns:
            logger.log_checkpoint(turn, model_text)
            completed_checkpoints.append(turn)

        # 9. Rate limit pause
        if turn < turn_limit and turn_pause_seconds > 0:
            await asyncio.sleep(turn_pause_seconds)

    # Allow pending writes to complete (best effort — don't block on failures)
    if pending_writes:
        await asyncio.gather(*pending_writes, return_exceptions=True)

    return {
        "turns_completed": turn_limit,
        "pass1_activations": pass1_activations,
        "total_tokens": total_tokens,
        "pass1_tokens": pass1_tokens,
        "pass2_tokens": pass2_tokens,
        "checkpoint_turns": completed_checkpoints,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _write_exchange(
    substrate_client,
    turn: int,
    question: str,
    answer: str,
) -> None:
    """Write a turn's exchange to the substrate as a ThoughtStream."""
    try:
        content = f"Turn {turn}\n\nQuestion: {question}\n\nAnswer: {answer}"
        summary = f"Turn {turn} reasoning exchange"
        await substrate_client.create_stream(
            content=content,
            summary=summary,
            source="conversation",
            source_id=f"turn-{turn}",
            completion_state="complete",
            confidence=0.7,
        )
    except Exception:
        # Write-back failures are non-fatal — the conversation continues
        pass


def _accumulate(total: dict, usage: dict) -> None:
    total["prompt"] += usage.get("prompt_tokens", 0)
    total["completion"] += usage.get("completion_tokens", 0)

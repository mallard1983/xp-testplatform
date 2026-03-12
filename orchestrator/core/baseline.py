"""
Baseline condition turn loop.

The orchestrator owns the test model's message array. The interviewer (GLM-5)
owns its own history. Both accumulate over 200 turns.

Flow per turn:
  1. Ask interviewer for its next question (given full conversation so far).
  2. Check test model's message array for compaction threshold.
  3. If threshold exceeded: fire compaction, replace history, log event.
  4. Append interviewer question to test model messages.
  5. Call test model (with search + MCP tools available).
  6. Log turn; append response to both message arrays.
  7. If turn is a checkpoint: log checkpoint artifact.
  8. Pause (rate limit buffer).

Turn 0 (opening exchange) is handled before the loop and not counted.
The closing prompt is delivered after turn 200 by the caller (experiment.py).
"""

from __future__ import annotations

import asyncio
from typing import Callable, Awaitable

from core.compaction import count_tokens, needs_compaction, run_compaction
from core.logger import RunLogger
from core.search import web_search
from core.tool_handler import baseline_tools


async def run_baseline(
    *,
    pass2_client,          # OllamaClient for the test model (Pass 2 role in baseline)
    interviewer_client,    # OllamaClient for the interviewer (GLM-5)
    test_model_system_prompt: str,
    interviewer_system_prompt: str,
    opening_question: str,
    closing_prompt: str,
    compaction_prompt: str,
    context_window: int,
    compaction_threshold_fraction: float,
    turn_limit: int,
    turn_pause_seconds: float,
    checkpoint_turns: list[int],
    logger: RunLogger,
    extra_mcp_tools: list[dict] | None = None,
    extra_tool_dispatch: dict[str, Callable[..., Awaitable[str]]] | None = None,
    on_turn_complete: Callable[..., Awaitable[None]] | None = None,
) -> dict:
    """
    Run the full baseline condition.

    Returns a summary dict:
      {
        "turns_completed": int,
        "compaction_events": list[{"turn": int, "tokens_before": int}],
        "total_tokens": {"prompt": int, "completion": int},
        "checkpoint_turns": list[int],
      }
    """
    tools = baseline_tools(extra_mcp_tools)
    tool_dispatch: dict[str, Callable[..., Awaitable[str]]] = {"web_search": web_search}
    if extra_tool_dispatch:
        tool_dispatch.update(extra_tool_dispatch)

    compaction_events: list[dict] = []
    total_tokens = {"prompt": 0, "completion": 0}
    completed_checkpoints: list[int] = []

    # ── Initialize message arrays ─────────────────────────────────────────────

    # Test model messages (orchestrator-managed, subject to compaction)
    test_messages: list[dict] = [
        {"role": "system", "content": test_model_system_prompt},
        {"role": "user", "content": opening_question},
    ]

    # ── Turn 0: opening exchange ──────────────────────────────────────────────
    logger.log_api_request(0, "baseline", pass2_client.model_identifier, test_messages, tools)

    opening_text, opening_tool_events, opening_usage = await pass2_client.run_with_tools(
        messages=test_messages,
        tools=tools,
        tool_dispatch=tool_dispatch,
    )
    _accumulate(total_tokens, opening_usage)

    for ev in opening_tool_events:
        logger.log_tool_call(0, "baseline", ev["tool"], ev["args"], ev["result"], ev["error"])

    logger.log_api_response(0, "baseline", pass2_client.model_identifier,
                            opening_text, [], opening_usage)

    test_messages.append({"role": "assistant", "content": opening_text})

    opening_stats = {
        "turn_tokens": count_tokens(test_messages),
        "total_tokens": dict(total_tokens),
        "compaction_count": 0,
        "turn_limit": turn_limit,
    }
    logger.log_opening_complete(opening_question, opening_text, opening_usage, opening_stats)
    if on_turn_complete:
        await on_turn_complete(0, opening_question, opening_text, opening_stats)

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

        # 2. Check compaction threshold
        tokens_before = count_tokens(test_messages)
        if needs_compaction(test_messages, context_window, compaction_threshold_fraction):
            summary, test_messages, compact_usage = await run_compaction(
                messages=test_messages,
                client=pass2_client,
                compaction_prompt=compaction_prompt,
                system_prompt=test_model_system_prompt,
                turn=turn - 1,
            )
            _accumulate(total_tokens, compact_usage)
            compaction_events.append({"turn": turn, "tokens_before": tokens_before})
            logger.log_compaction(turn, summary, tokens_before)
            logger.transcript_compaction_note(turn)

        # 3. Append interviewer question to test model messages
        test_messages.append({"role": "user", "content": iv_question})

        # 4. Call test model
        logger.log_api_request(turn, "baseline", pass2_client.model_identifier,
                               test_messages, tools)

        model_text, tool_events, model_usage = await pass2_client.run_with_tools(
            messages=test_messages,
            tools=tools,
            tool_dispatch=tool_dispatch,
        )
        _accumulate(total_tokens, model_usage)

        for ev in tool_events:
            logger.log_tool_call(turn, "baseline", ev["tool"], ev["args"],
                                 ev["result"], ev["error"])

        logger.log_api_response(turn, "baseline", pass2_client.model_identifier,
                                model_text, [], model_usage)

        # 5. Append response to test model messages
        test_messages.append({"role": "assistant", "content": model_text})

        # 6. Log turn
        turn_stats = {
            "turn_tokens": count_tokens(test_messages),
            "total_tokens": dict(total_tokens),
            "compaction_count": len(compaction_events),
            "turn_limit": turn_limit,
        }
        logger.log_turn_complete(turn, iv_question, model_text, model_usage, turn_stats)
        logger.transcript_turn(turn, iv_question, model_text)

        # 7. Update interviewer history (sees the full back-and-forth)
        interviewer_messages.append({"role": "assistant", "content": iv_question})
        interviewer_messages.append({"role": "user", "content": model_text})

        # 8. Checkpoint
        if turn in checkpoint_turns:
            logger.log_checkpoint(turn, model_text)
            completed_checkpoints.append(turn)

        # 9. Notify streaming callback
        if on_turn_complete:
            await on_turn_complete(turn, iv_question, model_text, turn_stats)

        # 10. Rate limit pause (skip on last turn to avoid unnecessary wait)
        if turn < turn_limit and turn_pause_seconds > 0:
            await asyncio.sleep(turn_pause_seconds)

    return {
        "turns_completed": turn_limit,
        "compaction_events": compaction_events,
        "total_tokens": total_tokens,
        "checkpoint_turns": completed_checkpoints,
    }


def _accumulate(total: dict, usage: dict) -> None:
    total["prompt"] += usage.get("prompt_tokens", 0)
    total["completion"] += usage.get("completion_tokens", 0)

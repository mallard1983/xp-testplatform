"""
Ollama cloud client — thin wrapper over the OpenAI-compatible chat completions API.

Handles:
- Single chat completion calls
- Tool call loop (model calls a tool → dispatch → continue → final response)
- Token usage extraction for logging
- API key injection via Authorization header
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable, Awaitable

import httpx

# Default timeout in seconds. Increase via LLM_TIMEOUT_SECONDS env var for slow local models.
_DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT_SECONDS", "600"))

_USER_AGENT = "XP-TestPlatform/1.0"
_MAX_RETRIES = 3


class OllamaClient:
    def __init__(self, endpoint_url: str, model_identifier: str, api_key: str | None = None):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_identifier = model_identifier
        self.api_key = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> dict:
        """
        Single chat completion. Returns the full response dict including:
          - response["content"]: text content (may be None if tool calls present)
          - response["tool_calls"]: list of tool call dicts (may be empty)
          - response["usage"]: {"prompt_tokens": int, "completion_tokens": int}
          - response["raw"]: full API response for logging
        """
        payload: dict[str, Any] = {
            "model": self.model_identifier,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        resp = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    resp = await client.post(
                        f"{self.endpoint_url}/v1/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 30))
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(retry_after)
                        continue
                    raise RuntimeError(
                        f"Rate limited (HTTP 429) after {_MAX_RETRIES + 1} attempts "
                        f"[model={self.model_identifier}]"
                    )

                if resp.status_code >= 500:
                    if attempt < _MAX_RETRIES:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    raise RuntimeError(
                        f"HTTP {resp.status_code} server error "
                        f"[model={self.model_identifier}]: {resp.text[:300]} "
                        f"— {_MAX_RETRIES} retries exhausted"
                    )

                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise RuntimeError(
                        f"HTTP {exc.response.status_code} error "
                        f"[model={self.model_identifier}]: {exc.response.text[:300]}"
                    ) from exc

                break  # success

            except httpx.TimeoutException as exc:
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Request timed out after {timeout}s "
                    f"[model={self.model_identifier} endpoint={self.endpoint_url}] "
                    f"— {_MAX_RETRIES} retries exhausted"
                ) from exc

            except httpx.ConnectError as exc:
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Connection failed "
                    f"[model={self.model_identifier} endpoint={self.endpoint_url}]: {exc} "
                    f"— {_MAX_RETRIES} retries exhausted"
                ) from exc

        raw = resp.json()

        choice = raw["choices"][0]
        message = choice["message"]
        usage = raw.get("usage", {})

        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls") or [],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            "raw": raw,
        }

    async def run_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_dispatch: dict[str, Callable[..., Awaitable[str]]],
        max_iterations: int = 12,
    ) -> tuple[str, list[dict], dict]:
        """
        Chat completion loop with tool handling.

        On each iteration:
          1. Call the model with the current messages and tool list.
          2. If the response contains tool_calls, execute each, append results, repeat.
          3. If no tool_calls, return the final text response.

        Returns:
          (final_text, tool_events, total_usage)

          tool_events is a list of dicts for logging:
            {"tool": name, "args": dict, "result": str, "error": bool}

          total_usage accumulates prompt + completion tokens across all iterations.
        """
        current_messages = list(messages)
        tool_events: list[dict] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        last_prompt_tokens: int = 0

        for _ in range(max_iterations):
            response = await self.chat(current_messages, tools=tools)
            last_prompt_tokens = response["usage"]["prompt_tokens"]
            total_usage["prompt_tokens"] += last_prompt_tokens
            total_usage["completion_tokens"] += response["usage"]["completion_tokens"]

            if not response["tool_calls"]:
                # last_prompt_tokens = actual context window size at the final call.
                # total_usage["prompt_tokens"] is the billing sum across all tool iterations.
                usage_out = {**total_usage, "last_prompt_tokens": last_prompt_tokens}
                return response["content"] or "", tool_events, usage_out

            # Append assistant message with tool_calls
            current_messages.append({
                "role": "assistant",
                "content": response["content"],
                "tool_calls": response["tool_calls"],
            })

            # Execute each tool call and append results
            for tc in response["tool_calls"]:
                tool_name = tc["function"]["name"]
                try:
                    raw_args = tc["function"].get("arguments", "{}")
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    args = {}

                event: dict = {"tool": tool_name, "args": args, "error": False}

                handler = tool_dispatch.get(tool_name)
                if handler is None:
                    result = f"Error: unknown tool '{tool_name}'"
                    event["error"] = True
                else:
                    try:
                        result = await handler(**args)
                    except Exception as exc:
                        result = f"Error: {exc}"
                        event["error"] = True

                event["result"] = result
                tool_events.append(event)

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })

        # Reached max_iterations without a non-tool response
        return "", tool_events, {**total_usage, "last_prompt_tokens": last_prompt_tokens}

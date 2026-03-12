"""
Run queue — serialises experiment runs so they execute one at a time.

When a run is enqueued and nothing is active it starts immediately.
When a run is enqueued while one is active it waits until that run
completes (or errors or is cancelled), then advances automatically.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional


class RunQueue:
    def __init__(self) -> None:
        self._pending: list[str] = []           # ordered run_ids waiting to execute
        self._active_run_id: Optional[str] = None
        self._execute_fn: Optional[Callable[[str], Awaitable[None]]] = None

    def set_execute_fn(self, fn: Callable[[str], Awaitable[None]]) -> None:
        self._execute_fn = fn

    def enqueue(self, run_id: str) -> None:
        self._pending.append(run_id)
        if self._active_run_id is None:
            asyncio.create_task(self._advance())

    def remove(self, run_id: str) -> bool:
        try:
            self._pending.remove(run_id)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        self._pending.clear()

    def notify_complete(self) -> None:
        """Called when the active run finishes — advance to next if any are queued."""
        self._active_run_id = None
        if self._pending:
            asyncio.create_task(self._advance())

    @property
    def pending_ids(self) -> list[str]:
        return list(self._pending)

    @property
    def active_run_id(self) -> Optional[str]:
        return self._active_run_id

    async def _advance(self) -> None:
        if self._active_run_id is not None or not self._pending or not self._execute_fn:
            return
        run_id = self._pending.pop(0)
        self._active_run_id = run_id
        try:
            await self._execute_fn(run_id)
        except Exception:
            logging.exception("Queue failed to start run %s", run_id)
            self.notify_complete()


run_queue = RunQueue()

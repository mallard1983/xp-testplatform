"""
Neo4j manager — stop/remount/restart between proxy test runs.

The Docker Compose stack binds neo4j's data to ./data/active_neo4j/.
Before each proxy run:
  1. Stop the neo4j container.
  2. Clear or populate ./data/active_neo4j/ (fresh vs. rerun from snapshot).
  3. Start the neo4j container.
  4. Wait for neo4j to report healthy.

After each proxy run, save_snapshot() copies active_neo4j to a timestamped
directory so the run's substrate state is preserved for future reruns.

Baseline runs do not interact with neo4j (no substrate write-back).

Uses the Python docker SDK (already in requirements.txt via mcp) to avoid
needing the docker CLI binary inside the container.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path


class Neo4jManager:
    def __init__(
        self,
        data_dir: Path | None = None,
        container_name: str | None = None,
    ):
        self.data_dir = data_dir or Path(os.environ.get("DATA_DIR", "/app/data"))
        self.container_name = container_name or os.environ.get(
            "NEO4J_CONTAINER_NAME", "xp-testplatform-neo4j-1"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def prepare_fresh(self) -> None:
        """Clear active_neo4j and restart neo4j with a clean database."""
        await self._stop()
        self._clear_active()
        await self._start()
        await self._wait_healthy()

    async def prepare_from_snapshot(self, test_id: str, source_timestamp: str) -> None:
        """
        Restore a prior proxy run's snapshot into active_neo4j and restart neo4j.
        The source snapshot is never mutated — a new run always starts from a copy.
        """
        snapshot = self._snapshot_dir(test_id, source_timestamp)
        if not snapshot.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot}")
        await self._stop()
        self._clear_active()
        self._populate_active_from(snapshot)
        await self._start()
        await self._wait_healthy()

    def save_snapshot(self, test_id: str, timestamp: str) -> Path:
        """
        Copy active_neo4j contents to a timestamped snapshot directory.
        Returns the snapshot path.
        """
        dest = self._snapshot_dir(test_id, timestamp)
        dest.mkdir(parents=True, exist_ok=True)
        active = self._active_dir()
        for sub in ("data", "logs"):
            src = active / sub
            if src.exists():
                shutil.copytree(src, dest / sub, dirs_exist_ok=True)
        return dest

    def list_snapshots(self, test_id: str) -> list[str]:
        """Return sorted list of available snapshot timestamps for a test."""
        base = self.data_dir / "tests" / test_id / "neo4j"
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _active_dir(self) -> Path:
        return self.data_dir / "active_neo4j"

    def _snapshot_dir(self, test_id: str, timestamp: str) -> Path:
        return self.data_dir / "tests" / test_id / "neo4j" / timestamp

    def _clear_active(self) -> None:
        """
        Empty data/ and logs/ inside active_neo4j/ without removing the directories.

        We must NOT delete the directories themselves — on WSL2/Docker Desktop the
        bind-mount source paths are registered by inode. Deleting and recreating a
        directory changes the inode and causes Docker to fail with
        'error while creating mount source path … file exists' on the next start.
        """
        active = self._active_dir()
        active.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "logs"):
            sub_dir = active / sub
            sub_dir.mkdir(exist_ok=True)
            for item in sub_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    def _populate_active_from(self, snapshot: Path) -> None:
        """Copy snapshot subdirs into active_neo4j/."""
        active = self._active_dir()
        for sub in ("data", "logs"):
            src = snapshot / sub
            if src.exists():
                shutil.copytree(src, active / sub, dirs_exist_ok=True)

    def _get_container(self):
        """Return the docker container object. Raises if not found."""
        import docker  # type: ignore
        client = docker.from_env()
        return client.containers.get(self.container_name)

    async def _stop(self) -> None:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._get_container().stop()
        )

    async def _start(self) -> None:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._get_container().start()
        )

    async def _wait_healthy(self, timeout: float = 90.0, interval: float = 3.0) -> None:
        """Poll container health status until healthy."""
        import docker  # type: ignore
        client = docker.from_env()
        elapsed = 0.0
        while elapsed < timeout:
            container = client.containers.get(self.container_name)
            health = container.attrs.get("State", {}).get("Health", {}).get("Status", "")
            if health == "healthy":
                return
            await asyncio.sleep(interval)
            elapsed += interval
        raise TimeoutError(
            f"{self.container_name} did not become healthy within {timeout}s"
        )

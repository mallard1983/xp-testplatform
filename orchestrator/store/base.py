"""
Shared file I/O helpers for the store layer.
All store modules read from and write to DATA_DIR (env var, default /app/data).
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def get_data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/app/data"))


def setup_dir() -> Path:
    return get_data_dir() / "setup"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

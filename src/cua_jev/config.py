from __future__ import annotations

import os
from pathlib import Path

_LOCAL_ENV_KEYS = {
    "TYPESAFE_API_KEY", "CUA_JEV_MODEL", "CUA_JEV_API_URL",
    "CUA_JEV_MODEL_API_KEY", "CUA_JEV_PLANNER_API_KEY",
}


def load_local_env(path: str | Path = ".env") -> bool:
    """Load allowlisted local settings without overriding the process environment."""
    source = Path(path)
    if not source.is_file():
        return False
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in _LOCAL_ENV_KEYS:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value:
            os.environ.setdefault(key, value)
    return True

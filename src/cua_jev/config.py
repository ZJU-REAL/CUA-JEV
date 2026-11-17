from __future__ import annotations

import json
import os
from pathlib import Path

_LOCAL_ENV_KEYS = {
    "TYPESAFE_API_KEY", "CUA_JEV_MODEL", "CUA_JEV_API_URL",
    "CUA_JEV_MODEL_API_KEY", "CUA_JEV_PLANNER_API_KEY",
    "CUA_JEV_MODEL_BASE_URL", "CUA_JEV_VLM_MODEL",
    "CUA_JEV_SERVICE_PROFILE",
}


def load_local_env(path: str | Path = ".env") -> bool:
    """Load allowlisted local settings without overriding the process environment."""
    source = Path(path)
    if not source.is_file():
        if profile := os.getenv("CUA_JEV_SERVICE_PROFILE"):
            load_service_profile(profile)
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
    if profile := os.getenv("CUA_JEV_SERVICE_PROFILE"):
        load_service_profile(profile)
    return True


def load_service_profile(path: str | Path) -> None:
    """Load an explicitly selected private JSON profile; never log its contents."""
    try:
        profile = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        values = {
            "TYPESAFE_API_KEY": profile["jev"]["api_key"],
            "CUA_JEV_API_URL": profile["jev"]["api_url"],
            "CUA_JEV_MODEL_API_KEY": profile["vlm"]["api_key"],
            "CUA_JEV_MODEL_BASE_URL": profile["vlm"]["base_url"],
        }
        for key, value in (("CUA_JEV_MODEL", profile["jev"].get("model")),
                           ("CUA_JEV_VLM_MODEL", profile["vlm"].get("model"))):
            if value is not None:
                values[key] = value
        if not all(isinstance(v, str) and v.strip() for v in values.values()):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ValueError("invalid or unreadable private service profile") from None
    for key, value in values.items():
        os.environ.setdefault(key, value)

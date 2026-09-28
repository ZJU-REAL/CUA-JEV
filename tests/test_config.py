import os
from pathlib import Path

from cua_jev.config import load_local_env


def test_local_env_loads_only_allowlisted_missing_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("CUA_JEV_MODEL", "existing")
    source = tmp_path / ".env"
    source.write_text(
        "TYPESAFE_API_KEY='local-secret'\n"
        "CUA_JEV_MODEL=ignored\n"
        "UNRELATED_SETTING=not-loaded\n",
        encoding="utf-8",
    )

    assert load_local_env(source)
    assert os.environ["TYPESAFE_API_KEY"] == "local-secret"
    assert os.environ["CUA_JEV_MODEL"] == "existing"
    assert "UNRELATED_SETTING" not in os.environ


def test_planner_credentials_load_locally_without_overriding_process(tmp_path, monkeypatch):
    monkeypatch.delenv("CUA_JEV_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("CUA_JEV_PLANNER_API_KEY", "process-value")
    path = tmp_path / ".env"
    path.write_text("CUA_JEV_MODEL_API_KEY=local-value\nCUA_JEV_PLANNER_API_KEY=ignored\nPATH=bad\n")
    old_path = os.environ.get("PATH")
    load_local_env(path)
    assert os.environ["CUA_JEV_MODEL_API_KEY"] == "local-value"
    assert os.environ["CUA_JEV_PLANNER_API_KEY"] == "process-value"
    assert os.environ.get("PATH") == old_path

import json
import os
from pathlib import Path

import pytest

from cua_jev.config import load_local_env, load_service_profile


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


def test_agent_model_settings_load_without_overriding_process_or_granting_opt_ins(tmp_path, monkeypatch):
    values = {
        "CUA_JEV_MODEL_BASE_URL": "https://model.test/v1",
        "CUA_JEV_VLM_MODEL": "local-config-vlm",
        "CUA_JEV_MODEL_API_KEY": "test-model-key",
        "CUA_JEV_PLANNER_API_KEY": "test-planner-key",
    }
    for key in values:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CUA_JEV_VLM_MODEL", "process-vlm")
    opt_ins = ("CUA_JEV_ALLOW_SCREENSHOT_UPLOAD", "CUA_JEV_ALLOW_ACTIONS", "CUA_JEV_ALLOW_INSECURE_HTTP")
    for key in opt_ins:
        monkeypatch.delenv(key, raising=False)
    source = tmp_path / ".env"
    source.write_text(
        "\n".join(f'{key}="{value}"' for key, value in values.items())
        + "\n" + "\n".join(f"{key}=1" for key in opt_ins),
        encoding="utf-8",
    )
    assert load_local_env(source)
    for key, value in values.items():
        assert os.environ[key] == ("process-vlm" if key == "CUA_JEV_VLM_MODEL" else value)
    assert all(key not in os.environ for key in opt_ins)


def test_private_service_profile_is_explicit_and_preserves_environment(tmp_path, monkeypatch):
    profile = tmp_path / "services.json"
    profile.write_text(json.dumps({
        "jev": {"api_key": "fixture-jev", "api_url": "https://jev.test", "model": "jev-test"},
        "vlm": {"api_key": "fixture-vlm", "base_url": "http://local.test/v1", "model": "vlm-test",
                "allow_insecure_http": True},
    }))
    for key in ("TYPESAFE_API_KEY", "CUA_JEV_API_URL", "CUA_JEV_MODEL", "CUA_JEV_MODEL_API_KEY",
                "CUA_JEV_MODEL_BASE_URL", "CUA_JEV_VLM_MODEL", "CUA_JEV_ALLOW_INSECURE_HTTP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CUA_JEV_MODEL_API_KEY", "process-key")
    monkeypatch.setenv("CUA_JEV_SERVICE_PROFILE", str(profile))
    assert load_local_env(tmp_path / "missing.env") is False
    assert os.environ["TYPESAFE_API_KEY"] == "fixture-jev"
    assert os.environ["CUA_JEV_MODEL_API_KEY"] == "process-key"
    assert os.environ["CUA_JEV_VLM_MODEL"] == "vlm-test"
    assert os.environ["CUA_JEV_MODEL"] == "jev-test"
    assert "CUA_JEV_ALLOW_INSECURE_HTTP" not in os.environ


@pytest.mark.parametrize("contents", ['PRIVATE invalid JSON', '{}', '{"jev": null}', '[]'])
def test_invalid_service_profile_does_not_expose_contents(tmp_path, contents):
    profile = tmp_path / "services.json"
    profile.write_text(contents)
    with pytest.raises(ValueError, match="^invalid or unreadable private service profile$"):
        load_service_profile(profile)

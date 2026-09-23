import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cua_jev.ui.app import create_app
from cua_jev.ui.manager import RunManager


def test_console_bootstrap_and_local_security(tmp_path: Path) -> None:
    with TestClient(create_app(root=tmp_path, data=tmp_path / "runs")) as client:
        bootstrap = client.get("/api/bootstrap").json()
        assert set(bootstrap["tasks"]) == {"edge", "excel", "vscode", "explorer"}
        assert "PyAutoGUI · Edge" in bootstrap["tasks"]["edge"]["gui_routes"]
        assert "Playwright DOM" in bootstrap["tasks"]["edge"]["hybrid_routes"]
        assert bootstrap["tasks"]["edge"]["hybrid_channels"] == ["gui", "script"]
        assert bootstrap["tasks"]["edge"]["benchmark_version"] == "long-horizon-v2"
        assert "DOM Script" in bootstrap["tasks"]["edge"]["evaluation_routes"]
        assert client.post("/api/runs", json={"task": "bad", "policy": "rule"}).status_code == 403
        response = client.post(
            "/api/runs",
            json={"task": "bad", "policy": "rule"},
            headers={"X-CUA-JEV-CSRF": bootstrap["csrf"]},
        )
        assert response.status_code == 400


def test_external_codex_baseline_uses_shared_benchmark_contract(tmp_path: Path) -> None:
    with TestClient(create_app(root=tmp_path, data=tmp_path / "runs")) as client:
        bootstrap = client.get("/api/bootstrap").json()
        payload = {
            "agent": "codex_computer_use",
            "task": "edge",
            "action_space": "gui_only",
            "success": True,
            "duration_ms": 42000,
            "actions": 8,
            "channels": {"gui": 8},
            "verifier": "shared_terminal_verifier",
        }
        assert client.post("/api/baselines", json=payload).status_code == 403
        response = client.post(
            "/api/baselines",
            json=payload,
            headers={"X-CUA-JEV-CSRF": bootstrap["csrf"]},
        )
        assert response.status_code == 200
        assert response.json()["metrics"]["gui_ratio"] == 1.0
        rows = client.get("/api/benchmarks?task=edge").json()["rows"]
        assert rows == [
            {
                "agent": "codex_computer_use",
                "action_space": "gui_only",
                "samples": 1,
                "successful_samples": 1,
                "representative_run_id": response.json()["id"],
                "success_rate": 1.0,
                "median_wall_time_ms": 42000.0,
                "mean_wall_time_ms": 42000.0,
                "median_decision_time_ms": None,
                "median_execution_time_ms": None,
                "mean_actions": 8,
                "mean_gui_ratio": 1.0,
                "mean_route_diversity": 1,
                "median_input_tokens": None,
                "median_cached_input_tokens": None,
                "median_output_tokens": None,
                "median_model_cost_usd": None,
                "median_reference_cost_usd": None,
                "median_standard_credits": None,
                "token_samples": 0,
                "cost_samples": 0,
            }
        ]


def test_baseline_steps_are_explicitly_recorded_and_retrievable(tmp_path: Path) -> None:
    with TestClient(create_app(root=tmp_path, data=tmp_path / "runs")) as client:
        headers = {"X-CUA-JEV-CSRF": client.get("/api/bootstrap").json()["csrf"]}
        run = client.post(
            "/api/baselines",
            headers=headers,
            json={
                "agent": "codex_computer_use",
                "task": "edge",
                "action_space": "hybrid",
                "success": True,
                "duration_ms": 1200,
                "actions": 2,
                "channels": {"script": 2},
            },
        ).json()
        path = f"/api/baselines/{run['id']}/steps"
        assert client.get(path.replace("/api/baselines/", "/api/runs/")).json()["steps"] == []
        payload = {
            "source": "local_session_tool_trace",
            "steps": [
                {"title": "Open demo store", "channel": "script"},
                {"title": "Verify receipt", "channel": "verification", "verified": True},
            ],
        }
        assert client.post(path, json=payload).status_code == 403
        assert (
            client.post(
                path, headers=headers, json={**payload, "steps": [{"title": "bad", "channel": ["script"]}]}
            ).status_code
            == 400
        )
        assert client.post(path, headers=headers, json=payload).status_code == 200
        steps = client.get(f"/api/runs/{run['id']}/steps").json()
        assert steps["source"] == "recorded_tool_batches"
        assert steps["terminal_verified"] is True
        assert [item["title"] for item in steps["steps"]] == ["Open demo store", "Verify receipt"]


def test_jev_steps_use_full_trace_without_exposing_raw_state(tmp_path: Path) -> None:
    manager = RunManager(tmp_path, tmp_path / "runs")
    run_id = "jev-steps"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir()
    trace = run_dir / "trace-edge.jsonl"
    events = [
        {"kind": "observation", "payload": {"subgoal": "Repair a test", "state": {"secret": "hidden"}}},
        {
            "kind": "candidates",
            "payload": {
                "items": [
                    {"id": "mcp_fix", "description": "Repair the failing test through the filesystem tool."}
                ]
            },
        },
        {"kind": "decision", "payload": {"latency_ms": 43}},
        {
            "kind": "commitment",
            "payload": {"candidate_id": "mcp_fix", "channel": "mcp", "capability": "filesystem.write_text"},
        },
        {"kind": "receipt", "payload": {"duration_ms": 12, "output": "private content"}},
        {"kind": "verification", "payload": {"passed": True}},
        {"kind": "episode", "payload": {"status": "success"}},
    ]
    trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    manager._save({"id": run_id, "task": "edge", "trace_base": str(run_dir / "trace.jsonl")})
    steps = manager.steps(run_id)
    assert steps["terminal_verified"] is True
    assert steps["steps"] == [
        {
            "number": 1,
            "title": "Repair the failing test through the filesystem tool.",
            "channel": "mcp",
            "capability": "filesystem.write_text",
            "decision_ms": 43,
            "execution_ms": 12,
            "verified": True,
        }
    ]
    assert "private content" not in json.dumps(steps)
    assert "hidden" not in json.dumps(steps)


def test_codex_baseline_token_cost_uses_official_enterprise_reference(tmp_path: Path) -> None:
    with TestClient(create_app(root=tmp_path, data=tmp_path / "runs")) as client:
        csrf = client.get("/api/bootstrap").json()["csrf"]
        headers = {"X-CUA-JEV-CSRF": csrf}
        run = client.post(
            "/api/baselines",
            headers=headers,
            json={
                "agent": "codex_computer_use",
                "task": "edge",
                "action_space": "hybrid",
                "success": True,
                "duration_ms": 1000,
                "actions": 1,
                "channels": {"script": 1},
            },
        ).json()
        path = f"/api/baselines/{run['id']}/usage"
        usage = {
            "model": "gpt-6-sol",
            "source": "local_session_token_count",
            "input_tokens": 1_000_000,
            "cached_input_tokens": 900_000,
            "output_tokens": 10_000,
        }
        assert client.post(path, json=usage).status_code == 403
        invalid = {**usage, "cached_input_tokens": 1_000_001}
        assert client.post(path, headers=headers, json=invalid).status_code == 400
        updated = client.post(path, headers=headers, json=usage)
        assert updated.status_code == 200
        assert updated.json()["metrics"]["standard_credits"] == pytest.approx(12.0)
        assert updated.json()["metrics"]["reference_cost_usd"] == pytest.approx(0.48)
        row = client.get("/api/benchmarks?task=edge").json()["rows"][0]
        assert row["median_input_tokens"] == 1_000_000
        assert row["median_reference_cost_usd"] == pytest.approx(0.48)
        assert row["cost_samples"] == 1


def test_jev_trace_usage_yields_api_cost_and_rejects_partial_usage(tmp_path: Path) -> None:
    manager = RunManager(tmp_path, tmp_path / "runs")
    events = [
        {
            "kind": "policy_exchange",
            "payload": {
                "response": {"model": "jev-1.13.0"},
                "usage": {"input_tokens": 20_000, "output_tokens": 100},
            },
        }
    ]
    metrics = manager._metrics({"events": events, "execution_profile": "adaptive"})
    assert metrics["input_tokens"] == 20_000
    assert metrics["model_cost_usd"] == pytest.approx(0.00084)
    events.append({"kind": "policy_exchange", "payload": {"response": {"model": "jev-1.13.0"}}})
    assert manager._metrics({"events": events, "execution_profile": "adaptive"})["model_cost_usd"] is None


def test_console_serves_brand_assets(tmp_path: Path) -> None:
    demos = tmp_path / "artifacts" / "demos"
    demos.mkdir(parents=True)
    (demos / "edge-hybrid.mp4").write_bytes(b"demo")
    with TestClient(create_app(root=tmp_path, data=tmp_path / "runs")) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "CUA-JEV" in page.text
        assert "How Jev powers computer use" in page.text
        assert "Four curated workflows and one experimental open-task pilot" in page.text
        assert "Structured state, no VLM" in page.text
        assert "arbitrary Windows tasks are not yet supported" in page.text
        assert "Adapter observes" in page.text
        assert "Jev chooses" in page.text
        assert "Runtime acts &amp; checks" in page.text
        assert "not general app automation" in page.text
        assert "Four Windows workflows cases" in page.text
        assert "Hybrid vs GUI Only" in page.text
        assert "Jev vs Codex Computer Use" in page.text
        assert "Model plans." in page.text
        assert "media/open-workspace-python.mp4" in page.text
        assert page.text.index('id="action-rows"') < page.text.index('id="agent-rows"')
        assert "43 ms" not in page.text
        assert ">REAL LAB<" not in page.text
        assert "qiushi-eagle" not in page.text
        assert client.get("/static/logo-mark.svg").status_code == 200
        for app in ("edge", "excel", "vscode", "explorer"):
            assert client.get(f"/static/icons/{app}.svg").status_code == 200
        assert client.get("/static/qiushi-eagle.svg").status_code == 404
        assert client.get("/api/bootstrap").json()["demos"]["edge"] == {"hybrid": "/demos/edge-hybrid.mp4"}
        assert client.get("/demos/edge-hybrid.mp4").content == b"demo"


def test_metrics_use_full_trace_while_detail_is_bounded(tmp_path: Path) -> None:
    manager = RunManager(tmp_path, tmp_path / "runs")
    run_id = "long-trace"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir()
    trace_base = run_dir / "trace.jsonl"
    events = [
        {"timestamp": index, "kind": "receipt", "payload": {"channel": "gui", "duration_ms": 1}}
        for index in range(115)
    ]
    events.extend(
        {"timestamp": 200 + index, "kind": "decision", "payload": {"latency_ms": 1}} for index in range(115)
    )
    events.append(
        {
            "timestamp": 400,
            "kind": "episode",
            "payload": {"duration_ms": 230, "status": "success"},
        }
    )
    trace_base.with_name("trace-edge.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )
    record = {
        "id": run_id,
        "task": "edge",
        "benchmark_version": "long-horizon-v2",
        "policy": "jev",
        "execution_profile": "adaptive",
        "status": "completed",
        "created_at": 1,
        "updated_at": 2,
        "trace_base": str(trace_base),
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")

    detail = manager.detail(run_id)

    assert len(detail["events"]) == 100
    assert detail["event_counts"]["decision"] == 115
    assert detail["metrics"]["actions"] == 115

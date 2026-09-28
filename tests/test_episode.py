import pytest

from cua_jev.episode import EpisodeConfig, EpisodeRunner, EpisodeStatus, Evaluation
from cua_jev.executors import ControlExecutor, FileSystemExecutor
from cua_jev.guard import ActionGuard
from cua_jev.models import ActionCandidate, Channel, Observation
from cua_jev.policy import RulePolicy
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime
from cua_jev.sandbox import FileOrganizationTask, sandbox_mcp_executor


def runner(tmp_path):
    def runtime_factory():
        executors = ExecutorRegistry()
        executors.register(Channel.API, FileSystemExecutor())
        executors.register(Channel.MCP, sandbox_mcp_executor())
        executors.register(Channel.CONTROL, ControlExecutor())
        return AgentRuntime(
            policy=RulePolicy(),
            guard=ActionGuard(allowed_roots=[tmp_path], allow_writes=True),
            executors=executors,
        )

    return EpisodeRunner(runtime_factory, EpisodeConfig(max_steps=20))


def test_episode_runs_closed_loop_to_independent_success(tmp_path):
    result = runner(tmp_path).run(FileOrganizationTask(tmp_path))
    assert result.status == EpisodeStatus.SUCCESS
    assert [step.decision.candidate_id for step in result.steps] == [
        "mcp_copy_compliance",
        "mcp_copy_customer-success",
        "mcp_copy_finance",
        "mcp_copy_inventory",
        "mcp_copy_marketing",
        "mcp_copy_operations",
        "mcp_copy_reliability",
        "mcp_copy_sales",
        "mcp_copy_security",
        "mcp_copy_support",
        "mcp_write_manifest",
        "mcp_write_release_note",
        "mcp_write_checksums",
        "mcp_write_publication_index",
        "mcp_write_audit_record",
        "done",
    ]
    assert result.channel_counts == {"mcp": 15, "control": 1}


def test_episode_reset_makes_repeated_runs_reproducible(tmp_path):
    task = FileOrganizationTask(tmp_path)
    assert runner(tmp_path).run(task).success
    assert runner(tmp_path).run(task).success


class StagnantTask:
    name = "stagnant"

    def reset(self):
        pass

    def observe(self, history):
        return Observation("wait", "wait", {"unchanged": True})

    def candidates(self, observation, history):
        return [ActionCandidate("wait", Channel.CONTROL, "control.wait", "wait")]

    def evaluate(self, observation, candidate, receipt, verification):
        return Evaluation(False, False, "still_waiting")


def test_episode_detects_repeated_unchanged_action(tmp_path):
    result = runner(tmp_path).run(StagnantTask())
    assert result.status == EpisodeStatus.STUCK
    assert len(result.steps) == 3


def test_composite_model_usage_includes_both_vision_adapters(tmp_path):
    from types import SimpleNamespace

    from cua_jev.trace import JsonlTrace

    trace = JsonlTrace()
    item = runner(tmp_path)
    original = item.runtime_factory

    def factory():
        runtime = original()
        runtime.trace = trace
        return runtime

    item.runtime_factory = factory
    environment = StagnantTask()
    environment.model_adapters = lambda: (
        ("planner", SimpleNamespace(model="text", usage_totals={"total_tokens": 10}, planning_requests=1)),
        ("browser_vision", SimpleNamespace(
            model="vision", usage_totals={"total_tokens": 20}, vision_requests=2)),
        ("desktop_vision", SimpleNamespace(
            model="vision", usage_totals={"total_tokens": 30}, vision_requests=3)),
    )
    item.run(environment)
    usage = {event["payload"]["role"]: event["payload"]
             for event in trace.events if event["kind"] == "model_usage"}
    assert usage["browser_vision"]["usage"]["total_tokens"] == 20
    assert usage["desktop_vision"]["requests"] == 3


def test_recording_cleanup_failure_prevents_success_trace(tmp_path):
    from cua_jev.trace import JsonlTrace

    trace = JsonlTrace()
    item = runner(tmp_path)
    original = item.runtime_factory

    def factory():
        runtime = original()
        runtime.trace = trace
        return runtime

    item.runtime_factory = factory
    environment = StagnantTask()

    def fail_close():
        raise RuntimeError("recording did not finalize")

    environment.close = fail_close
    result = item.run(environment)
    assert result.status == EpisodeStatus.ENVIRONMENT_ERROR
    assert "recording did not finalize" in result.reason
    episodes = [event for event in trace.events if event["kind"] == "episode"]
    assert len(episodes) == 1
    assert episodes[0]["payload"]["status"] == "environment_error"


def test_runtime_initialization_failure_still_closes_environment():
    closed = []
    environment = StagnantTask()
    environment.close = lambda: closed.append("environment")

    def factory():
        raise RuntimeError("backend initialization failed")

    result = EpisodeRunner(factory).run(environment)
    assert result.status == EpisodeStatus.ENVIRONMENT_ERROR
    assert "backend initialization failed" in result.reason
    assert closed == ["environment"]


@pytest.mark.parametrize("failure_stage", ["reset", "usage"])
def test_cancellation_and_usage_errors_always_release_resources(tmp_path, failure_stage):
    closed = []
    environment = StagnantTask()
    environment.close = lambda: closed.append("environment")
    item = runner(tmp_path)
    runtime = item.runtime_factory()
    runtime.policy.close = lambda: closed.append("policy")
    item.runtime_factory = lambda: runtime

    if failure_stage == "reset":
        def cancel():
            raise KeyboardInterrupt()

        environment.reset = cancel
        with pytest.raises(KeyboardInterrupt):
            item.run(environment)
    else:
        def fail_usage():
            raise RuntimeError("usage unavailable")

        environment.model_adapters = fail_usage
        result = item.run(environment)
        assert result.status == EpisodeStatus.ENVIRONMENT_ERROR
        assert "usage unavailable" in result.reason
        assert runtime.trace.events[-1]["kind"] == "episode"
    assert closed == ["policy", "environment"]


def test_policy_cleanup_failure_does_not_skip_environment_cleanup(tmp_path):
    closed = []
    environment = StagnantTask()
    environment.close = lambda: closed.append("environment")
    item = runner(tmp_path)
    runtime = item.runtime_factory()
    item.runtime_factory = lambda: runtime

    def fail_close():
        raise RuntimeError("policy close failed")

    runtime.policy.close = fail_close
    result = item.run(environment)
    assert closed == ["environment"]
    assert result.status == EpisodeStatus.ENVIRONMENT_ERROR
    assert "repetition limit" in result.reason
    assert "policy close failed" in result.reason


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_cancellation_during_policy_close_still_closes_environment(tmp_path, interruption):
    closed = []
    environment = StagnantTask()
    environment.close = lambda: closed.append("environment")
    item = runner(tmp_path)
    runtime = item.runtime_factory()
    item.runtime_factory = lambda: runtime

    def cancel_close():
        closed.append("policy")
        raise interruption("cancel during close")

    runtime.policy.close = cancel_close
    with pytest.raises(interruption, match="cancel during close"):
        item.run(environment)
    assert closed == ["policy", "environment"]
    assert not any(event["kind"] == "episode" for event in runtime.trace.events)


def test_cleanup_interruption_does_not_replace_original_cancellation(tmp_path):
    closed = []
    environment = StagnantTask()
    environment.close = lambda: closed.append("environment")
    item = runner(tmp_path)
    runtime = item.runtime_factory()
    item.runtime_factory = lambda: runtime

    def cancel_reset():
        raise KeyboardInterrupt("original cancellation")

    def cancel_close():
        raise SystemExit("second cancellation")

    environment.reset = cancel_reset
    runtime.policy.close = cancel_close
    with pytest.raises(KeyboardInterrupt, match="original cancellation"):
        item.run(environment)
    assert closed == ["environment"]


@pytest.mark.parametrize("stage", ["observe", "candidates", "policy"])
def test_expired_budget_never_dispatches_another_action(tmp_path, monkeypatch, stage):
    clock = [100.0]
    monkeypatch.setattr("cua_jev.episode.time.monotonic", lambda: clock[0])
    item = runner(tmp_path)
    item.config = EpisodeConfig(timeout_s=1)
    environment = StagnantTask()
    runtime = item.runtime_factory()
    item.runtime_factory = lambda: runtime
    owner, method = (runtime.policy, "choose") if stage == "policy" else (environment, stage)
    original = getattr(owner, method)

    def slow(*args):
        clock[0] += 2
        return original(*args)

    setattr(owner, method, slow)
    result = item.run(environment)
    assert result.status == EpisodeStatus.TIMEOUT
    assert not result.steps
    assert not any(event["kind"] == "receipt" for event in runtime.trace.events)


def test_wall_clock_adjustment_does_not_change_timeout_budget(tmp_path, monkeypatch):
    wall_clock = [100.0]
    monkeypatch.setattr("cua_jev.episode.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("cua_jev.episode.time.time", lambda: wall_clock[0])
    item = runner(tmp_path)
    environment = StagnantTask()
    original = environment.observe

    def observe(history):
        wall_clock[0] += 1000
        return original(history)

    environment.observe = observe
    assert item.run(environment).status == EpisodeStatus.STUCK


def test_late_action_keeps_receipt_and_reports_timeout(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("cua_jev.episode.time.monotonic", lambda: clock[0])
    item = runner(tmp_path)
    item.config = EpisodeConfig(timeout_s=1)
    environment = StagnantTask()

    def evaluate(*_args):
        clock[0] += 2
        return Evaluation(True, True, "completed late")

    environment.evaluate = evaluate
    result = item.run(environment)
    assert result.status == EpisodeStatus.TIMEOUT
    assert len(result.steps) == 1
    assert result.steps[0].receipt.success


@pytest.mark.parametrize("config", [
    {"timeout_s": float("nan")}, {"timeout_s": float("inf")}, {"timeout_s": 0},
    {"max_steps": 0}, {"unchanged_limit": -1}, {"repeated_action_limit": True},
])
def test_episode_rejects_invalid_budgets(config):
    with pytest.raises(ValueError):
        EpisodeConfig(**config)

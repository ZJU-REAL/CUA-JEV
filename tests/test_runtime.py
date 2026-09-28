import pytest

from cua_jev.executors import FileSystemExecutor
from cua_jev.guard import ActionGuard
from cua_jev.models import ActionCandidate, Channel, Observation
from cua_jev.policy import RulePolicy
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime, StepDeadlineExceeded


def test_runtime_reads_and_verifies_file(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text("hello", encoding="utf-8")
    candidate = ActionCandidate(
        "read",
        Channel.API,
        "filesystem.read_text",
        "Read report",
        {"path": str(path)},
        verifier="receipt.success",
    )
    registry = ExecutorRegistry()
    registry.register(Channel.API, FileSystemExecutor())
    runtime = AgentRuntime(
        policy=RulePolicy(), guard=ActionGuard(allowed_roots=[tmp_path]), executors=registry
    )
    result = runtime.step(Observation("read", "read report", {}), [candidate])
    assert result.receipt.output["text"] == "hello"
    assert result.verification.passed
    assert [event["kind"] for event in runtime.trace.events] == [
        "observation",
        "candidates",
        "decision",
        "commitment",
        "guard",
        "receipt",
        "verification",
    ]


def test_deadline_after_policy_prevents_filesystem_write(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("cua_jev.runtime.time.monotonic", lambda: clock[0])
    path = tmp_path / "late.txt"
    candidate = ActionCandidate(
        "write", Channel.API, "filesystem.write_text", "Write output",
        {"path": str(path), "text": "must not be written"},
    )
    policy = RulePolicy()
    choose = policy.choose

    def slow_choose(*args):
        clock[0] = 12.0
        return choose(*args)

    policy.choose = slow_choose
    registry = ExecutorRegistry()
    registry.register(Channel.API, FileSystemExecutor())
    runtime = AgentRuntime(
        policy=policy, guard=ActionGuard(allowed_roots=[tmp_path], allow_writes=True),
        executors=registry,
    )
    with pytest.raises(StepDeadlineExceeded, match="before action execution"):
        runtime.step(Observation("write", "write output", {}), [candidate], deadline=11.0)
    assert not path.exists()
    assert not any(event["kind"] == "receipt" for event in runtime.trace.events)

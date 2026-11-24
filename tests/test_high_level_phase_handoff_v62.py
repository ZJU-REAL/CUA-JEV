"""Zero-input typed high-level handoff through the original bounded H1 runner."""

import copy

import pytest
from test_goal_recheck_frame_UNEXECUTED import ProofBackend
from test_semantic_tools_v57 import current
from test_semantic_tools_v57 import observed as observed
from test_stage_program_contract_v60 import declaration
from test_stage_program_planning_preference_v61 import requested
from test_stage_program_runner_v60 import MemoryFields, events, two_fields
from test_stage_program_runner_v60 import run as run_program
from test_whole_goal_recheck import stage

from cua_jev import agent_runner
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.desktop_phase_handoff_v62 import HighLevelContextRequest
from cua_jev.trace import JsonlTrace

FORMAT = "semantic_tools_v62"
CURRENT_TOOLS = (
    "declare_native_current_action", "declare_semantic_current_action",
    "declare_visual_current_action", "declare_key_page_action", "declare_completion",
)


def context(reason="missing_future_semantics"):
    return HighLevelContextRequest(reason, "ab" * 32)


class RecordingBackend(ProofBackend):
    def __init__(self):
        super().__init__()
        self.compiles = []

    def compile(self, step, planned, current):
        self.compiles.append((step, planned, current))
        return super().compile(step, planned, current)


class PhasedPlanner:
    profile = "desktop"
    desktop_contract = "property_intent_v52"
    desktop_output_format = FORMAT

    def __init__(self, answers, *, after_response=None):
        self.answers = iter(answers)
        self.calls, self.requests, self.deadlines = 0, [], []
        self.after_response = after_response
        self.usage_totals = {"total_tokens": 0}

    def set_episode_deadline(self, value):
        self.deadlines.append(value)

    def plan(self, goal, snapshot, history, *, max_actions=6, **options):
        self.calls += 1
        self.usage_totals["total_tokens"] += 17
        self.requests.append({"limit": max_actions, "snapshot": snapshot,
                              "history": copy.deepcopy(history), "options": options})
        answer = next(self.answers)
        if self.after_response:
            self.after_response(self.calls)
        return answer


def episode(answers, *, backend=None, planner=None, **settings):
    backend = backend or RecordingBackend()
    planner = planner or PhasedPlanner(answers)
    trace = JsonlTrace()
    cfg = {"allow_actions": True, "mode": "hybrid", "plan_horizon": 6,
           "max_plans": 3, "max_failures": 3, "timeout_s": 30, **settings}
    result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
                         config=AgentConfig(**cfg)).run("Apply the requested Name and City values")
    return result, backend, planner, trace


@pytest.mark.parametrize("reason", [
    "unsupported_high_level_action", "missing_future_semantics", "transient_or_unexplained_state",
])
def test_typed_H6_context_requests_fresh_original_H1_then_returns_to_normal(reason):
    result, backend, planner, trace = episode([
        context(reason), stage("Name", "Ada", completion=False),
        stage("City", "Hangzhou", completion=False),
    ])
    assert [item["limit"] for item in planner.requests] == [6, 1, 6]
    assert planner.requests[1]["snapshot"] is not planner.requests[0]["snapshot"]
    assert planner.requests[1]["history"][-1]["recovery_required"] is True
    assert planner.requests[1]["history"][-1]["context_reason"] == reason
    assert backend.executed == ["Name", "City"] and len(backend.compiles) == 2
    assert result.steps == 2 and result.planner_calls == planner.calls == 3
    assert result.model_usage["total_tokens"] == 51
    assert result.plan_reuses == result.navigation_steps == result.multi_candidate_steps == 0
    handoff = events(trace, "agent_high_level_context_requested")
    assert len(handoff) == 1 and handoff[0]["positive_input_credit"] is False
    assert handoff[0]["logical_attempts"] == 0 and handoff[0]["planning_failures"] == 1
    assert not events(trace, "agent_recovery")
    assert [item["call"] for item in events(trace, "agent_plan")] == [2, 3]
    assert result.status == "max_plans" and not result.success


def test_repeated_context_cannot_escape_original_H1_or_original_failure_limit():
    result, backend, planner, trace = episode([context(), context(), context(), context()], max_plans=20)
    assert [item["limit"] for item in planner.requests] == [6, 1, 1]
    assert planner.calls == result.planner_calls == 3 and result.status == "planner_error"
    assert result.steps == result.plan_reuses == result.jev_calls == 0
    assert backend.environment_steps == 0 and backend.compiles == backend.executed == []
    assert len(events(trace, "agent_high_level_context_requested")) == 1
    assert len(events(trace, "agent_recovery")) == 2  # The two H1 violations are invalid responses.
    assert not events(trace, "agent_plan")


@pytest.mark.parametrize("settings", [
    {"mode": "vlm"}, {"plan_horizon": 1}, {"max_failures": 1},
])
def test_context_has_no_authority_in_baseline_H1_or_after_failure_boundary(settings):
    result, backend, _planner, trace = episode([context()], max_plans=1, **settings)
    assert result.planner_calls == 1 and result.steps == 0 and not result.success
    assert backend.compiles == backend.executed == [] and backend.environment_steps == 0
    if settings.get("max_failures") == 1:
        assert result.status == "planner_error"
        assert result.reason == "high-level context handoff budget exhausted"
    else:
        assert not events(trace, "agent_high_level_context_requested")


def test_context_respects_original_plan_count_without_free_recovery_request():
    result, backend, planner, _trace = episode([context()], max_plans=1)
    assert result.status == "max_plans" and result.planner_calls == planner.calls == 1
    assert not backend.compiles and not backend.executed and result.steps == 0


def test_context_does_not_renew_episode_deadline_or_old_plan_clock(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    planner = PhasedPlanner([context()], after_response=lambda _call: now.__setitem__(0, 41.0))
    result, backend, planner, trace = episode([], planner=planner)
    assert result.status == "timeout" and planner.calls == result.planner_calls == 1
    assert planner.deadlines == [40.0, None]  # Original runner clears its deadline on final cleanup.
    assert backend.environment_steps == 0 and backend.compiles == []
    handoff = events(trace, "agent_high_level_context_requested")[0]
    assert handoff["original_plan_at"] == 0.0 and handoff["original_plan_actions"] == 0
    assert handoff["selection_recovery_ticket_used"] is False


def test_context_preserves_exhausted_previous_plan_attempt_clock_before_new_H1(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    result, backend, planner, trace = episode([
        stage("Name", "Ada", completion=False), context(),
        stage("City", "Hangzhou", completion=False),
    ])
    assert [item["limit"] for item in planner.requests] == [6, 6, 1]
    handoff = events(trace, "agent_high_level_context_requested")[0]
    assert handoff["original_plan_at"] == 10.0 and handoff["original_plan_actions"] == 1
    assert handoff["positive_input_credit"] is False and handoff["logical_attempts"] == 0
    assert result.steps == 2 and backend.executed == ["Name", "City"]
    assert result.plan_reuses == 0


@pytest.mark.parametrize("delayed", [False, True])
def test_normal62_retains_original_program_and_fresh_stage0_admission(monkeypatch, delayed):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    backend = MemoryFields(reveal_after_first=True)
    planner = two_fields(backend, after_first_response=(
        (lambda: now.__setitem__(0, now[0] + 1.1)) if delayed else None
    ))
    planner.desktop_output_format = FORMAT
    result, trace = run_program(backend, planner)
    assert backend.executed == ["Name", "City"] and result.steps == 2
    assert planner.calls == result.planner_calls == 2
    admitted = events(trace, "agent_program_stage_admitted")
    assert len(admitted) == 1 and admitted[0]["stage_index"] == 0
    assert events(trace, "agent_program_completed")
    assert not events(trace, "agent_program_abandoned")
    assert not events(trace, "agent_high_level_context_requested")


@pytest.mark.parametrize("tool", CURRENT_TOOLS)
@pytest.mark.parametrize("formal", [False, True])
def test_62_strong_H1_full_actual_request_bytes_still_equal_public58(observed, tool, formal):
    raw = ({"label": "Observed completion", "conditions": [{"kind": "text_contains", "value": "Editor"}]}
           if tool == "declare_completion" else current(1, tool))
    options = {"whole_goal_recheck": True} if formal else {}
    expected, old, _ = requested(observed, raw, tool, fmt="semantic_tools_v58", limit=1,
                                 planning_options=options)
    actual, new, planner = requested(observed, raw, tool, fmt=FORMAT, limit=1, planning_options=options)
    assert actual == expected and new.content == old.content
    assert planner.calls == 1


@pytest.mark.parametrize("kind", ["context", "property", "program"])
def test_context_is_not_reported_as_a_stage_mapping_but_real_stages_still_are(observed, kind):
    if kind == "context":
        raw, tool = {"reason": "missing_future_semantics"}, "declare_current_context_needed"
    elif kind == "property":
        raw = {"label": "Requested setting", "scope": {"app": "Editor"}, "completion": "stage",
               "edit_operation": "fill", "edit_value": "requested literal", "numeric_finalize_key": "none",
               "confirmation_label": "none"}
        tool = "declare_property_stage"
    else:
        raw, tool = declaration(), "declare_stage_program"
    plan, _request, planner = requested(observed, raw, tool, fmt=FORMAT)
    assert isinstance(plan, HighLevelContextRequest) is (kind == "context")
    assert planner.last_response_transport["stage_mapping_applied"] is (kind != "context")
    assert planner.last_request_provenance["stage_mapping_applied"] is (kind != "context")

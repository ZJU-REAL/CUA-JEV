"Regression tests for test program aware reuse integration v60."

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from test_deferred_property_confirmation import current_dialog
from test_property_preparation import Backend as NativeSnapshotBackend
from test_property_preparation import DialogEnvironment, field
from test_property_preparation import initial as initial
from test_semantic_tools_v57 import packet
from test_semantic_tools_v57 import planner as real_planner

from cua_jev import agent_runner
from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentControl
from cua_jev.models import Decision
from cua_jev.trace import JsonlTrace

FORMAT = "semantic_tools_v60"
PROGRAM_TOOL = "declare_stage_program"
GOAL = "Set and confirm the requested preference, then fill the newly available City field."
PREFERENCES = {
    "name": "Preferences",
    "role": "dialog",
    "app": "Editor",
    "window": "Preferences",
}
HELP = {"name": "Help", "role": "button", "app": "Editor", "window": "Main"}
CITY = {"name": "City", "role": "textbox", "app": "Editor", "window": "Main"}


def predicate(kind, target, value=""):
    return {"kind": kind, "target": target, "value": value}


def program_declaration():
    # Future arguments deliberately use a natural, unsorted JSON member order.
    # Real codec ordered mapped hashes must not be confused with canonical source
    # declaration hashes by the offline consumer.
    return {
        "label": "Open, edit and confirm, then continue in the resulting form",
        "stages": [
            {
                "id": "reach-dialog",
                "tool": "declare_navigation_stage",
                "entry_conditions": [predicate("exists", HELP)],
                "arguments": {
                    "label": "Reach the requested preference dialog",
                    "scope": {"app": "Editor"},
                    "arrival": [predicate("exists", PREFERENCES)],
                    "completion": "stage",
                },
            },
            {
                "id": "edit-and-confirm",
                "tool": "declare_property_stage",
                "entry_conditions": [predicate("exists", PREFERENCES)],
                "arguments": {
                    "label": "Set and explicitly confirm the requested preference",
                    "scope": {"app": "Editor", "window": "Preferences"},
                    "edit_operation": "fill",
                    "edit_value": "50",
                    "numeric_finalize_key": "none",
                    "confirmation_label": "Confirm the requested property",
                    "completion": "stage",
                },
            },
            {
                "id": "new-form-field",
                "tool": "declare_field_edits_stage",
                "entry_conditions": [predicate("exists", CITY)],
                "arguments": {
                    "label": "Fill the field made available by the confirmed transition",
                    "completion": "stage",
                    "edits": [
                        {"operation": "fill", "target": CITY, "value": "requested-City"},
                    ],
                },
            },
        ],
    }


class DialogThenFieldEnvironment(DialogEnvironment):
    """Deterministic native-snapshot SDK fixture, never an OSWorld VM.

    Original DialogEnvironment handles field clicks/typing and its actual OK
    transition. We only model opening that dialog and the next observed form.
    The backend still compiles, binds, preflights, inputs and verifies normally.
    """

    def __init__(self, first):
        super().__init__(first)
        self.first, self.opened, self.focus = first, False, ""

    def step(self, action, pause):
        if not self.opened and action["action_type"] == "CLICK":
            self.actions.append(action)
            self.opened = True
            self.snapshot = current_dialog(self.first, value="80")
            return self.snapshot, 0, False, {}
        was_closed = self.closed
        super().step(action, pause)
        if not was_closed and self.closed:
            self.snapshot = replace(
                self.snapshot,
                title="Main",
                controls=(
                    AgentControl(
                        "after-main",
                        "Main",
                        "container",
                        role="frame",
                        app="Editor",
                        window="Main",
                        active=True,
                        box=(0, 0, 1000, 1000),
                    ),
                    field("after-city", "City", value="", box=(100, 100, 400, 200)),
                ),
            )
        return self.snapshot, 0, False, {}


class RecordingNativeBackend(NativeSnapshotBackend):
    """Keep original native compiler/guards/receipt verification unmodified."""

    def __init__(self, environment):
        super().__init__(environment, pause=0)
        self.observed_frames, self.compiled = [], []

    def observe(self):
        snapshot = super().observe()
        self.observed_frames.append(snapshot)
        return snapshot

    def compile(self, step, planned, current):
        self.compiled.append((step, planned, current))
        return super().compile(step, planned, current)


class RecordingRealPlanner(MultimodalPlanner):
    """Record arguments around the real implementation, never replace parsing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_frames, self.initial_compilation_frames, self.activation_frames = [], [], []

    def plan(self, goal, snapshot, history, **kwargs):
        self.original_frames.append(snapshot)
        return super().plan(goal, snapshot, history, **kwargs)

    def activate_program_stage(self, program, index, snapshot, *, remaining_actions):
        before = self.calls
        result = super().activate_program_stage(
            program,
            index,
            snapshot,
            remaining_actions=remaining_actions,
        )
        assert self.calls == before
        if index == 0:
            # plan() compiles the first stage against the uploaded request frame.
            # This is distinct from the runner activating a later live stage.
            self.initial_compilation_frames.append((index, snapshot, remaining_actions))
        else:
            self.activation_frames.append((index, snapshot, remaining_actions))
        return result


class PreferDeclaredNativeAction:
    def choose(self, observation, candidates):
        if observation.state.get("explicit_property_confirmation") is True:
            selected = next(item for item in candidates if item.description.startswith("click OK "))
        else:
            selected = next((item for item in candidates if item.capability == "agent.osworld.fill"), None)
            if selected is None:
                selected = next(item for item in candidates if item.description.startswith("click Help "))
        return Decision(
            observation.observation_id,
            selected.id,
            {item.id: float(item is selected) for item in candidates},
            1.0,
            "offline-integration-policy",
            0,
        )


def payloads(trace, kind):
    return [event["payload"] for event in trace.events if event["kind"] == kind]


def public_auditor():
    # Only the promoted public consumer is eligible. Never load a .txt draft or
    # a historical runs artifact as a fallback; a missing promotion is a failure.
    source = Path(__file__).resolve().parents[1] / "scripts/analyze_stage_program_reuse_v60.py"
    assert source.is_file()
    spec = importlib.util.spec_from_file_location("stage_program_reuse_public_auditor", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def real_episode(first, monkeypatch, *, invalidate_first_admission=False):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    environment = DialogThenFieldEnvironment(first)
    backend, requests = RecordingNativeBackend(environment), []

    def answer(request):
        assert request.url.host == "offline.invalid"
        assert request.url.path == "/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        assert len(requests) <= 2, "No hidden planning retry is permitted in this integration fixture"
        if len(requests) == 1:
            # The declaration is parsed against its actual uploaded frame. The
            # runner must subsequently admit stage0 on a truly current full read.
            now[0] += 1.1
            if invalidate_first_admission:
                environment.snapshot = replace(environment.snapshot, controls=(first.controls[0],))
            return httpx.Response(200, json=packet(program_declaration(), PROGRAM_TOOL, content=""))
        conditions = (
            [{"kind": "window_contains", "value": "Main"}]
            if invalidate_first_admission
            else [
                predicate("absent", PREFERENCES),
                predicate("value_equals", CITY, "requested-City"),
            ]
        )
        return httpx.Response(
            200,
            json=packet(
                {"label": "Fresh observed whole-goal completion", "conditions": conditions},
                "declare_completion",
                content="",
            ),
        )

    trace = JsonlTrace(run_id="offline-real-stage-program-integration")
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = real_planner(client, fmt=FORMAT, cls=RecordingRealPlanner)
        result = AgentRunner(
            backend,
            planner,
            policy=PreferDeclaredNativeAction(),
            trace=trace,
            config=AgentConfig(
                allow_actions=True,
                plan_horizon=6,
                max_plans=2,
                max_steps=8,
                max_failures=1,
                timeout_s=420,
                plan_ttl_s=300,
                singleton_fast_path=False,
                whole_goal_recheck=True,
                inband_whole_goal_recheck=False,
                reuse_verified_post_snapshot=False,
            ),
        ).run(GOAL)
    return result, backend, environment, planner, requests, trace


def completed_integration(first, monkeypatch):
    result, backend, environment, planner, requests, trace = real_episode(first, monkeypatch)
    assert result.steps == 4 and result.planner_calls == planner.calls == len(requests) == 2
    assert len(planner.initial_compilation_frames) == 1
    assert planner.initial_compilation_frames[0] == (0, planner.original_frames[0], 6)
    assert [index for index, _snapshot, _remaining in planner.activation_frames] == [1, 2]
    assert not any(control.window == "Preferences" for control in planner.original_frames[0].controls)
    assert not any(control.label == "City" for control in planner.original_frames[0].controls)
    assert any(control.window == "Preferences" for control in planner.activation_frames[0][1].controls)
    assert any(control.label == "City" for control in planner.activation_frames[1][1].controls)
    assert (
        environment.closed
        and next(control for control in environment.snapshot.controls if control.label == "City").value
        == "requested-City"
    )
    assert [
        action["parameters"]["text"] for action in environment.actions if action["action_type"] == "TYPING"
    ] == ["50", "requested-City"]
    assert all(step["verified"] is True for step in payloads(trace, "agent_step"))
    assert all(guard["approved"] is True for guard in payloads(trace, "guard"))
    assert len(payloads(trace, "receipt")) == len(payloads(trace, "guard")) == 4
    assert all(
        type(receipt["output"]["environment_steps"]) is int and receipt["output"]["environment_steps"] > 0
        for receipt in payloads(trace, "receipt")
    )
    completed = payloads(trace, "agent_program_stage_completed")
    assert [item["stage_index"] for item in completed] == [0, 1, 2]
    assert [item["plan_actions"] for item in completed] == [1, 3, 4]
    assert len({item["original_plan_at"] for item in completed}) == 1
    assert not result.success  # Local runtime completion never invents official task success.
    return result, backend, planner, requests, trace


def test_real_planner_runtime_trace_has_source_bound_cross_stage_positive_reuse(initial, monkeypatch):
    result, backend, planner, requests, trace = completed_integration(initial, monkeypatch)
    assert len(backend.observed_frames) > 1
    (admission,) = payloads(trace, "agent_program_stage_admitted")
    assert admission["source"] == "backend.observe" and admission["admitted"] is True
    assert admission["fresh_observe_calls"] >= 2
    assert admission["one_shot_entry_admission"] is True
    assert admission["initial_plan_not_recompiled"] is True
    assert admission["planned_request_binding_unchanged"] is True
    assert admission["original_action_preconditions_and_preflight_still_required"] is True
    assert "WHOLE USER GOAL RECHECK" in requests[1]["messages"][0]["content"]
    (declared,) = payloads(trace, "agent_program_declared")
    assert not any(plan["call"] == declared["origin_planner_call"] for plan in payloads(trace, "agent_plan"))
    property_activation = payloads(trace, "agent_program_stage_activated")[1]
    property_steps = property_activation["compiled_steps"]
    assert property_steps[1]["property_confirmation_of"] == property_steps[0]["id"]
    assert any(item["http_calls"] == 0 for item in payloads(trace, "agent_program_stage_activated")[1:])
    auditor = public_auditor()
    unchanged = copy.deepcopy(trace.events)
    report = auditor.audit_trace(trace.events)  # Real events, directly, with no inserted old agent_plan.
    assert trace.events == unchanged
    assert report["program_metric_status"] == "complete", report
    assert report["positive_input_program_continuations"] == 3
    assert report["cross_stage_positive_input_program_continuations"] == 2
    assert report["cross_stage_positive_input_program_continuations"] > 0
    assert report["runtime_join_issues"] == [] and report["unbound_runtime_receipts"] == []
    assert report["legacy_fixed_original"] == auditor.legacy_audit_trace(trace.events)
    assert report["legacy_fixed_metric_status_original"] == "partial_or_unknown"
    assert report["positive_input_plan_continuations"] is None
    assert report["legacy_fixed_applies_to_programs"] is False
    assert report["agent_episode_original"]["steps"] == result.steps
    assert report["acceptance_satisfied"] is False


def test_real_uploaded_true_entry_cannot_replace_false_fresh_stage0_admission(initial, monkeypatch):
    result, _backend, environment, planner, requests, trace = real_episode(
        initial,
        monkeypatch,
        invalidate_first_admission=True,
    )
    assert len(requests) == planner.calls == 2 and not environment.actions and result.steps == 0
    assert not payloads(trace, "agent_program_stage_admitted")
    assert not payloads(trace, "agent_program_stage_completed")
    assert payloads(trace, "agent_program_stage_admission_rejected")
    assert payloads(trace, "agent_program_abandoned")[0]["whole_program_retired"] is True
    assert len(planner.initial_compilation_frames) == 1
    assert planner.initial_compilation_frames[0] == (0, planner.original_frames[0], 6)
    assert planner.activation_frames == []
    assert not result.success
    report = public_auditor().audit_trace(trace.events)
    assert report["cross_stage_positive_input_program_continuations"] in (None, 0)
    assert report["acceptance_satisfied"] is False


@pytest.mark.parametrize("damage", ["guard_denied", "receipt_decision", "activation_frame"])
def test_real_trace_join_or_fresh_frame_damage_withholds_program_credit(initial, monkeypatch, damage):
    _result, _backend, _planner, _requests, trace = completed_integration(initial, monkeypatch)
    altered = copy.deepcopy(trace.events)
    if damage == "activation_frame":
        event = next(
            event
            for event in altered
            if event["kind"] == "agent_program_stage_activated" and event["payload"]["stage_index"] == 1
        )
        event["payload"]["snapshot_fingerprint"] = "0" * 64
    else:
        kind = "guard" if damage == "guard_denied" else "receipt"
        event = next(event for event in altered if event["kind"] == kind)
        event["payload"]["approved" if damage == "guard_denied" else "decision_id"] = (
            False if damage == "guard_denied" else "unbound-decision"
        )
    report = public_auditor().audit_trace(altered)
    assert report["program_metric_status"] == "unknown"
    assert report["positive_input_program_continuations"] is None
    assert report["cross_stage_positive_input_program_continuations"] is None
    assert report["acceptance_satisfied"] is False
    assert trace.events != altered

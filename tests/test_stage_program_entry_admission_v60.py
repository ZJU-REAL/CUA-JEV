'Regression tests for test stage program entry admission v60.'

from dataclasses import replace

import pytest
from test_property_preparation import Backend as NativeBackend
from test_property_preparation import Environment as NativeEnvironment
from test_property_preparation import initial as initial
from test_stage_program_contract_v60 import parse_original
from test_stage_program_runner_v60 import (
    MemoryFields,
    ProgramPlanner,
    events,
    facts,
    field_member,
    memory_parser,
    run,
)

from cua_jev import agent_runner
from cua_jev.agent_types import AgentCondition, AgentTarget


class GateFields(MemoryFields):
    def __init__(self, names=("Name", "City", "Gate"), **options):
        super().__init__(names, **options)
        self.values["Gate"] = "ready"
        self.hide_gate = self.partial = False
        self.compilations = 0

    def observe(self):
        snapshot = super().observe()
        if self.hide_gate or self.partial:
            snapshot = replace(snapshot, controls=tuple(control for control in snapshot.controls
                               if not (self.hide_gate and control.label == "Gate")),
                               a11y_complete=not self.partial)
            self._last_observed_snapshot = snapshot
            self.frames[-1] = snapshot
        return snapshot

    def compile(self, *args):
        self.compilations += 1
        return super().compile(*args)


def gated_raw(first_names=("Name",)):
    first = field_member("first", first_names)
    first["entry_conditions"] = [{"kind": "value_equals", "target": {
        "name": "Gate", "role": "textbox", "app": "Editor", "window": "Main"}, "value": "ready"}]
    return {"label": "Apply requested values after a one-shot observed gate", "stages": [
        first, field_member("second", ("City",)),
    ]}


@pytest.mark.parametrize("change", ["false", "target_missing", "incomplete"])
def test_delayed_stage0_entry_true_in_upload_but_not_true_in_fresh_frame_sends_no_input(monkeypatch, change):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    backend = GateFields()

    def delayed_response():
        now[0] += 1.1
        if change == "false":
            backend.values["Gate"] = "blocked"
        elif change == "target_missing":
            backend.hide_gate = True
        else:
            backend.partial = True

    planner = ProgramPlanner(gated_raw(), memory_parser,
        (facts("Name", "requested-Name"), facts("City", "requested-City")),
        after_first_response=delayed_response)
    _result, trace = run(backend, planner)
    # The contract already accepted entry=true in the actual uploaded frame.
    uploaded_gate = next(control for control in planner.requests[0]["snapshot"].controls
                         if control.label == "Gate")
    assert uploaded_gate.value == "ready"
    assert backend.compilations == 0 and backend.executed == []
    assert planner.activations == [] and planner.requests[1]["limit"] == 1
    assert not events(trace, "agent_program_stage_completed")
    assert events(trace, "agent_program_abandoned")
    assert not events(trace, "agent_program_stage_admitted")


def test_fresh_true_entry_is_admitted_once_and_does_not_become_a_persistent_guard(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    backend = GateFields(("Name", "Zone", "City", "Gate"))
    backend.after_execute = lambda: backend.values.__setitem__("Gate", "blocked")
    planner = ProgramPlanner(gated_raw(("Name", "Zone")), memory_parser,
        tuple(facts(name, "requested-" + name) for name in ("Name", "Zone", "City")),
        after_first_response=lambda: now.__setitem__(0, now[0] + 1.1))
    result, trace = run(backend, planner)
    assert backend.executed == ["Name", "Zone", "City"] and backend.values["Gate"] == "blocked"
    assert result.steps == 3 and result.planner_calls == 2
    admitted = [item for item in events(trace, "agent_program_stage_admitted") if item["stage_index"] == 0]
    assert len(admitted) == 1 and admitted[0]["plan_actions"] == 0
    assert admitted[0]["snapshot_fingerprint"] != planner.requests[0]["snapshot"].fingerprint() or (
        admitted[0]["source"] == "backend.observe"
    )  # An identical image/state may be genuinely reread; object/source proof matters.
    assert not events(trace, "agent_program_abandoned")


@pytest.mark.parametrize("preparation", ["current_ref", "visual_box"])
def test_fresh_admission_never_recompiles_or_rebinds_stage0_current_preparation(
    initial, monkeypatch, preparation,
):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    original = replace(initial, controls=tuple(replace(control, box=(100, 100, 400, 200))
                       if control.ref == "search" else control for control in initial.controls))
    environment = NativeEnvironment(original)
    backend = NativeBackend(environment, pause=0)
    prepare = ({"operation": "fill", "ref": "search", "value": "requested query"}
               if preparation == "current_ref" else
               {"operation": "visual_fill", "label": "Current search input", "box_space": "normalized_1000",
                "box": [100, 100, 400, 200], "value": "requested query"})
    raw = {"label": "Prepare current search then inspect requested settings", "stages": [
        {"id": "prepare-current", "tool": "declare_property_stage", "entry_conditions": [
            {"kind": "exists", "target": {
                "name": "Search", "role": "textbox", "app": "Editor", "window": "Main"}}],
         "arguments": {"label": "Set requested numeric setting", "scope": {"app": "Editor"},
            "prepare": prepare, "edit_operation": "set", "edit_value": "50", "numeric_finalize_key": "none",
            "confirmation_label": "none", "completion": "stage"}},
        {"id": "inspect", "tool": "declare_navigation_stage", "arguments": {
            "label": "Inspect requested setting", "scope": {"app": "Editor"},
            "arrival": [{"kind": "exists", "target": {"name": "Requested setting", "app": "Editor"}}],
            "completion": "stage"}},
    ]}
    parser_calls = []

    def parser(raw, snapshot, horizon):
        parser_calls.append(snapshot)
        return parse_original(raw, snapshot, horizon)

    def change_after_uploaded_frame():
        now[0] += 1.1
        environment.snapshot = replace(environment.snapshot,
            controls=tuple(replace(control, ref="new-search", box=(500, 300, 800, 400))
                           if control.ref == "search" else control
                           for control in environment.snapshot.controls),
            image=replace(environment.snapshot.image,
                          jpeg=environment.snapshot.image.jpeg + b"changed-frame"))

    planner = ProgramPlanner(raw, parser,
        (AgentCondition("exists", AgentTarget(name="Requested setting", app="Editor")),),
        after_first_response=change_after_uploaded_frame)
    _result, trace = run(backend, planner)
    assert len(parser_calls) == 1 and parser_calls[0] is planner.requests[0]["snapshot"]
    assert environment.actions == [] and planner.activations == []
    assert events(trace, "agent_program_abandoned")
    assert planner.requests[1]["limit"] == 1

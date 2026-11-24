"""Current-observation retrieval explains unknowns without authorizing input."""

import copy
import hashlib
import io
import json
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest
from PIL import Image

import cua_jev.agent_planner as planner_module
from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_runner import _postcondition_feedback
from cua_jev.agent_types import AgentCondition, AgentControl, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.desktop_semantics import condition_evidence, executable_controls, matching_controls
from cua_jev.desktop_target_diagnostics import observed_target_diagnostics
from cua_jev.osworld_backend import OSWorldBackend

FIXTURE = json.loads((Path(__file__).parent / "fixtures/osworld_predicate_diagnostics.json").read_text(
    encoding="utf-8",
))


def control(**changes):
    return replace(AgentControl("b", "Help", "button", role="button", app="Reader", window="Main",
                                box=(100, 100, 250, 150)), **changes)


def snapshot(*controls, complete=True):
    return AgentSnapshot("osworld://desktop", "Main", "private page text", controls or (control(),),
                         a11y_complete=complete)


def history(target=None, *, kind="exists", outcome="unknown", reason="target_missing"):
    target = target or AgentTarget(name="Help", role="menuitem", app="Reader")
    # Use the real runner sanitizer; comparison values never enter this history.
    condition = AgentCondition(kind, target, "private comparison value")
    step = AgentStep("open", "click", target=target, postconditions=(condition,))
    details = {"postcondition_evidence": [{"kind": kind, "target": asdict(target),
                                          "outcome": outcome, "reason": reason}]}
    return [{"action": "Open requested panel", "verified": False, "recovery_required": True,
             "operation": "click", "target": asdict(target),
             "postconditions": _postcondition_feedback(step, details)}]


class SavedDesktop:
    def __init__(self, xml):
        self.xml = xml
        self.inputs = 0
        image = io.BytesIO()
        Image.new("RGB", tuple(FIXTURE["screen_size"]), "white").save(image, format="PNG")
        self.png = image.getvalue()

    def _get_obs(self):
        return {"screenshot": self.png, "accessibility_tree": self.xml}

    def step(self, *args, **kwargs):
        self.inputs += 1
        raise AssertionError("Diagnostic replays must not send input")


@pytest.mark.parametrize("case", ["help", "troubleshooting", "caption"])
def test_recorded_controls_explain_wrong_role_or_missing_capability_without_changing_truth(case):
    record = FIXTURE["cases"][case]
    assert hashlib.sha256(record["xml"].encode()).hexdigest() == record["fixture_xml_sha256"]
    desktop = SavedDesktop(record["xml"])
    backend = OSWorldBackend(desktop, pause=0)
    try:
        current = backend.observe()
        assert current.a11y_complete
        target = AgentTarget(name=record["name"], role="" if case == "caption" else "menuitem",
                             app="gjs" if case == "caption" else "Thunderbird")
        kind = "ready_to_fill" if case == "caption" else "exists"
        condition = AgentCondition(kind, target, "private value" if case == "caption" else "")
        evidence = condition_evidence(condition, current)
        assert evidence == {"outcome": "unknown", "reason": (
            "unsupported_state" if case == "caption" else "target_missing"
        )}
        previous = history(target, kind=kind, **evidence)
        before = copy.deepcopy(previous)
        diagnostic = observed_target_diagnostics(current, previous)
        item, = diagnostic["predicates"]
        assert item["match_count"] == 1 and item["truncated"] is False
        observed, = item["observed_controls"]
        assert observed["role"] == ("label" if case == "caption" else "button")
        assert observed["operations"] == ([] if case == "caption" else ["click"])
        assert "value_equals" not in observed["condition_kinds"]
        assert previous == before
        assert condition_evidence(condition, current) == evidence
        if case != "caption":
            assert not executable_controls(current, "click", target)
            explicit = replace(target, role=observed["role"])
            step = AgentStep("new", "click", target=explicit)
            assert len(backend.compile(step, current, current)) == 1
        else:
            assert not executable_controls(current, "fill", target)
        assert desktop.inputs == 0
    finally:
        backend.close()


def test_preserves_exact_name_and_every_context_qualifier():
    target = AgentTarget(name="Help", role="menuitem", app="Reader", window="Main", ancestor="Tools")
    wanted = control(ancestors=("section: Tools",))
    current = snapshot(wanted, replace(wanted, ref="app", app="Other"),
                       replace(wanted, ref="window", window="Other"),
                       replace(wanted, ref="ancestor", ancestors=("section: Other",)),
                       replace(wanted, ref="name", label="Help Ctrl+H"))
    item, = observed_target_diagnostics(current, history(target))["predicates"]
    assert item["historical_predicate"]["target"] == asdict(target)
    assert item["match_count"] == 1
    assert item["observed_controls"][0]["ref"] == wanted.ref
    assert item["observed_controls"][0]["matches_requested_ancestor"] is True
    assert not matching_controls(current, target)


@pytest.mark.parametrize("target", [
    AgentTarget(role="menuitem", app="Reader"),
    AgentTarget(name="Help", role="menuitem"),
    AgentTarget(name="Help", role="menuitem", app="Reader", match="contains"),
    AgentTarget(name="Help...", role="menuitem", app="Reader"),
    AgentTarget(name="Help", role="menuitem", app="Reader", window="Other"),
    AgentTarget(name="Help", role="menuitem", app="\u200b"),
    AgentTarget(name="Help", role="menuitem", app="Reader", ancestor="\u200b"),
    AgentTarget(name="x" * 241, role="menuitem", app="Reader"),
])
def test_no_fuzzy_name_scope_expansion_or_oversized_query(target):
    assert observed_target_diagnostics(snapshot(), history(target)) is None


@pytest.mark.parametrize("outcome,reason", [
    ("true", "condition_met"), ("false", "state_mismatch"),
    ("unknown", "ambiguous_target"), ("unknown", "incomplete_observation"),
    ("unknown", "invalid_condition"), ("unknown", "made_up_reason"),
])
def test_only_typed_missing_or_unsupported_unknowns_are_eligible(outcome, reason):
    assert observed_target_diagnostics(snapshot(), history(outcome=outcome, reason=reason)) is None


def test_incomplete_snapshot_never_claims_a_complete_retrieval():
    assert observed_target_diagnostics(snapshot(complete=False), history()) is None


@pytest.mark.parametrize("suffix", [
    [{"action": "New success", "verified": True}],
    [{"action": "New visual failure", "verified": False}],
    [{"action": "New semantic failure", "verified": False, "operation": "click", "postconditions": []}],
    [{"feedback": "Planning failed"}] * 8,
])
def test_new_action_or_history_window_ends_previous_diagnostic(suffix):
    assert observed_target_diagnostics(snapshot(), history() + suffix) is None


def test_planning_failure_keeps_latest_action_context_but_retrieves_only_current_controls():
    previous = history() + [{"feedback": "Planning failed", "error_type": "ValueError"}]
    assert observed_target_diagnostics(snapshot(control(label="Gone")), previous) is None
    result = observed_target_diagnostics(snapshot(control(ref="new", role="checkbox", input_type="checkbox")),
                                         previous)
    shown = result["predicates"][0]["observed_controls"][0]
    assert shown["ref"] == "new" and shown["role"] == "checkbox"
    assert "checked" in shown["condition_kinds"]


def test_limits_display_without_creating_uniqueness_or_mutating_private_state():
    controls = tuple(control(ref=f"c{i}", value="private value", edit_buffer="private buffer")
                     for i in range(7))
    current, previous = snapshot(*controls), history()
    previous[0]["postconditions"] *= 4
    before = copy.deepcopy(previous), current.to_dict()
    result = observed_target_diagnostics(current, previous)
    assert len(result["predicates"]) == 2
    for item in result["predicates"]:
        assert item["match_count"] == 7 and item["truncated"] is True
        assert len(item["observed_controls"]) == 4
    assert len(executable_controls(
        current, "click", AgentTarget(name="Help", role="button", app="Reader"),
    )) == 7
    serialized = json.dumps(result)
    assert all(secret not in serialized for secret in (
        "private value", "private buffer", "private comparison value", "private page text",
    ))
    assert not any('"' + field + '":' in serialized for field in ("value", "edit_buffer", "box", "text"))
    assert (previous, current.to_dict()) == before


@pytest.mark.parametrize("input_type", ["password", "file", "hidden"])
def test_sensitive_matches_omit_whole_group_instead_of_claiming_filtered_uniqueness(input_type):
    assert observed_target_diagnostics(snapshot(control(), control(ref="private", input_type=input_type)),
                                       history()) is None


def test_disabled_control_is_observed_but_receives_no_operations():
    item = observed_target_diagnostics(snapshot(control(disabled=True)), history())["predicates"][0]
    assert item["match_count"] == 1
    assert item["observed_controls"][0]["operations"] == []
    assert not executable_controls(snapshot(control(disabled=True)), "click",
                                   AgentTarget(name="Help", role="button", app="Reader"))


@pytest.mark.parametrize("profile,max_actions", [("desktop", 1), ("desktop", 6), ("browser", 1)])
def test_actual_request_adds_only_optional_desktop_section_and_never_changes_shared_history(
    monkeypatch, profile, max_actions,
):
    current, previous = snapshot(), history()
    saved_history = copy.deepcopy(previous)
    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "summary": "No independent success claim", "done": True, "steps": [],
        })}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        planner = MultimodalPlanner("https://model.invalid/v1", "test", client=client, profile=profile)
        planner.plan("Inspect the current task", current, previous, max_actions=max_actions)
        monkeypatch.setattr(planner_module, "observed_target_diagnostics", lambda *_: None)
        planner.plan("Inspect the current task", current, previous, max_actions=max_actions)
    first, original = requests
    state = json.loads(first["messages"][1]["content"][0]["text"])
    if profile == "desktop":
        diagnostic = state.pop("current_predicate_diagnostics")
        assert diagnostic["source"] == "current_planning_snapshot"
        assert diagnostic["predicates"][0]["historical_predicate"]["outcome"] == "unknown"
        first["messages"][1]["content"][0]["text"] = json.dumps(
            state, ensure_ascii=False, separators=(",", ":"),
        )
    else:
        assert "current_predicate_diagnostics" not in state
    assert first == original
    assert state["recent_actions"] == saved_history == previous


def test_unrelated_history_keeps_desktop_request_field_absent():
    assert observed_target_diagnostics(snapshot(), [{"feedback": "initial"}]) is None

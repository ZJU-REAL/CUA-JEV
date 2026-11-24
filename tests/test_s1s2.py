import time

import pytest

from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.jev_decisions import JevDecisionError, JevMeter
from cua_jev.models import ActionCandidate, ActionReceipt, Channel, Verification
from cua_jev.s1s2_planner import S2Meter, S2Plan, Subgoal, parse_plan
from cua_jev.s1s2_runner import S1S2Config, S1S2Runner, compatible, lexical


def ctl(ref, label, tag="button", input_type="", value="", checked=False, box=None):
    return AgentControl(ref, label, tag, input_type=input_type, value=value, checked=checked, box=box)


def snap(*controls, text="page"):
    return AgentSnapshot("http://x/t", "T", text, tuple(controls))


def test_parse_plan_accepts_ref_for_unlabeled_and_records_identity():
    s = snap(ctl("e0", "", "input", "checkbox", box=(1, 1, 5, 5)), ctl("e1", "Submit"))
    plan = parse_plan(
        {"done": False, "subgoals": [{"op": "check", "target": "e0"}, {"op": "click", "target": "Submit"}]},
        s,
        horizon=6,
        grounded=False,
    )
    assert plan.subgoals[0].ref == "e0" and plan.subgoals[0].identity == s.controls[0].identity()
    assert plan.subgoals[1].target == "Submit" and plan.subgoals[1].ref == ""


def test_parse_plan_rejects_targetless_element_and_bad_values():
    s = snap(ctl("e0", "Name", "input", "text"))
    with pytest.raises(ValueError):
        parse_plan({"done": False, "subgoals": [{"op": "click", "target": ""}]}, s, horizon=6, grounded=False)
    with pytest.raises(ValueError):
        parse_plan(
            {"done": False, "subgoals": [{"op": "select", "target": "Name", "value": ""}]},
            s,
            horizon=6,
            grounded=False,
        )
    with pytest.raises(ValueError):
        parse_plan({"done": False, "subgoals": []}, s, horizon=6, grounded=False)


def test_grounded_box_only_on_first_subgoal():
    s = snap(ctl("e0", "Name", "input", "text"))
    plan = parse_plan(
        {"done": False, "subgoals": [{"op": "click", "target": "x", "box": [1, 2, 30, 40]}]},
        s,
        horizon=6,
        grounded=True,
    )
    assert plan.subgoals[0].box == (1, 2, 30, 40)
    plan = parse_plan(
        {"done": False, "subgoals": [{"op": "click", "target": "x", "box": [1, 2, 30, 40]}]},
        s,
        horizon=6,
        grounded=False,
    )
    assert plan.subgoals[0].box is None


def test_compatible_and_lexical():
    controls = [ctl("e0", "Name", "input", "text"), ctl("e1", "Agree", "input", "checkbox"), ctl("e2", "Go")]
    assert [c.ref for c in compatible(Subgoal("fill", "Name", "a"), controls)] == ["e0"]
    assert [c.ref for c in compatible(Subgoal("check", "Agree"), controls)] == ["e1"]
    assert lexical("Go", controls[2]) == 1.0 and lexical("Go", controls[0]) == 0.0


class FakeBackend:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.current = self.snapshots.pop(0)
        self.executed = []

    def observe(self):
        return self.current

    def compile(self, step, planned, current):
        return (
            ActionCandidate(
                f"{step.id}:dom",
                Channel.SCRIPT,
                "agent.browser." + step.operation,
                "x",
                {"route": "dom", "ref": step.ref, "box": step.box},
            ),
        )

    def execute(self, candidate, observation_id, decision_id):
        self.executed.append(candidate.arguments)
        if self.snapshots:
            self.current = self.snapshots.pop(0)
        now = time.time()
        return ActionReceipt(
            observation_id,
            decision_id,
            candidate.id,
            candidate.channel,
            candidate.capability,
            True,
            now,
            now,
            {},
            None,
        )

    def verify(self, candidate, receipt):
        return Verification(True, "agent.effect")


class FakePlanner:
    def __init__(self, plans):
        self.plans = list(plans)
        self.meter = S2Meter()
        self.failures = []

    def plan(self, goal, snapshot, history, *, failure=None, deadline=None):
        self.meter.calls += 1
        self.failures.append(failure)
        return self.plans.pop(0)


class FakeJev:
    def __init__(self, choice=None, error=None):
        self.choice, self.error = choice, error
        self.meter = JevMeter()

    def choose(self, state, instructions, criteria, image_jpeg=None):
        self.meter.calls += 1
        if self.error:
            raise JevDecisionError(self.error)
        assert "none" in criteria
        return self.choice, 0.9, {}

    def yes_no(self, *a, **k):
        self.meter.calls += 1
        return 0.9


def test_rule_unique_exact_executes_without_s1_call_and_succeeds():
    s0 = snap(ctl("e0", "Submit"), ctl("e1", "Cancel"))
    done = snap(ctl("e0", "Submit"), text="done")
    backend = FakeBackend([s0, done])
    planner = FakePlanner([S2Plan(False, (Subgoal("click", "Submit"),))])
    jev = FakeJev()
    result = S1S2Runner(backend, planner, jev, config=S1S2Config(s1="jev")).run(
        "g", lambda s: s.text == "done"
    )
    assert result["status"] == "success" and result["routes"] == {"rule_unique_exact": 1}
    assert jev.meter.calls == 0 and backend.executed[0]["ref"] == "e0"


def test_jev_chooses_among_ambiguous_candidates():
    s0 = snap(ctl("e0", "Save"), ctl("e1", "Save"), text="a")
    done = snap(ctl("e0", "Save"), text="done")
    backend = FakeBackend([s0, done])
    planner = FakePlanner([S2Plan(False, (Subgoal("click", "Save"),))])
    result = S1S2Runner(backend, planner, FakeJev(choice="e1"), config=S1S2Config(s1="jev")).run(
        "g", lambda s: s.text == "done"
    )
    assert result["status"] == "success" and backend.executed[0]["ref"] == "e1"
    assert result["routes"] == {"jev_choice_structured": 1}


def test_rule_mode_escalates_ambiguity_to_s2():
    s0 = snap(ctl("e0", "Save"), ctl("e1", "Save"), text="a")
    backend = FakeBackend([s0])
    planner = FakePlanner([S2Plan(False, (Subgoal("click", "Save"),)), S2Plan(True, ())])
    result = S1S2Runner(backend, planner, None, config=S1S2Config(s1="rule")).run("g", lambda s: False)
    assert result["escalations"] == {"rule_ambiguous": 1} and planner.failures[1]
    assert backend.executed == []


def test_keep_violation_escalates():
    name = ctl("e0", "Name", "input", "text", value="Ann")
    s0 = snap(name, ctl("e1", "Go"))
    changed = snap(ctl("e0", "Name", "input", "text", value="Bob"), ctl("e1", "Go"))
    backend = FakeBackend([s0, changed])
    planner = FakePlanner([S2Plan(False, (Subgoal("click", "Go"),), keep=("e0",)), S2Plan(True, ())])
    result = S1S2Runner(backend, planner, None, config=S1S2Config(s1="rule")).run("g", lambda s: False)
    assert result["escalations"] == {"keep_violated": 1}


def test_jev_error_escalates_without_input():
    s0 = snap(ctl("e0", "Save"), ctl("e1", "Save"))
    backend = FakeBackend([s0])
    planner = FakePlanner([S2Plan(False, (Subgoal("click", "Save"),)), S2Plan(True, ())])
    result = S1S2Runner(backend, planner, FakeJev(error="timeout"), config=S1S2Config(s1="jev")).run(
        "g", lambda s: False
    )
    assert result["escalations"] == {"jev_error_timeout": 1} and backend.executed == []


def test_stale_ref_is_not_rebound_to_a_different_control():
    from cua_jev.structured_browser import ScopedBrowserControl

    a = ScopedBrowserControl("e0", "", "input", input_type="checkbox", box=(0, 0, 5, 5))
    s0 = snap(a, ctl("e1", "Go"))
    moved = snap(
        ScopedBrowserControl("e0", "", "input", input_type="checkbox", box=(0, 9, 5, 14)), ctl("e1", "Go")
    )
    backend = FakeBackend([s0])
    backend.current = moved
    sub = Subgoal("check", "", ref="e0", identity=a.identity())
    planner = FakePlanner([S2Plan(False, (sub,)), S2Plan(True, ())])
    result = S1S2Runner(backend, planner, None, config=S1S2Config(s1="rule")).run("g", lambda s: False)
    assert result["escalations"] == {"stale_ref": 1} and backend.executed == []


def test_sole_compatible_field_is_not_used_for_a_different_target():
    s0 = snap(ctl("e0", "Username", "input", "text"), ctl("e1", "Login"))
    backend = FakeBackend([s0])
    planner = FakePlanner([S2Plan(False, (Subgoal("fill", "Password", "x"),)), S2Plan(True, ())])
    result = S1S2Runner(backend, planner, None, config=S1S2Config(s1="rule")).run("g", lambda s: False)
    assert backend.executed == [] and result["escalations"] == {"rule_no_match": 1}


def test_ref_for_labelled_control_becomes_text_target():
    s = snap(ctl("e0", "Search"), ctl("e1", "", "input", "checkbox"))
    plan = parse_plan(
        {
            "done": False,
            "subgoals": [
                {"op": "click", "target": "Search", "ref": "e0"},
                {"op": "check", "target": "", "ref": "e1"},
            ],
        },
        s,
        horizon=6,
        grounded=False,
    )
    assert (plan.subgoals[0].ref, plan.subgoals[0].target) == ("", "Search")
    assert plan.subgoals[1].ref == "e1"
    grounded = parse_plan(
        {"done": False, "subgoals": [{"op": "click", "target": "Search", "ref": "e0"}]},
        s,
        horizon=6,
        grounded=True,
    )
    assert grounded.subgoals[0].ref == "e0"


def test_without_evaluator_stops_on_done_and_returns_answer():
    s0 = snap(ctl("e0", "Go"))
    backend = FakeBackend([s0])
    planner = FakePlanner([S2Plan(True, (), answer="42")])
    result = S1S2Runner(backend, planner, None, config=S1S2Config(s1="rule")).run("g", None)
    assert result["status"] == "done" and result["final_answer"] == "42"


def test_parse_plan_reads_answer():
    plan = parse_plan({"done": True, "subgoals": [], "answer": "Paris"}, snap(), horizon=6, grounded=False)
    assert plan.done and plan.answer == "Paris"


class StaleOnceBackend(FakeBackend):
    def __init__(self, snapshots):
        super().__init__(snapshots)
        self.stale = True

    def execute(self, candidate, observation_id, decision_id):
        if self.stale:
            self.stale = False
            now = time.time()
            return ActionReceipt(
                observation_id,
                decision_id,
                candidate.id,
                candidate.channel,
                candidate.capability,
                False,
                now,
                now,
                {},
                "ValueError: stale browser snapshot; replan before execution",
            )
        return super().execute(candidate, observation_id, decision_id)

    def verify(self, candidate, receipt):
        return Verification(receipt.success, "agent.effect")


def test_stale_preflight_rebinds_locally_without_s2():
    s0 = snap(ctl("e0", "Go"))
    done = snap(ctl("e0", "Go"), text="done")
    backend = StaleOnceBackend([s0, done])
    planner = FakePlanner([S2Plan(False, (Subgoal("click", "Go"),))])
    result = S1S2Runner(backend, planner, None, config=S1S2Config(s1="rule")).run(
        "g", lambda s: s.text == "done"
    )
    assert result["status"] == "success" and planner.meter.calls == 1
    assert result["routes"]["stale_rebind"] == 1 and result["escalations"] == {}


def test_format_slips_are_normalized_not_rejected():
    s = snap(ctl("e0", "Search", "input", "text"))
    plan = parse_plan(
        {
            "done": False,
            "subgoals": [
                {"op": "scroll", "target": "", "value": ""},
                {"op": "wait", "value": "2000"},
                {"op": "fill", "target": "Search", "value": ""},
            ],
        },
        s,
        horizon=6,
        grounded=False,
    )
    assert [g.value for g in plan.subgoals] == ["down", "1000", ""]


def test_binding_miss_after_a_step_rebinds_once_before_escalating():
    s0 = snap(ctl("e0", "Menu"))
    opened = snap(ctl("e0", "Menu"), text="loading")
    rendered = snap(ctl("e0", "Menu"), ctl("e1", "Orders"))
    done = snap(ctl("e1", "Orders"), text="done")

    class Delayed(FakeBackend):
        def observe(self):
            if self.current is opened:
                self.current = rendered
                return opened
            return self.current

    backend = Delayed([s0, opened, done])
    planner = FakePlanner([S2Plan(False, (Subgoal("click", "Menu"), Subgoal("click", "Orders")))])
    result = S1S2Runner(backend, planner, None, config=S1S2Config(s1="rule")).run(
        "g", lambda s: s.text == "done"
    )
    assert result["status"] == "success" and planner.meter.calls == 1
    assert result["routes"]["rebind_after_wait"] == 1 and result["escalations"] == {}

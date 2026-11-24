"""An explicit fresh preparation can retain its unexecuted property stage and budgets."""

import copy
import json
import xml.etree.ElementTree as ET
from dataclasses import replace

import httpx
import pytest
from test_observed_input_geometry import COMPONENT, STATE, VISIBLE, form
from test_observed_input_geometry import png as png

from cua_jev.agent_planner import MultimodalPlanner, _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentPlan, AgentSnapshot, AgentStep
from cua_jev.models import ActionReceipt, Decision, Verification
from cua_jev.osworld_backend import OSWorldBackend


class Clock:
    now = 100.0

    def read(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("time.perf_counter", clock.read)
    monkeypatch.setattr("time.time", clock.read)
    monkeypatch.setattr("time.sleep", clock.advance)
    return clock


class Desktop:
    def __init__(self, png, clock, *, confirm=False, preparation_effect=True, navigate_first=False):
        self.png, self.clock = png, clock
        self.root, self.frame, self.parent, self.proxy = form()
        self.actions = []
        self.confirm, self.preparation_effect = confirm, preparation_effect
        self.navigate_first = navigate_first
        self.focus = self.proxy
        self.opened = self.closed = False
        self.delay_next_read = self.delay_next_input = 0
        self.dialog = None
        self.refresh()

    def refresh(self):
        self.snapshot = OSWorldBackend(object())._snapshot({
            "screenshot": self.png, "accessibility_tree": ET.tostring(self.root, encoding="unicode"),
        })

    def drift(self):
        # This unrelated current label changes freshness while native input,
        # selected owner, and parent rectangle remain exactly the same.
        ET.SubElement(self.frame, "label", {"name": "New unrelated status", **VISIBLE})
        self.refresh()

    def _get_obs(self):
        self.clock.advance(self.delay_next_read)
        self.delay_next_read = 0
        return self.snapshot

    def add_fields(self):
        owner = self.frame
        if self.confirm:
            self.dialog = ET.SubElement(self.frame, "dialog", {
                "name": "Preferences", **VISIBLE, STATE + "active": "true", STATE + "modal": "true",
                COMPONENT + "screencoord": "(600, 250)", COMPONENT + "size": "(800, 550)",
            })
            owner = self.dialog
        for name, value, y in (("Wrap column", "80", 400), ("Cache size", "20", 500)):
            entry = ET.SubElement(owner, "entry", {
                "name": name, **VISIBLE, STATE + "editable": "true", STATE + "focusable": "true",
                COMPONENT + "screencoord": f"(800, {y})", COMPONENT + "size": "(300, 40)",
            })
            entry.text = value
        if self.confirm:
            ET.SubElement(owner, "push-button", {
                "name": "OK", **VISIBLE, COMPONENT + "screencoord": "(1000, 700)",
                COMPONENT + "size": "(100, 40)",
            })

    def step(self, action, pause):
        self.actions.append(action)
        self.clock.advance(self.delay_next_input)
        self.delay_next_input = 0
        kind, parameters = action["action_type"], action["parameters"]
        if kind == "CLICK":
            y = parameters["y"]
            if self.opened and self.confirm and y >= 700:
                self.frame.remove(self.dialog)
                self.closed = True
            elif self.opened and self.navigate_first and y >= 600:
                self.navigate_first = False
                self.add_fields()
            elif self.opened:
                self.focus = next((node for node in self.root.iter("entry")
                                   if node.get("name") == ("Wrap column" if y < 460 else "Cache size")),
                                  self.proxy)
                for node in self.root.iter("entry"):
                    node.attrib.pop(STATE + "focused", None)
                self.focus.set(STATE + "focused", "true")
        elif kind == "TYPING" and self.preparation_effect:
            if not self.opened:
                self.opened = True
                if self.navigate_first:
                    ET.SubElement(self.frame, "push-button", {
                        "name": "Reveal fields", **VISIBLE, COMPONENT + "screencoord": "(800, 620)",
                        COMPONENT + "size": "(300, 40)",
                    })
                else:
                    self.add_fields()
            else:
                self.focus.text = parameters["text"]
        self.refresh()
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation if isinstance(observation, AgentSnapshot) else super()._snapshot(observation)


def root_data(*, prepare=True, confirm=False, goal="true"):
    condition = ({"kind": "window_contains", "value": "Settings"} if goal == "true" else {
        "kind": "value_equals", "target": {"name": "Wrap column" if goal == "false" else "Unknown field",
                                               "app": "Editor"}, "value": "81",
    })
    raw = {"intent": "pursue_property", "label": "Set the requested column property",
           "scope": {"app": "Editor"}, "edit": {"operation": "fill", "value": "50"},
           "completion": "goal", "goal_conditions": [condition]}
    if prepare:
        raw["prepare"] = {"operation": "visual_fill", "label": "Current visible input",
                          "box_space": "normalized_1000", "box": [302, 131, 820, 149],
                          "value": "word wrap column"}
    if confirm:
        raw["confirm"] = {"label": "Apply the requested change and close its form"}
    return raw


def root_plan(snapshot, *, prepare=True, confirm=False, goal="true"):
    raw = root_data(prepare=prepare, confirm=confirm, goal=goal)
    plan = parse_plan(raw, snapshot, max_actions=6, profile="desktop", allow_property_preparation=True,
                      allow_property_goal_conditions=True, allow_property_value_binding=True,
                      allow_observed_property_goal=True)
    _validate_desktop_execution_plan(plan, snapshot, 6)
    return plan


class Planner:
    profile = "desktop"
    desktop_contract = "property_intent_v33"

    def __init__(self, env, *, repair="token", goal="true", expiry=None):
        self.env, self.repair, self.goal, self.expiry = env, repair, goal, expiry
        self.calls = 0
        self.context = self.deadline = None
        self.offers, self.limits, self.deadlines = [], [], []
        self.original_plan = None

    def set_prepare_repair_context(self, context):
        self.context = copy.deepcopy(context)

    def set_episode_deadline(self, deadline):
        self.deadline = deadline
        self.deadlines.append(deadline)

    def plan(self, _goal, snapshot, _history, *, max_actions):
        self.calls += 1
        self.offers.append(copy.deepcopy(self.context))
        self.limits.append(max_actions)
        if self.calls == 1:
            self.original_plan = root_plan(snapshot, confirm=self.env.confirm, goal=self.goal)
            self.env.drift()
            return self.original_plan
        if self.calls == 2:
            if self.expiry == "request":
                self.env.clock.advance(6)
            elif self.expiry == "observe":
                self.env.delay_next_read = 6
            elif self.expiry == "partial-input":
                self.env.delay_next_input = 6
            token = ""
            if self.repair != "omit":
                assert self.context is not None
                token = "0" * 32 if self.repair == "wrong-token" else self.context["token"]
            value = "changed query" if self.repair == "changed-value" else "word wrap column"
            return AgentPlan("Repair the current input", (AgentStep(
                "current", "visual_fill", value=value, box=(302, 131, 820, 149), label="Fresh current input",
            ),), prepare_repair_token=token)
        if self.repair in {"wrong-token", "changed-value"} or not self.env.preparation_effect:
            return AgentPlan("End negative fixture without authorizing input", (), done=True)
        return root_plan(snapshot, prepare=False, confirm=self.env.confirm, goal=self.goal)


class Policy:
    def __init__(self, clock, *, slow=False):
        self.clock, self.slow = clock, slow
        self.calls = []

    def choose(self, observation, candidates):
        self.calls.append({"count": len(candidates), "controls": observation.state["controls"]})
        chosen = next((c for c in candidates if c.capability == "agent.osworld.fill"
                       and c.description.startswith("fill Wrap column ")), None)
        chosen = chosen or next((c for c in candidates if c.description.startswith("click Reveal fields ")),
                                None)
        chosen = chosen or next((c for c in candidates if c.description.startswith("click OK ")),
                                candidates[0])
        if self.slow:
            self.clock.advance(6)
            self.slow = False
        return Decision(observation.observation_id, chosen.id, {c.id: float(c is chosen) for c in candidates},
                        1, "offline-public-choice", 0)


def run_case(png, clock, *, repair="token", goal="true", expiry=None, confirm=False,
             preparation_effect=True, navigate_first=False, **settings):
    env = Desktop(png, clock, confirm=confirm, preparation_effect=preparation_effect,
                  navigate_first=navigate_first)
    planner = Planner(env, repair=repair, goal=goal, expiry=expiry)
    policy = Policy(clock, slow=expiry == "policy")
    events = []
    config = {"allow_actions": True, "max_plans": 2, "max_failures": 4,
              "max_steps": 10, "plan_horizon": 6, "timeout_s": 120, "plan_ttl_s": 60}
    config.update(settings)
    if expiry is not None:
        config["plan_ttl_s"] = 5
    if expiry == "policy":
        config["singleton_fast_path"] = False
    result = AgentRunner(Backend(env, pause=0), planner, policy=policy, progress=events.append,
                         config=AgentConfig(**config)).run("Set and persist the requested property")
    return result, env, planner, policy, events


@pytest.mark.parametrize("confirm", [False, True])
def test_two_model_calls_repair_before_input_then_reuse_a_real_multi_candidate_property(png, clock, confirm):
    result, env, planner, policy, events = run_case(png, clock, confirm=confirm)
    assert result.planner_calls == planner.calls == 2 and planner.limits == [6, 1]
    assert planner.offers[0] is None and planner.offers[1]["token"]
    assert result.property_edits == 1 and result.plan_reuses == (2 if confirm else 1)
    assert result.plan_completion_claims == 1 and not result.success
    assert any(call["count"] > 1 for call in policy.calls)
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == [
        "word wrap column", "50",
    ]
    repairs = [e for e in events if e["event"] == "agent_prepare_repair"]
    assert [e["status"] for e in repairs] == ["offered", "prefix_replaced_before_input"]
    assert all(e["old_prepare_executed"] is False for e in repairs)
    assert not planner.context and planner.deadline is None
    if confirm:
        assert env.closed and any(e["event"] == "agent_property_commit" for e in events)


def test_actual_multimodal_planner_and_mock_transport_parse_the_explicit_repair_token(png, clock):
    env, requests, events = Desktop(png, clock), [], []

    def reply(request):
        body = json.loads(request.content)
        state = json.loads(body["messages"][1]["content"][0]["text"])
        requests.append(state)
        if len(requests) == 1:
            raw = root_data()
            env.drift()
        else:
            raw = {"summary": "Replace the unexecuted preparation", "done": False,
                   "prepare_repair": state["prepare_repair_offer"]["token"], "steps": [{
                       "id": "fresh", "operation": "visual_fill", "label": "Current visible input",
                       "box_space": "normalized_1000", "box": [302, 131, 820, 149],
                       "value": "word wrap column",
                   }]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    policy = Policy(clock)
    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        planner = MultimodalPlanner("https://offline.invalid", "fixture", client=client,
                                    profile="desktop", desktop_contract="property_intent_v33")
        result = AgentRunner(Backend(env, pause=0), planner, policy=policy, progress=events.append,
                             config=AgentConfig(allow_actions=True, max_plans=2, max_failures=4,
                                                plan_ttl_s=60, timeout_s=120)).run("Set requested property")
    assert planner.calls == result.planner_calls == len(requests) == 2
    assert "prepare_repair_offer" not in requests[0]
    assert requests[1]["prepare_repair_offer"]["retained_stage"]["whole_goal_conditions"]
    assert result.property_edits == result.plan_reuses == result.plan_completion_claims == 1
    assert any(call["count"] > 1 for call in policy.calls)
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == [
        "word wrap column", "50",
    ]


def test_omitting_token_keeps_the_ordinary_three_call_recovery_path(png, clock):
    result, env, planner, _policy, events = run_case(png, clock, repair="omit", max_plans=3)
    assert result.planner_calls == 3 and planner.limits == [6, 1, 6]
    assert result.property_edits == 1 and result.plan_reuses == 0
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == [
        "word wrap column", "50",
    ]
    assert not any(e.get("status") == "prefix_replaced_before_input" for e in events)


@pytest.mark.parametrize("goal", ["false", "unknown"])
def test_repair_preserves_false_or_unknown_whole_goal_conditions_through_confirmation(png, clock, goal):
    result, env, planner, _policy, events = run_case(png, clock, goal=goal, confirm=True)
    assert env.closed and result.property_edits == 1
    assert result.plan_completion_claims == 0 and not result.success
    assert planner.offers[1]["retained_stage"]["whole_goal_conditions"]
    assert any(e["event"] == "agent_property_commit" for e in events)


@pytest.mark.parametrize("repair", ["wrong-token", "changed-value"])
def test_rejected_assigned_repair_never_survives_validation_into_the_next_loop(png, clock, repair):
    result, env, planner, _policy, events = run_case(png, clock, repair=repair, max_plans=3)
    assert planner.calls == 3 and planner.offers[1] and planner.offers[2] is None
    assert env.actions == [] and result.steps == 0
    assert not any(e.get("status") == "prefix_replaced_before_input" for e in events)
    assert not planner.context


@pytest.mark.parametrize("expiry", ["request", "observe", "policy", "partial-input"])
def test_original_plan_ttl_bounds_model_wait_fresh_observation_policy_and_each_input(png, clock, expiry):
    result, env, planner, _policy, _events = run_case(png, clock, expiry=expiry)
    assert planner.calls == 2 and 105.0 in planner.deadlines
    assert planner.offers[1]["remaining_time_s"] == 5
    assert result.property_edits == 0 and not result.plan_completion_claims
    assert [a["action_type"] for a in env.actions] == (["CLICK"] if expiry == "partial-input" else [])
    assert not planner.context and planner.deadline is None


def test_repair_does_not_refresh_the_logical_action_budget(png, clock):
    result, env, _planner, _policy, _events = run_case(png, clock, max_steps=1)
    assert result.status == "max_steps" and result.steps == 1 and result.planner_calls == 2
    assert result.property_edits == 0
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == [
        "word wrap column",
    ]


def test_a_repair_offer_does_not_add_another_model_call_to_an_exhausted_budget(png, clock):
    result, env, planner, _policy, _events = run_case(png, clock, max_plans=1)
    assert result.status == "max_plans" and planner.calls == 1 and env.actions == []


def test_retained_horizon_counts_preparation_and_navigation_before_the_final_edit(png, clock):
    result, env, planner, _policy, _events = run_case(png, clock, plan_horizon=2, navigate_first=True)
    assert result.status == "max_plans" and planner.calls == 2
    assert result.steps == 2 and result.property_edits == 0
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == [
        "word wrap column",
    ]


def test_failed_repaired_preparation_never_reoffers_the_same_retained_frame(png, clock):
    result, env, planner, _policy, events = run_case(png, clock, preparation_effect=False, max_plans=3)
    assert planner.calls == 3 and planner.offers[1] and planner.offers[2] is None
    assert not result.property_edits and not result.plan_completion_claims
    assert [e["status"] for e in events if e["event"] == "agent_prepare_repair"] == [
        "offered", "prefix_replaced_before_input",
    ]
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == [
        "word wrap column",
    ]


def test_owner_metadata_changed_during_model_request_cancels_repair_before_any_input(png, clock):
    env, events = Desktop(png, clock), []

    class ChangedMetadataPlanner(Planner):
        def plan(self, *args, **kwargs):
            response = super().plan(*args, **kwargs)
            if self.calls == 2:
                before = self.env.snapshot
                context, = before.input_geometry
                self.env.snapshot = replace(before, input_geometry=(replace(
                    context, ancestor_class="A different actual anonymous input container",
                ),))
                # Ordinary freshness surfaces deliberately remain identical;
                # this change must be caught by the retained source binding.
                assert self.env.snapshot.to_dict() == before.to_dict()
                assert self.env.snapshot.fingerprint() == before.fingerprint()
                assert self.env.snapshot.scope_key() == before.scope_key()
                assert self.env.snapshot.image is before.image
                assert self.env.snapshot.controls == before.controls
            return response

    planner = ChangedMetadataPlanner(env)
    result = AgentRunner(Backend(env, pause=0), planner, policy=Policy(clock), progress=events.append,
                         config=AgentConfig(allow_actions=True, max_plans=2, max_failures=4,
                                            plan_ttl_s=60, timeout_s=120)).run("Set requested property")
    assert planner.calls == 2 and planner.offers[1]
    assert env.actions == [] and not env.opened
    assert result.steps == result.property_edits == result.plan_reuses == result.plan_completion_claims == 0
    cancelled, = [event for event in events if event["event"] == "agent_prepare_repair"
                  and event["status"] == "cancelled_before_compilation"]
    assert cancelled["input_sent"] is False
    assert cancelled["reason"] == "original_deadline_or_observed_owner_changed"


@pytest.mark.parametrize("claimed_input_count", [0, True, None])
def test_success_claim_without_positive_integer_input_cannot_unlock_retained_stage(
    png, clock, claimed_input_count,
):
    env, events = Desktop(png, clock), []

    class ClaimingBackend(Backend):
        """Deliberately violate the backend contract to test the runner boundary.

        Real OSWorld verification binds original receipts and rejects tampering;
        this fixture is neither an OSWorld result nor external evidence.
        """

        attempts = 0

        def execute(self, candidate, observation_id, decision_id):
            self.attempts += 1
            assert candidate.capability == "agent.osworld.visual_fill"
            return ActionReceipt(
                observation_id, decision_id, candidate.id, candidate.channel, candidate.capability,
                True, clock.now, clock.now,
                {"environment_steps": claimed_input_count, "postconditions_met": True},
            )

        def verify(self, candidate, receipt):
            return Verification(True, "agent.effect", {"postconditions_met": True, "task_success": False})

    planner, backend = Planner(env), ClaimingBackend(env, pause=0)
    result = AgentRunner(backend, planner, policy=Policy(clock), progress=events.append,
                         config=AgentConfig(allow_actions=True, max_plans=2, max_failures=4,
                                            plan_ttl_s=60, timeout_s=120)).run("Set requested property")
    assert backend.attempts == 1 and planner.calls == 2
    assert env.actions == [] and not env.opened
    assert result.steps == 1
    assert result.property_edits == result.plan_reuses == result.plan_completion_claims == 0
    refused, = [event for event in events if event["event"] == "agent_prepare_repair"
                and event["status"] == "prefix_input_unconfirmed"]
    assert refused["receipt_id"] and refused["task_success"] is False
    attempted, = [event for event in events if event["event"] == "agent_step"]
    assert attempted["verified"] is False

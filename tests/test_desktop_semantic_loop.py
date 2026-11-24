"""Integration contracts for live rebinding, Jev choice, and slow-plan reuse."""

import io
import json
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentStep, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.policy import JevPolicy
from cua_jev.trace import JsonlTrace


def target(name, role="", **kwargs):
    return AgentTarget(name=name, role=role, app="Fixture", **kwargs)


def condition(kind, name, role=""):
    return AgentCondition(kind, target(name, role))


class Desktop:
    """A typed GUI simulator; its private reward must never reach either model."""

    def __init__(self, *, ignore_toggles=False):
        self.opened = False
        self.logging = self.metrics = False
        self.ignore_toggles = ignore_toggles
        self.actions = []

    def _get_obs(self):
        if not self.opened:
            content = (
                '<panel name="Input"><button name="Configure" enabled="true" '
                'screencoord="(20, 40)" size="(140, 40)"/></panel>'
                '<panel name="Output"><button name="Configure" enabled="true" '
                'screencoord="(400, 40)" size="(140, 40)"/></panel>'
            )
        else:
            # A layout transition moves the remaining checkbox after the first
            # toggle; the second action must bind the NEW coordinates.
            y = 300 if self.metrics else 120
            content = (
                f'<check-box name="Logging" enabled="true" checked="{str(self.logging).lower()}" '
                f'screencoord="(80, {y})" size="(140, 40)"/>'
                f'<check-box name="Metrics" enabled="true" checked="{str(self.metrics).lower()}" '
                'screencoord="(400, 120)" size="(140, 40)"/>'
            )
        xml = (
            '<desktop><application name="Fixture"><frame name="Control Center" active="true">'
            + content + '</frame></application></desktop>'
        )
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), "gray" if self.opened else "white").save(buffer, "PNG")
        return {"screenshot": buffer.getvalue(), "accessibility_tree": xml,
                "reward": "PRIVATE_EVALUATOR", "instruction": "PRIVATE_CONFIG"}

    def step(self, action, pause=0):
        assert action["action_type"] == "CLICK"
        x, y = action["parameters"]["x"], action["parameters"]["y"]
        self.actions.append((x, y))
        if not self.opened:
            # Only the second of two valid Configure controls opens this panel.
            self.opened = x > 300
        elif not self.ignore_toggles:
            if x > 300:
                self.metrics = not self.metrics
            else:
                assert y > 250 if self.metrics else y < 200
                self.logging = not self.logging
        return self._get_obs(), "PRIVATE_REWARD", True, {}


class Planner:
    def __init__(self):
        self.calls = 0

    def plan(self, goal, snapshot, history, *, max_actions=6):
        self.calls += 1
        steps = []
        names = {c.label for c in snapshot.controls}
        if "Logging" not in names:
            steps.append(AgentStep(
                "open", "click", target=target("Configure", "button"),
                postconditions=(condition("exists", "Logging"),),
            ))
        dep = ("open",) if steps else ()
        for name in ("Logging", "Metrics"):
            if any(c.label == name and c.checked for c in snapshot.controls):
                continue
            steps.append(AgentStep(
                name.lower(), "check", target=target(name, "checkbox"), depends_on=dep,
                preconditions=(condition("exists", name),),
                postconditions=(condition("checked", name),),
            ))
        return AgentPlan("Configure Output and enable both options", tuple(steps[:max_actions]),
                         done=not steps, reusable=True)


def test_real_backend_rebinds_across_screens_and_jev_selects_nonfirst_candidates():
    desktop, planner, requests = Desktop(), Planner(), []

    def answer(request):
        body = json.loads(request.content)
        requests.append(body)
        assert "PRIVATE" not in request.content.decode()
        candidates = body["state"]["available_actions"]
        assert all(not c.get("arguments") for c in candidates)
        selected = next((c for c in candidates if "panel: Output" in c["description"]), None)
        selected = selected or next((c for c in candidates if c["intent"] == "metrics"), candidates[0])
        return httpx.Response(200, json={"answers": {"action": {
            "choice": selected["id"], "confidence": 1,
            "probabilities": {c["id"]: float(c is selected) for c in candidates},
        }}})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        backend = OSWorldBackend(desktop, pause=0)
        result = AgentRunner(
            backend, planner, policy=JevPolicy(api_key="test-only", client=client, retries=0),
            config=AgentConfig(allow_actions=True),
        ).run("Enable the two Output options", success=lambda _: desktop.logging and desktop.metrics)
        backend.close()
    assert result.success
    assert result.steps == 3 and planner.calls == 1 and result.plan_reuses == 2
    assert result.candidate_counts == (2, 2, 1) and result.multi_candidate_steps == 2
    assert result.distinct_target_counts == (2, 2, 1)
    assert result.to_dict()["distinct_target_counts"] == (2, 2, 1)
    assert result.semantic_steps == 3
    assert desktop.actions == [(470, 60), (470, 140), (150, 320)]
    assert requests[0]["state"]["state"]["controls"]
    assert requests[1]["state"]["state"]["ready_steps"][0]["conditions"][0]["satisfied"]


def test_failed_postcondition_replans_without_completing_dependent_steps():
    desktop, planner, trace = Desktop(ignore_toggles=True), Planner(), JsonlTrace()
    desktop.opened = True
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(), trace=trace,
        config=AgentConfig(allow_actions=True, max_failures=2),
    ).run("Enable options")
    backend.close()
    assert result.status == "stuck" and planner.calls == 2
    assert result.semantic_steps == 0 and result.replan_reasons["effect_failed"] == 1
    assert all(not event["payload"]["verified"] for event in trace.events if event["kind"] == "agent_step")


@pytest.mark.parametrize("operation,initial,desired", [("check", False, True), ("uncheck", True, False)])
def test_unrelated_true_postcondition_does_not_skip_required_checkbox_input(operation, initial, desired):
    class FixedPlanner:
        def plan(self, *args, **kwargs):
            return AgentPlan("Update Logging", (AgentStep(
                "logging", operation, target=target("Logging", "checkbox"),
                # A model's broad or mistaken predicate cannot substitute for
                # the explicit operation's own target-state requirement.
                postconditions=(condition("checked", "Metrics"),),
            ),), reusable=True)

    desktop = Desktop()
    desktop.opened = desktop.metrics = True
    desktop.logging = initial
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, FixedPlanner(), policy=DirectPlanPolicy(), config=AgentConfig(allow_actions=True),
    ).run("Set Logging to the requested state", success=lambda _: desktop.logging == desired)
    backend.close()
    assert result.success and result.steps == result.semantic_steps == 1
    assert desktop.logging == desired and len(desktop.actions) == 1
    assert result.condition_skips == 0


def test_unrelated_visible_text_does_not_skip_fill_target_value():
    class Editor:
        value = "old"
        focused = False

        def __init__(self):
            self.actions = []

        def _get_obs(self):
            output = io.BytesIO()
            Image.new("RGB", (800, 600), "white").save(output, "PNG")
            return {
                "screenshot": output.getvalue(),
                "accessibility_tree": (
                    '<desktop><application name="Fixture"><frame name="Editor" active="true">'
                    '<label name="Ada"/><entry name="Name" editable="true" enabled="true" '
                    f'value="{self.value}" focused="{str(self.focused).lower()}" '
                    'screencoord="(50, 50)" size="(200, 40)"/>'
                    '</frame></application></desktop>'
                ),
            }

        def step(self, action, pause=0):
            self.actions.append(action)
            if action["action_type"] == "CLICK":
                self.focused = True
            elif action["action_type"] == "TYPING":
                self.value = action["parameters"]["text"]
            else:
                assert action == {"action_type": "HOTKEY", "parameters": {"keys": ["ctrl", "a"]}}
            return self._get_obs(), None, False, {}

    class FixedPlanner:
        def plan(self, *args, **kwargs):
            return AgentPlan("Fill Name", (AgentStep(
                "name", "fill", value="Ada", target=target("Name", "textbox"),
                postconditions=(AgentCondition("text_contains", value="Ada"),),
            ),), reusable=True)

    editor = Editor()
    backend = OSWorldBackend(editor, pause=0)
    result = AgentRunner(
        backend, FixedPlanner(), policy=DirectPlanPolicy(), config=AgentConfig(allow_actions=True),
    ).run("Set the name to Ada", success=lambda _: editor.value == "Ada")
    backend.close()
    assert result.success and result.steps == result.semantic_steps == 1
    assert result.condition_skips == 0
    assert [action["action_type"] for action in editor.actions] == ["CLICK", "HOTKEY", "TYPING"]


def test_correct_target_state_still_skips_idempotent_step_and_unlocks_dependency():
    class FixedPlanner:
        def plan(self, *args, **kwargs):
            return AgentPlan("Enable both options", (
                AgentStep("metrics", "check", target=target("Metrics"),
                          postconditions=(condition("checked", "Metrics"),)),
                AgentStep("logging", "check", target=target("Logging"), depends_on=("metrics",),
                          postconditions=(condition("checked", "Logging"),)),
            ), reusable=True)

    desktop = Desktop()
    desktop.opened = desktop.metrics = True
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, FixedPlanner(), policy=DirectPlanPolicy(), config=AgentConfig(allow_actions=True),
    ).run("Enable both options", success=lambda _: desktop.metrics and desktop.logging)
    backend.close()
    assert result.success and len(desktop.actions) == result.steps == 1
    assert result.condition_skips == 1 and result.plan_reuses == 0


def test_ambiguous_target_state_requires_choice_instead_of_skipping():
    class DuplicateLabels(Desktop):
        def _get_obs(self):
            observation = super()._get_obs()
            observation["accessibility_tree"] = observation["accessibility_tree"].replace(
                'name="Logging"', 'name="Option"'
            ).replace('name="Metrics"', 'name="Option"')
            return observation

    class FixedPlanner:
        def plan(self, *args, **kwargs):
            return AgentPlan("Enable the remaining option", (AgentStep(
                "option", "check", target=target("Option", "checkbox"),
                postconditions=(AgentCondition("text_contains", value="Option"),),
            ),), reusable=True)

    class ChooseSecond(DirectPlanPolicy):
        def choose(self, observation, candidates):
            assert len(candidates) == 2
            return super().choose(observation, tuple(reversed(candidates)))

    desktop = DuplicateLabels()
    desktop.opened = desktop.logging = True
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, FixedPlanner(), policy=ChooseSecond(), config=AgentConfig(allow_actions=True),
    ).run("Enable both options", success=lambda _: desktop.logging and desktop.metrics)
    backend.close()
    assert result.success and result.steps == len(desktop.actions) == 1
    assert result.condition_skips == 0 and result.candidate_counts == (2,)


@pytest.mark.parametrize("mode", ["hybrid", "vlm"])
def test_missing_target_uses_bounded_observation_recovery_then_replans(monkeypatch, mode):
    from cua_jev import agent_runner

    monkeypatch.setattr(agent_runner.time, "sleep", lambda _: None)
    step = AgentStep(
        "future", "click", target=target("Missing"), recovery="observe",
        postconditions=(condition("exists", "Future panel"),),
    )

    class MissingPlanner:
        def __init__(self):
            self.calls = 0

        def plan(self, *args, **kwargs):
            self.calls += 1
            return AgentPlan("Await future control", (step,), reusable=True)

    planner, desktop = MissingPlanner(), Desktop()
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(),
        config=AgentConfig(mode=mode, allow_actions=True, max_failures=2),
    ).run("Wait for a genuinely available control")
    backend.close()
    assert result.status == "stuck" and not desktop.actions
    assert planner.calls == 2 and result.recovery_observations == 4


def test_baseline_remains_one_vlm_call_per_action():
    desktop, planner = Desktop(), Planner()
    desktop.opened = True
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner, config=AgentConfig(mode="vlm", allow_actions=True),
    ).run("Enable options", success=lambda _: desktop.logging and desktop.metrics)
    backend.close()
    assert result.success and result.steps == planner.calls == 2
    assert result.plan_reuses == result.jev_calls == 0


def test_expired_semantic_plan_replans_instead_of_reusing(monkeypatch):
    from cua_jev import agent_runner

    clock = [0.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: clock[0])
    desktop, planner = Desktop(), Planner()
    desktop.opened = True
    original_step = desktop.step

    def action(*args, **kwargs):
        observation = original_step(*args, **kwargs)
        clock[0] = 2.0
        return observation

    desktop.step = action
    backend = OSWorldBackend(desktop, pause=0)
    config = replace(AgentConfig(allow_actions=True), plan_ttl_s=1)
    result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), config=config).run(
        "Enable options", success=lambda _: desktop.logging and desktop.metrics,
    )
    backend.close()
    assert result.success and planner.calls == 2
    assert result.plan_reuses == 0 and result.replan_reasons["plan_expired"] == 1


def test_slow_jev_cannot_commit_an_expired_semantic_plan(monkeypatch):
    from cua_jev import agent_runner

    clock = [0.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: clock[0])

    class DelayedChoice(DirectPlanPolicy):
        calls = 0

        def choose(self, observation, candidates):
            self.calls += 1
            decision = super().choose(observation, candidates)
            # The second choice would reuse a plan that already executed one
            # verified action, but expires before any second input is sent.
            if self.calls == 2:
                clock[0] = 2.0
            return decision

    desktop, planner = Desktop(), Planner()
    desktop.opened = True
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner, policy=DelayedChoice(),
        config=AgentConfig(allow_actions=True, plan_ttl_s=1, singleton_fast_path=False),
    ).run("Enable options", success=lambda _: desktop.logging and desktop.metrics)
    backend.close()
    assert result.success and result.steps == 2 and len(desktop.actions) == 2
    assert planner.calls == 2 and result.replan_reasons["plan_expired_before_action"] == 1
    assert result.plan_reuses == 0
    assert result.jev_calls == 3 and result.candidate_counts == (2, 1, 1)


def test_failed_effect_after_a_verified_action_is_not_counted_as_plan_reuse():
    class PartialDesktop(Desktop):
        def step(self, action, pause=0):
            if self.logging:
                self.ignore_toggles = True
            return super().step(action, pause=pause)

    desktop, planner = PartialDesktop(), Planner()
    desktop.opened = True
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(),
        config=AgentConfig(allow_actions=True, max_failures=2),
    ).run("Enable both options")
    backend.close()
    assert result.status == "stuck" and desktop.logging and not desktop.metrics
    assert result.steps == 3 and result.semantic_steps == 1
    assert result.plan_reuses == 0 and planner.calls == 2


def test_delayed_postcondition_is_observed_without_repeating_input(monkeypatch):
    from cua_jev import agent_runner

    monkeypatch.setattr(agent_runner.time, "sleep", lambda _: None)

    class DelayedDesktop(Desktop):
        ticks = 0

        def step(self, action, pause=0):
            self.actions.append(action)
            self.ticks = 4
            return self._get_obs(), None, False, {}

        def _get_obs(self):
            if self.ticks:
                self.ticks -= 1
                if self.ticks == 0:
                    self.metrics = True
            return super()._get_obs()

    class DelayedPlanner:
        calls = 0

        def plan(self, *args, **kwargs):
            self.calls += 1
            return AgentPlan("Enable metrics", (AgentStep(
                "metrics", "check", target=target("Metrics", "checkbox"),
                preconditions=(condition("exists", "Metrics"),),
                postconditions=(condition("checked", "Metrics"),), recovery="observe",
            ),), reusable=True)

    desktop, planner = DelayedDesktop(), DelayedPlanner()
    desktop.opened = True
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(), config=AgentConfig(allow_actions=True),
    ).run("Enable metrics", success=lambda _: desktop.metrics)
    backend.close()
    assert result.success and result.steps == 1 and planner.calls == 1
    assert len(desktop.actions) == 1 and result.recovery_observations == 2


def test_excessive_candidates_replan_without_sending_an_invalid_jev_request():
    class BroadBackend(OSWorldBackend):
        def compile(self, step, planned, current):
            choices = super().compile(step, planned, current)
            return tuple(replace(choices[0], id=f"candidate-{i}") for i in range(256)) if choices else ()

    class UnusedPolicy:
        def choose(self, *_):
            raise AssertionError("over-budget candidates must not reach Jev")

    desktop, planner = Desktop(), Planner()
    backend = BroadBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner, policy=UnusedPolicy(), config=AgentConfig(allow_actions=True, max_failures=2),
    ).run("Open the correct configuration panel")
    backend.close()
    assert result.status == "stuck" and not desktop.actions
    assert result.replan_reasons["candidate_budget_exceeded"] == 1 and result.jev_calls == 0


@pytest.mark.parametrize("failure_kind", ["no_executable_target", "preconditions_unmet"])
def test_semantic_recovery_explains_blocker_without_values_and_restores_multi_step_limit(failure_kind):
    secret = "private-current-input"
    comparison = "private-comparison-input"

    class IncompleteControls(Desktop):
        def _get_obs(self):
            observation = super()._get_obs()
            observation["accessibility_tree"] = observation["accessibility_tree"].replace(
                "</frame>",
                '<button name="Unavailable" enabled="true"/>'
                f'<entry name="Private" editable="true" enabled="true" value="{secret}" '
                'screencoord="(10, 450)" size="(200, 40)"/></frame>',
            )
            return observation

    class RecoveryPlanner:
        def __init__(self):
            self.limits = []
            self.histories = []

        def plan(self, goal, snapshot, history, *, max_actions=6):
            self.limits.append(max_actions)
            self.histories.append(list(history))
            if len(self.limits) == 1:
                unavailable = failure_kind == "no_executable_target"
                return AgentPlan("Enable both options", (AgentStep(
                    "blocked", "click" if unavailable else "check",
                    target=target("Unavailable" if unavailable else "Logging"),
                    preconditions=(AgentCondition(
                        "value_equals", target("Private"), secret if unavailable else comparison,
                    ),),
                    postconditions=(condition("checked", "Logging"),),
                ),), reusable=True)
            return Planner().plan(goal, snapshot, history, max_actions=max_actions)

    class PublicPolicy(DirectPlanPolicy):
        requests = []

        def choose(self, observation, candidates):
            exposed = {
                "state": observation.state, "actions": [candidate.to_dict() for candidate in candidates]
            }
            self.requests.append(exposed)
            assert secret not in json.dumps(exposed) and comparison not in json.dumps(exposed)
            return super().choose(observation, candidates)

    desktop, planner, policy = IncompleteControls(), RecoveryPlanner(), PublicPolicy()
    desktop.opened = True
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner, policy=policy,
        config=AgentConfig(allow_actions=True, singleton_fast_path=False),
    ).run("Enable both options", success=lambda _: desktop.logging and desktop.metrics)
    backend.close()
    assert result.success and len(desktop.actions) == result.steps == 2
    assert planner.limits == [6, 1, 6]
    assert result.plan_reuses == 0 and result.replan_reasons["recovery_completed"] == 1
    feedback = planner.histories[1][-1]
    assert feedback["reason"] == failure_kind and feedback["recovery_required"]
    facts = feedback["ready_steps"][0]
    assert facts["step_id"] == "blocked" and facts["reason"] == failure_kind
    assert facts["match_count"] == 1 and facts["candidate_count"] == 0
    assert facts["matches"][0]["enabled"]
    assert facts["matches"][0]["has_box"] == (failure_kind == "preconditions_unmet")
    assert facts["preconditions"][0]["satisfied"] == (failure_kind == "no_executable_target")
    assert "value" not in facts["preconditions"][0]
    assert planner.histories[2][-1]["verified"]
    assert planner.histories[2][-1]["recovery_required"] is False
    assert policy.requests[0]["state"]["recent_actions"][-1] == feedback


def test_unconfirmed_effect_feedback_records_sent_input_then_allows_semantic_recovery():
    class RecoveryPlanner:
        def __init__(self):
            self.limits = []
            self.histories = []

        def plan(self, goal, snapshot, history, *, max_actions=6):
            self.limits.append(max_actions)
            self.histories.append(list(history))
            if len(self.limits) == 2:
                desktop.ignore_toggles = False
            return Planner().plan(goal, snapshot, history, max_actions=max_actions)

    desktop, planner = Desktop(ignore_toggles=True), RecoveryPlanner()
    desktop.opened = True
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(), config=AgentConfig(allow_actions=True),
    ).run("Enable both options", success=lambda _: desktop.logging and desktop.metrics)
    backend.close()
    assert result.success and result.steps == 3 and result.semantic_steps == 2
    assert planner.limits == [6, 1, 6] and result.plan_reuses == 0
    feedback = planner.histories[1][-1]
    assert feedback["reason"] == "postcondition_unconfirmed"
    assert feedback["verified"] is False and feedback["recovery_required"]
    assert feedback["target"]["name"] == "Logging"
    assert feedback["postconditions"][0]["kind"] == "checked"
    assert feedback["feedback"].startswith("Input was sent")


def test_unobservable_postcondition_feedback_does_not_claim_the_input_failed_or_expose_value():
    desktop = Desktop()
    desktop.opened = True
    histories = []
    private_comparison = "private expected content"

    class InspectThenStop:
        def plan(self, goal, snapshot, history, *, max_actions):
            histories.append(history)
            if history:
                return AgentPlan("Stop this diagnostic without asserting success", (), done=True)
            return AgentPlan("Enable the option", (
                AgentStep("enable", "check", target=target("Logging", "checkbox"),
                          postconditions=(AgentCondition(
                              "value_equals", target("Logging", "checkbox"), private_comparison,
                          ),)),
            ), reusable=True)

    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, InspectThenStop(), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True)).run("Enable logging")
    backend.close()
    assert desktop.logging and result.steps == 1 and not result.success
    feedback = histories[1][-1]
    assert feedback["reason"] == "postcondition_unconfirmed"
    assert feedback["postconditions"][0]["outcome"] == "unknown"
    assert feedback["postconditions"][0]["reason"] == "unsupported_state"
    assert "Unknown is not evidence to toggle again" in feedback["feedback"]
    assert private_comparison not in json.dumps(feedback)


@pytest.mark.parametrize("initially_observable", [True, False])
def test_failed_delayed_effect_uses_latest_observed_evidence(initially_observable):
    class ChangingObservationDesktop(Desktop):
        input_sent = False
        post_input_reads = 0

        def _get_obs(self):
            observation = super()._get_obs()
            if self.input_sent:
                self.post_input_reads += 1
                observable = initially_observable if self.post_input_reads == 1 else not initially_observable
                if not observable:
                    observation["accessibility_tree"] = observation["accessibility_tree"].replace(
                        'name="Logging"', 'name="Other option"',
                    )
            return observation

        def step(self, *args, **kwargs):
            self.input_sent = True
            return super().step(*args, **kwargs)

    desktop = ChangingObservationDesktop(ignore_toggles=True)
    desktop.opened = True
    histories = []

    class InspectThenStop:
        def plan(self, goal, snapshot, history, *, max_actions):
            histories.append(history)
            if history:
                return AgentPlan("Stop this diagnostic", (), done=True)
            return AgentPlan("Enable logging", (
                AgentStep("enable", "check", target=target("Logging", "checkbox"),
                          postconditions=(condition("checked", "Logging", "checkbox"),),
                          recovery="observe"),
            ), reusable=True)

    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, InspectThenStop(), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True)).run("Enable logging")
    backend.close()
    assert len(desktop.actions) == result.steps == 1 and not desktop.logging
    assert result.recovery_observations == 2 and not result.success
    feedback = histories[1][-1]
    evidence = feedback["postconditions"][0]
    assert evidence["outcome"] == ("unknown" if initially_observable else "false")
    assert evidence["reason"] == ("target_missing" if initially_observable else "state_mismatch")
    assert ("Unknown is not evidence" in feedback["feedback"]) == initially_observable


class NavigationDesktop:
    """Several distinct choices per screen; no fixed target survives the route."""

    def __init__(self, *, cycle=False):
        self.stage = 0
        self.light = False
        self.cycle = cycle
        self.actions = []

    def _get_obs(self):
        labels = [("Help", "Edit", "Tools"), ("Undo", "Preferences"), ("Network", "Display")]
        if self.stage < 3:
            content = ''.join(
                f'<button name="{name}" enabled="true" screencoord="({20 + i * 230}, 80)" '
                'size="(160, 50)"/>' for i, name in enumerate(labels[self.stage])
            )
        else:
            content = (
                f'<check-box name="Light mode" enabled="true" checked="{str(self.light).lower()}" '
                'screencoord="(60, 180)" size="(160, 50)"/>'
            )
        xml = '<desktop><application name="Fixture"><frame name="Workbench" active="true">'
        xml += content + '</frame></application></desktop>'
        image = io.BytesIO()
        Image.new("RGB", (800, 600), (230, 230 - self.stage * 40, 230)).save(image, "PNG")
        return {"screenshot": image.getvalue(), "accessibility_tree": xml}

    def step(self, action, pause=0):
        assert action["action_type"] == "CLICK"
        x = action["parameters"]["x"]
        self.actions.append((self.stage, x))
        if self.stage < 3 and 250 <= x < 450:
            self.stage = 0 if self.cycle and self.stage == 1 else self.stage + 1
        elif self.stage == 3:
            self.light = not self.light
        return self._get_obs(), "PRIVATE_REWARD", False, {}


class NavigationPlanner:
    def __init__(self, *, stop_on_recovery=False):
        self.calls = []
        self.stop_on_recovery = stop_on_recovery

    def plan(self, goal, snapshot, history, *, max_actions=6):
        self.calls.append((max_actions, history))
        if self.stop_on_recovery and len(self.calls) > 1:
            return AgentPlan("Stop after the bounded navigation failure", (), done=True)
        return AgentPlan("Expose theme settings and enable light mode", (
            AgentStep("find-theme", "click", label="Expose the application's theme settings",
                      target=AgentTarget(app="Fixture"), navigation=True,
                      postconditions=(condition("exists", "Light mode", "checkbox"),)),
            AgentStep("enable-light", "check", target=target("Light mode", "checkbox"),
                      depends_on=("find-theme",),
                      postconditions=(condition("checked", "Light mode", "checkbox"),)),
        ), reusable=True)


def test_navigation_jev_chooses_distinct_live_controls_for_multiple_hops_without_vlm():
    desktop, planner, requests = NavigationDesktop(), NavigationPlanner(), []

    def answer(request):
        body = json.loads(request.content)
        requests.append(body)
        candidates = body["state"]["available_actions"]
        name = ("Edit", "Preferences", "Display", "Light mode")[desktop.stage]
        selected = next(c for c in candidates if f" {name} [" in c["description"])
        return httpx.Response(200, json={"answers": {"action": {
            "choice": selected["id"], "confidence": 1,
            "probabilities": {c["id"]: float(c is selected) for c in candidates},
        }}})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        backend = OSWorldBackend(desktop, pause=0)
        result = AgentRunner(
            backend, planner, policy=JevPolicy(api_key="test-only", client=client, retries=0),
            config=AgentConfig(allow_actions=True),
        ).run("Enable light mode", success=lambda _: desktop.light)
        backend.close()
    assert result.success and len(planner.calls) == 1
    assert result.steps == 4 and result.plan_reuses == 3
    assert result.candidate_counts == (3, 2, 2, 1)
    assert result.navigation_steps == 3 and result.navigation_subgoals == 1
    for request in requests[:3]:
        state = request["state"]["state"]
        assert state["completed_steps"] == []
        assert len(state["ready_steps"]) == 1
        subgoal = state["ready_steps"][0]
        assert subgoal["navigation"] and "theme" in subgoal["label"]
        assert subgoal["target"]["name"] == "" and subgoal["target"]["role"] == ""
        assert subgoal["completion_conditions"][0]["satisfied"] is False
        assert all(c["intent"] == "find-theme" for c in request["state"]["available_actions"])
    assert len(requests) == result.jev_calls == 3 and result.singleton_decisions == 1


@pytest.mark.parametrize("repair_exact_route", [False, True])
def test_live_planner_contract_delegates_route_before_any_gui_input(repair_exact_route):
    """Exercise the production parser/validation path, not a fixed AgentPlan stub."""
    desktop, planner_requests, jev_requests = NavigationDesktop(), [], []
    trace = JsonlTrace()
    navigation = {
        "id": "find-theme", "operation": "click", "navigation": True,
        "label": "Expose the application's theme settings", "target": {"app": "Fixture"},
        "postconditions": [{"kind": "exists", "target": {"name": "Light mode", "app": "Fixture"}}],
    }
    edit = {
        "id": "edit", "operation": "click", "target": {"name": "Edit", "app": "Fixture"},
        "postconditions": [{"kind": "exists", "target": {"name": "Preferences", "app": "Fixture"}}],
    }
    enable = {
        "id": "enable-light", "operation": "check", "target": {"name": "Light mode", "app": "Fixture"},
        "depends_on": ["find-theme"],
        "postconditions": [{"kind": "checked", "target": {"name": "Light mode", "app": "Fixture"}}],
    }

    def plan_reply(request):
        body = json.loads(request.content)
        planner_requests.append(body)
        if len(planner_requests) <= 2:
            assert desktop.actions == []  # Rejected precise navigation never touches the GUI.
        else:
            assert repair_exact_route and desktop.stage == 1 and len(desktop.actions) == 1
            assert desktop.actions[0][0] == 0 and 250 <= desktop.actions[0][1] < 450
            state = json.loads(body["messages"][1]["content"][0]["text"])
            assert state["recent_actions"][-1]["verified"] is True
            assert state["recent_actions"][-1]["recovery_required"] is False
        steps = [navigation, enable]
        if repair_exact_route and len(planner_requests) == 1:
            steps = [edit, {**navigation, "depends_on": ["edit"]}, enable]
        elif repair_exact_route and len(planner_requests) == 2:
            steps = [edit]
        raw = {"summary": "Expose theme settings and enable light mode", "steps": steps, "done": False}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    def choose_reply(request):
        body = json.loads(request.content)
        jev_requests.append(body)
        candidates = body["state"]["available_actions"]
        label = ("Edit", "Preferences", "Display", "Light mode")[desktop.stage]
        selected = next(c for c in candidates if f" {label} [" in c["description"])
        return httpx.Response(200, json={"answers": {"action": {
            "choice": selected["id"], "confidence": 1,
            "probabilities": {c["id"]: float(c is selected) for c in candidates},
        }}})

    with (httpx.Client(transport=httpx.MockTransport(plan_reply)) as vlm_client,
          httpx.Client(transport=httpx.MockTransport(choose_reply)) as jev_client):
        planner = MultimodalPlanner("https://planner.example/v1", "test-only",
                                    profile="desktop", client=vlm_client)
        backend = OSWorldBackend(desktop, pause=0)
        result = AgentRunner(
            backend, planner, policy=JevPolicy(api_key="test-only", client=jev_client, retries=0),
            config=AgentConfig(allow_actions=True), trace=trace,
        ).run("Enable light mode", success=lambda _: desktop.light)
        backend.close()
    assert result.success and result.planner_calls == (3 if repair_exact_route else 1)
    assert len(desktop.actions) == 4
    if repair_exact_route:
        state = json.loads(planner_requests[1]["messages"][1]["content"][0]["text"])
        assert "navigation" in state["recent_actions"][-1]["validation"].lower()
        assert state["recent_actions"][-1]["recovery_required"] is True
        plans = [event["payload"] for event in trace.events if event["kind"] == "agent_plan"]
        assert [(item["max_actions"], item["recovery"]) for item in plans] == [(1, True), (6, False)]
        assert [step["id"] for step in plans[0]["steps"]] == ["edit"]
        assert [step["id"] for step in plans[1]["steps"]] == ["find-theme", "enable-light"]
        assert result.candidate_counts == (1, 2, 2, 1)
        assert result.navigation_steps == 2 and result.plan_reuses == 2
        assert len(jev_requests) == result.singleton_decisions == 2
    else:
        assert result.candidate_counts == (3, 2, 2, 1)
        assert result.navigation_steps == 3 and result.plan_reuses == 3
        assert len(jev_requests) == 3 and result.singleton_decisions == 1


def test_navigation_cycle_triggers_recovery_without_unlocking_dependent_actions():
    desktop, planner = NavigationDesktop(cycle=True), NavigationPlanner(stop_on_recovery=True)

    class SecondChoice:
        def choose(self, observation, candidates):
            return DirectPlanPolicy().choose(observation, candidates[1:2])

    trace = JsonlTrace()
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, planner, policy=SecondChoice(), trace=trace,
                         config=AgentConfig(allow_actions=True)).run("Enable light mode")
    backend.close()
    assert result.replan_reasons["navigation_cycle"] == 1
    assert result.steps == 2 and result.navigation_steps == 1 and result.navigation_subgoals == 0
    assert result.plan_reuses == 0 and [call[0] for call in planner.calls] == [6, 1]
    assert not desktop.light
    assert any(e["kind"] == "agent_recovery" and e["payload"].get("reason") == "navigation_cycle"
               for e in trace.events)


def test_explicit_whole_goal_conditions_stop_without_final_vlm_call_but_do_not_claim_success():
    desktop, original = NavigationDesktop(), NavigationPlanner()

    class CompletePlanner:
        def plan(self, *args, **kwargs):
            return replace(original.plan(*args, **kwargs),
                           completion_conditions=(condition("checked", "Light mode", "checkbox"),))

    class RoutePolicy:
        def choose(self, observation, candidates):
            index = 1 if desktop.stage < 3 else 0
            return DirectPlanPolicy().choose(observation, candidates[index:index + 1])

    backend, trace = OSWorldBackend(desktop, pause=0), JsonlTrace()
    result = AgentRunner(backend, CompletePlanner(), policy=RoutePolicy(), trace=trace,
                         config=AgentConfig(allow_actions=True)).run("Enable light mode")
    backend.close()
    assert result.status == "unverified" and not result.success
    assert desktop.light and result.steps == 4 and result.planner_calls == 1
    assert result.plan_completion_claims == 1 and result.navigation_steps == 3
    claim = next(e["payload"] for e in trace.events if e["kind"] == "agent_plan_completion")
    assert claim["conditions"][0]["satisfied"] and claim["task_success"] is False


@pytest.mark.parametrize("already_satisfied", [False, True])
def test_single_action_baseline_can_use_same_completion_protocol(already_satisfied):
    desktop = NavigationDesktop()
    desktop.stage, desktop.light = 3, already_satisfied
    complete = condition("checked", "Light mode", "checkbox")

    class FinalPlanner:
        def plan(self, goal, snapshot, history, *, max_actions):
            assert max_actions == 1
            return AgentPlan("Enable the requested setting", (
                AgentStep("enable", "check", target=target("Light mode", "checkbox"),
                          postconditions=(complete,)),
            ), reusable=True, completion_conditions=(complete,))

    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, FinalPlanner(),
                         config=AgentConfig(mode="vlm", allow_actions=True)).run("Enable light mode")
    backend.close()
    assert result.status == "unverified" and result.planner_calls == 1
    assert result.steps == int(not already_satisfied) and result.plan_completion_claims == 1
    assert result.jev_calls == 0


def test_goal_completion_rechecks_fresh_state_after_verified_last_action():
    class UnstableDesktop(NavigationDesktop):
        change_before_observation = False

        def _get_obs(self):
            if self.change_before_observation:
                self.light = False
                self.change_before_observation = False
            return super()._get_obs()

        def step(self, *args, **kwargs):
            response = super().step(*args, **kwargs)
            self.change_before_observation = self.light
            return response

    desktop = UnstableDesktop()
    desktop.stage = 3
    complete = condition("checked", "Light mode", "checkbox")

    class OncePlanner:
        calls = 0

        def plan(self, *args, **kwargs):
            self.calls += 1
            if self.calls > 1:
                return AgentPlan("Stop this diagnostic", (), done=True)
            return AgentPlan("Enable light mode", (
                AgentStep("enable", "check", target=target("Light mode", "checkbox"),
                          postconditions=(complete,)),
            ), reusable=True, completion_conditions=(complete,))

    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, OncePlanner(), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True)).run("Enable light mode")
    backend.close()
    assert result.semantic_steps == 1 and result.planner_calls == 2
    assert result.plan_completion_claims == 0 and not desktop.light and not result.success


def test_plan_completion_claim_cannot_override_failed_independent_acceptance():
    desktop = NavigationDesktop()
    desktop.stage, desktop.light = 3, True
    complete = condition("checked", "Light mode", "checkbox")

    class ClaimPlanner:
        def plan(self, *args, **kwargs):
            return AgentPlan("Already configured", (
                AgentStep("enable", "check", target=target("Light mode", "checkbox"),
                          postconditions=(complete,)),
            ), reusable=True, completion_conditions=(complete,))

    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, ClaimPlanner(), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True, max_failures=2)).run(
        "Enable light mode AND fulfill an unmet additional requirement", success=lambda _: False,
    )
    backend.close()
    assert result.status == "task_failure" and result.steps == 0 and not result.success
    assert result.plan_completion_claims == 2 and result.planner_calls == 2


@pytest.mark.parametrize("mode", ["hybrid", "vlm"])
def test_empty_done_conditions_stop_on_fresh_state_without_an_input(mode):
    desktop = NavigationDesktop()
    desktop.stage, desktop.light = 3, True
    snapshots = []

    class StopPlanner:
        def plan(self, goal, snapshot, history, *, max_actions):
            snapshots.append(snapshot)
            assert max_actions == (1 if mode == "vlm" else 6)
            return AgentPlan("The requested state already holds", (), done=True,
                             completion_conditions=(condition("checked", "Light mode", "checkbox"),))

    backend, trace = OSWorldBackend(desktop, pause=0), JsonlTrace()
    observations = []
    observe = backend.observe

    def counted_observe():
        snapshot = observe()
        observations.append(snapshot)
        return snapshot

    backend.observe = counted_observe
    result = AgentRunner(backend, StopPlanner(), policy=DirectPlanPolicy(), trace=trace,
                         config=AgentConfig(mode=mode, allow_actions=True)).run("Enable light mode")
    backend.close()
    assert result.status == "unverified" and not result.success
    assert result.planner_calls == 1 and result.plan_completion_claims == 1
    assert result.steps == result.jev_calls == 0 and not desktop.actions
    assert len(observations) == 2 and observations[0] is snapshots[0]
    assert observations[1] is not snapshots[0]
    claim = next(e["payload"] for e in trace.events if e["kind"] == "agent_plan_completion")
    assert claim["completed_steps"] == [] and claim["conditions"][0]["outcome"] == "true"
    assert claim["task_success"] is False


@pytest.mark.parametrize("mode", ["hybrid", "vlm"])
@pytest.mark.parametrize("changed_state", ["false", "unknown"])
def test_empty_done_rechecks_changes_during_planning_and_bounds_retry_without_input(mode, changed_state):
    desktop = NavigationDesktop()
    desktop.stage, desktop.light = 3, True

    class StaleStopPlanner:
        histories = []

        def plan(self, goal, snapshot, history, *, max_actions):
            self.histories.append(list(history))
            if len(self.histories) == 1:
                assert any(c.label == "Light mode" and c.checked for c in snapshot.controls)
                if changed_state == "false":
                    desktop.light = False
                else:
                    desktop.stage = 0
            return AgentPlan("Check the requested final state", (), done=True,
                             completion_conditions=(condition("checked", "Light mode", "checkbox"),))

    planner, trace = StaleStopPlanner(), JsonlTrace()
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
                         config=AgentConfig(mode=mode, allow_actions=True, max_failures=2)).run(
        "Enable light mode",
    )
    backend.close()
    assert result.status == "stuck" and not result.success
    assert result.planner_calls == 2 and result.plan_completion_claims == 0
    assert result.steps == result.jev_calls == 0 and not desktop.actions
    assert result.replan_reasons["completion_unconfirmed"] == 1
    feedback = planner.histories[1][-1]
    assert feedback["completion_conditions"][0]["outcome"] == changed_state
    assert feedback["completion_conditions"][0]["reason"] == (
        "state_mismatch" if changed_state == "false" else "target_missing"
    )
    assert feedback["completion_conditions"][0]["satisfied"] is False


@pytest.mark.parametrize("mode", ["hybrid", "vlm"])
@pytest.mark.parametrize("accept_after_planning", [False, True])
def test_empty_done_never_overrides_fresh_independent_acceptance(mode, accept_after_planning):
    desktop = NavigationDesktop()
    desktop.stage, desktop.light = 3, True

    class StopPlanner:
        calls = 0

        def plan(self, *args, **kwargs):
            self.calls += 1
            return AgentPlan("Already configured", (), done=True,
                             completion_conditions=(condition("checked", "Light mode", "checkbox"),))

    planner = StopPlanner()
    checks = []

    def accept(snapshot):
        checks.append(snapshot)
        return accept_after_planning and planner.calls > 0

    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, planner, policy=DirectPlanPolicy(),
                         config=AgentConfig(mode=mode, allow_actions=True, max_failures=2)).run(
        "Enable light mode and satisfy an independent requirement", success=accept,
    )
    backend.close()
    assert result.success is accept_after_planning
    assert result.status == ("success" if accept_after_planning else "task_failure")
    assert result.plan_completion_claims == (0 if accept_after_planning else 2)
    assert result.planner_calls == (1 if accept_after_planning else 2)
    assert len(checks) == (2 if accept_after_planning else 3)
    assert result.steps == 0 and not desktop.actions


@pytest.mark.parametrize("mode", ["hybrid", "vlm"])
@pytest.mark.parametrize("matches", [False, True])
def test_empty_done_condition_values_do_not_enter_trace_or_recovery_history(mode, matches):
    private_current, private_expected = "private-current-content", "private-expected-content"

    class TextDesktop(NavigationDesktop):
        def _get_obs(self):
            observation = super()._get_obs()
            observation["accessibility_tree"] = observation["accessibility_tree"].replace(
                "</frame>",
                '<entry name="Document" editable="true" enabled="true" '
                f'value="{private_expected if matches else private_current}" '
                'screencoord="(20, 300)" size="(200, 40)"/></frame>',
            )
            return observation

    class StopPlanner:
        histories = []

        def plan(self, goal, snapshot, history, *, max_actions):
            self.histories.append(list(history))
            return AgentPlan("Check the complete document state", (), done=True,
                             completion_conditions=(AgentCondition(
                                 "value_equals", target("Document"), private_expected,
                             ),))

    desktop, planner, trace = TextDesktop(), StopPlanner(), JsonlTrace()
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
                         config=AgentConfig(mode=mode, allow_actions=True, max_failures=2)).run(
        "Keep the requested document state",
    )
    backend.close()
    assert result.status == ("unverified" if matches else "stuck") and not result.success
    assert result.steps == result.jev_calls == 0 and not desktop.actions
    exposed = json.dumps({"trace": trace.events, "history": planner.histories})
    assert private_current not in exposed and private_expected not in exposed


@pytest.mark.parametrize("mode", ["hybrid", "vlm"])
def test_successful_final_recovery_retains_completion_until_fresh_check(mode):
    class IgnoredFirstInput(NavigationDesktop):
        ignored = False

        def step(self, action, pause=0):
            if not self.ignored:
                self.ignored = True
                return self._get_obs(), 0, False, {}
            return super().step(action, pause)

    desktop = IgnoredFirstInput()
    desktop.stage = 3
    complete = condition("checked", "Light mode", "checkbox")

    class RecoveryPlanner:
        calls = 0

        def plan(self, goal, snapshot, history, *, max_actions):
            self.calls += 1
            assert self.calls <= 2, "final recovery must not require another VLM call"
            if self.calls == 2:
                assert max_actions == 1 and history[-1]["recovery_required"]
            return AgentPlan("Enable light mode", (
                AgentStep("enable", "check", target=target("Light mode", "checkbox"),
                          postconditions=(complete,)),
            ), reusable=True, completion_conditions=(complete,))

    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, RecoveryPlanner(), policy=DirectPlanPolicy(),
                         config=AgentConfig(mode=mode, allow_actions=True)).run("Enable light mode")
    backend.close()
    assert result.status == "unverified" and not result.success
    assert desktop.light and result.planner_calls == 2 and result.steps == 2
    assert result.plan_completion_claims == 1 and result.replan_reasons["effect_failed"] == 1


def test_delayed_navigation_progress_is_included_in_cycle_detection():
    class DelayedNavigation(NavigationDesktop):
        deferred_stage = None
        delay_reads = 0
        delayed_once = False

        def _get_obs(self):
            if self.deferred_stage is not None:
                if self.delay_reads:
                    self.delay_reads -= 1
                else:
                    self.stage = self.deferred_stage
                    self.deferred_stage = None
            return super()._get_obs()

        def step(self, action, pause=0):
            response = super().step(action, pause)
            if not self.delayed_once:
                self.delayed_once = True
                self.deferred_stage, self.stage, self.delay_reads = self.stage, 0, 2
                return self._get_obs(), 0, False, {}
            return response

    class SecondChoice:
        def choose(self, observation, candidates):
            return DirectPlanPolicy().choose(observation, (candidates[1], candidates[0], *candidates[2:]))

    desktop, planner = DelayedNavigation(cycle=True), NavigationPlanner(stop_on_recovery=True)
    backend = OSWorldBackend(desktop, pause=0)
    # Delayed observations are only authorized by the subgoal's declared recovery policy.
    original = planner.plan

    def with_observation_recovery(*args, **kwargs):
        plan = original(*args, **kwargs)
        return replace(plan, steps=tuple(replace(step, recovery="observe") for step in plan.steps))

    planner.plan = with_observation_recovery
    result = AgentRunner(backend, planner, policy=SecondChoice(),
                         config=AgentConfig(allow_actions=True)).run("Enable light mode")
    backend.close()
    assert result.steps == 2 and result.navigation_steps == 1
    assert result.recovery_observations == 2
    assert result.replan_reasons["navigation_cycle"] == 1
    assert result.navigation_subgoals == 0 and not desktop.light

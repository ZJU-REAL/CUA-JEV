"""Native metadata plus same-frame OCR, with no real OCR/VM/service calls in tests."""

import copy
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from cua_jev.agent_planner import parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentControl
from cua_jev.desktop_menu_grounding import ImageWord, ImageWords, ground_menu_items, native_menu_items
from cua_jev.desktop_semantics import executable_control
from cua_jev.models import Decision
from cua_jev.osworld_backend import OSWorldBackend

FIXTURES = Path(__file__).parent / "fixtures"
STATE = "{https://accessibility.ubuntu.example.org/ns/state}"


@pytest.fixture
def recorded():
    fixture = json.loads((FIXTURES / "osworld_menu_text_v28.json").read_bytes())
    observations = {}
    for frame, data in fixture["frames"].items():
        png = (FIXTURES / data["image"]).read_bytes()
        assert hashlib.sha256(png).hexdigest() == data["image_sha256"]
        observations[frame] = {"screenshot": png, "accessibility_tree": data["xml"]}
    data = fixture["ocr"]
    words = ImageWords(data["source_image_sha256"], tuple(data["size"]), tuple(
        tuple(ImageWord(w["text"], tuple(w["box"])) for w in line) for line in data["lines"]
    ))
    return observations, words


class RecordedWords:
    def __init__(self, words):
        self.words, self.calls = words, []

    def recognize(self, raw, *, deadline=None):
        self.calls.append((hashlib.sha256(raw).hexdigest(), deadline))
        return self.words


def parse(observation, provider=None):
    return OSWorldBackend(object(), pause=0, menu_text_provider=provider)._snapshot(observation)


def test_real_menu_labels_obtain_real_ocr_boxes_without_new_state_or_field_capabilities(recorded):
    observations, words = recorded
    original = parse(observations["step-001"])
    augmented = parse(observations["step-001"], RecordedWords(words))
    added = [c for c in augmented.controls if c not in original.controls]
    assert {c.label for c in added} == {"Undo History", "Preferences", "Input Devices", "Modules", "Units"}
    pref = next(c for c in added if c.label == "Preferences")
    assert pref.box == (73, 594, 115, 605)
    assert pref.app == "gimp" and pref.window == "GNU Image Manipulation Program"
    assert all(c.role == "menuitem" and c.active and not c.editable and not c.selected for c in added)
    assert all(executable_control(c, "click") and not executable_control(c, "fill") for c in added)
    assert all(not executable_control(c, "check") for c in added)
    assert augmented.text == original.text and augmented.semantic_text == original.semantic_text
    assert augmented.a11y_complete and len({c.ref for c in augmented.controls}) == len(augmented.controls)


def test_disabled_default_is_identical_and_closed_menu_never_calls_ocr(recorded):
    observations, words = recorded
    first = parse(observations["step-001"])
    second = parse(observations["step-001"], None)
    assert first == second and first.to_dict() == second.to_dict()
    assert first.fingerprint() == second.fingerprint()
    provider = RecordedWords(words)
    assert parse(observations["initial"], provider) == parse(observations["initial"])
    assert provider.calls == []


@pytest.mark.parametrize("change", ["foreign_state", "foreign_action", "foreign_component", "inactive",
                                    "menu_unfocused", "disabled_ancestor", "two_windows", "two_menus"])
def test_invalid_or_ambiguous_native_context_never_gets_pointer_controls(recorded, change):
    observations, words = recorded
    observation = copy.deepcopy(observations["step-001"])
    xml = observation["accessibility_tree"]
    if change.startswith("foreign_"):
        namespace = change.split("_", 1)[1]
        xml = xml.replace(f"https://accessibility.ubuntu.example.org/ns/{namespace}",
                          f"https://unrelated.invalid/ns/{namespace}")
    else:
        root = ET.fromstring(xml)
        frame = next(n for n in root.iter() if n.tag == "frame")
        menu = next(n for n in root.iter() if n.tag == "menu" and n.get("name") == "Edit")
        if change == "inactive":
            frame.set(STATE + "active", "false")
        elif change == "menu_unfocused":
            menu.set(STATE + "focused", "false")
        elif change == "disabled_ancestor":
            frame.set(STATE + "sensitive", "false")
        elif change == "two_windows":
            twin = copy.deepcopy(frame)
            twin.set("name", "Sibling")
            next(n for n in root.iter() if n.tag == "application").append(twin)
        else:
            twin = copy.deepcopy(menu)
            twin.set("name", "Other")
            next(n for n in root.iter() if n.tag == "menu-bar").append(twin)
        xml = ET.tostring(root, encoding="unicode")
    observation["accessibility_tree"] = xml
    original = parse(observation)
    augmented = parse(observation, RecordedWords(words))
    assert augmented == original


@pytest.mark.parametrize("change", ["wrong_image", "wrong_size", "nan", "outside", "over_budget"])
def test_bad_ocr_provenance_or_regions_never_supply_controls(recorded, change):
    observations, words = recorded
    original = parse(observations["step-001"])
    if change == "wrong_image":
        words = replace(words, image_sha256="0" * 64)
    elif change == "wrong_size":
        words = replace(words, size=(100, 100))
    else:
        word = ImageWord("Preferences", (100.0, 640.0, 220.0, 655.0))
        if change == "nan":
            word = replace(word, box=(float("nan"), 640, 220, 655))
        elif change == "outside":
            word = replace(word, box=(-1, 640, 220, 655))
        words = replace(words, lines=((word,),) * (2001 if change == "over_budget" else 1))
    assert parse(observations["step-001"], RecordedWords(words)) == original


@pytest.mark.parametrize("kind", ["disabled", "same_native_name", "competing_native_name",
                                  "two_image_regions", "isolated_word", "noisy_long_word"])
def test_lexical_ambiguities_and_missing_menu_peers_fail_closed(recorded, kind):
    observations, words = recorded
    observation = copy.deepcopy(observations["step-001"])
    root = ET.fromstring(observation["accessibility_tree"])
    menu = next(n for n in root.iter() if n.tag == "menu" and n.get("name") == "Edit")
    pref = next(n for n in menu if n.get("name") == "Preferences")
    if kind == "disabled":
        pref.set(STATE + "enabled", "false")
    elif kind in {"same_native_name", "competing_native_name"}:
        twin = copy.deepcopy(pref)
        if kind == "competing_native_name":
            twin.set("name", "Pxeferences")
        menu.append(twin)
    elif kind == "two_image_regions":
        words = replace(words, lines=(*words.lines, (ImageWord("Preferences", (400, 640, 480, 655)),)))
    elif kind == "isolated_word":
        words = replace(words, lines=((ImageWord("Preferences", (140, 640, 220, 655)),),))
    else:
        words = replace(words, lines=((ImageWord("Pxxyerences", (140, 640, 220, 655)),),))
    observation["accessibility_tree"] = ET.tostring(root, encoding="unicode")
    snapshot = parse(observation, RecordedWords(words))
    assert not any(c.ref.startswith("m") and c.label == "Preferences" for c in snapshot.controls)


def test_optional_provider_failure_preserves_original_observation(recorded):
    observations, _ = recorded

    class Failed:
        def recognize(self, *_args, **_kwargs):
            raise TimeoutError("local diagnostic timeout")

    backend = OSWorldBackend(object(), menu_text_provider=Failed())
    assert backend._snapshot(observations["step-001"]) == parse(observations["step-001"])
    assert backend.menu_grounding_metrics["failures"] == 1


def test_added_control_overflow_marks_incomplete_without_a_truncated_prefix(recorded):
    observations, words = recorded
    observation = observations["step-001"]
    before = parse(observation)
    before = replace(before, controls=(*before.controls, *(AgentControl(f"extra{i}", "text", "text")
                     for i in range(512 - len(before.controls)))))
    after = ground_menu_items(before, observation["screenshot"], words,
                              native_menu_items(observation["accessibility_tree"], before))
    assert not after.a11y_complete and after.controls == before.controls and len(after.controls) == 512


@pytest.mark.parametrize("change", ["moved", "provider_failure", "menu_closed", "deadline"])
def test_augmented_targets_keep_fresh_input_preflight_and_deadline_guards(recorded, monkeypatch, change):
    observations, words = recorded

    class Desktop:
        observation = observations["step-001"]
        actions = []

        def _get_obs(self):
            return self.observation

        def step(self, action, pause=0):
            self.actions.append(action)
            return self.observation, 0, False, {}

    clock = {"now": 0.0}

    class Provider(RecordedWords):
        def recognize(self, raw, *, deadline=None):
            if deadline is not None:
                clock["now"] = deadline + 1
            return super().recognize(raw, deadline=deadline)

    env, provider = Desktop(), Provider(words)
    backend = OSWorldBackend(env, pause=0, menu_text_provider=provider)
    before = backend.observe()
    data = {"intent": "pursue_property", "label": "Set undo levels to100", "scope": {"app": "gimp"},
            "edit": {"operation": "fill", "value": "100"}}
    plan = parse_plan(data, before, profile="desktop", max_actions=6)
    candidate = next(c for c in backend.compile(plan.steps[0], before, before)
                     if c.arguments["bound_control"]["label"] == "Preferences")
    if change == "moved":
        shifted = tuple(
            tuple(replace(w, box=(w.box[0] + 20, w.box[1], w.box[2] + 20, w.box[3])) for w in line)
            for line in words.lines if all(w.box[2] + 20 <= 1920 for w in line)
        )
        provider.words = replace(words, lines=shifted)
    elif change == "provider_failure":
        provider.recognize = lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError())
    elif change == "menu_closed":
        env.observation = observations["initial"]
    if change == "deadline":
        monkeypatch.setattr("cua_jev.osworld_backend.time.perf_counter", lambda: clock["now"])
        receipt = backend.execute_with_deadline(candidate, "obs", "decision", deadline=1.0)
    else:
        receipt = backend.execute(candidate, "obs", "decision")
        assert receipt.output["failure_code"] == "stale_observation"
    assert not receipt.success and env.actions == []
    assert not backend.verify(candidate, receipt).passed


def test_one_property_plan_can_continue_through_the_menu_geometry_gap(recorded):
    observations, words = recorded
    output = io.BytesIO()
    Image.new("RGB", (1920, 1080), "white").save(output, "PNG")

    class Desktop:
        def __init__(self):
            self.stage, self.actions, self.value, self.focus = 0, [], "5", False

        def _get_obs(self):
            if self.stage < 2:
                return observations["initial" if self.stage == 0 else "step-001"]
            state = 'enabled="true" sensitive="true" showing="true" visible="true"'
            xml = (f'<desktop-frame xmlns:s="https://accessibility.ubuntu.example.org/ns/state" '
                   'xmlns:c="https://accessibility.ubuntu.example.org/ns/component">'
                   '<application name="gimp">'
                   '<frame name="Preferences" s:active="true" s:enabled="true" s:sensitive="true" '
                   's:showing="true" s:visible="true" c:screencoord="(300, 200)" c:size="(1000, 600)">'
                   f'<spin-button name="Undo levels" {" ".join("s:" + p for p in state.split())} '
                   f's:editable="true" s:focused="{str(self.focus).lower()}" value="{self.value}" '
                   'c:screencoord="(500, 400)" c:size="(150, 30)" /></frame></application></desktop-frame>')
            return {"screenshot": output.getvalue(), "accessibility_tree": xml}

        def step(self, action, pause=0):
            self.actions.append(action)
            if action["action_type"] == "CLICK":
                if self.stage < 2:
                    self.stage += 1
                else:
                    self.focus = True
            if action["action_type"] == "TYPING":
                self.value = action["parameters"]["text"]
            return self._get_obs(), 0, False, {}

    class Planner:
        def plan(self, _goal, snapshot, *_args, **_kwargs):
            return parse_plan({"intent": "pursue_property", "label": "Set undo levels to100",
                               "scope": {"app": "gimp"}, "edit": {"operation": "fill", "value": "100"},
                               "goal_conditions": [{"kind": "value_equals", "target": {
                                   "name": "Undo levels", "app": "gimp"}, "value": "100"}]}, snapshot,
                              profile="desktop", max_actions=6, allow_property_goal_conditions=True)

    class Select:
        def choose(self, observation, candidates):
            candidate = next(c for c in candidates if c.description.startswith("click Preferences ")
                             or c.capability == "agent.osworld.fill")
            return Decision(observation.observation_id, candidate.id,
                            {c.id: float(c is candidate) for c in candidates}, 1.0, "local-fixture", 0)

    env = Desktop()
    backend = OSWorldBackend(env, pause=0, menu_text_provider=RecordedWords(words))
    result = AgentRunner(backend, Planner(), policy=Select(),
                         config=AgentConfig(allow_actions=True, max_plans=1)).run("Set undo levels to100")
    assert result.status == "unverified" and result.planner_calls == 1 and result.plan_reuses == 2
    assert result.steps == 3 and result.property_edits == 1 and env.value == "100"
    assert max(result.candidate_counts) > 1 and result.plan_completion_claims == 1
    assert [a["action_type"] for a in env.actions] == ["CLICK", "CLICK", "CLICK", "HOTKEY", "TYPING"]

"""Observation-only path pooling preserves the entire public state and model grammar."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from test_native_toggle_actions import Desktop

from cua_jev.agent_planner import MultimodalPlanner, _desktop_snapshot_payload
from cua_jev.agent_types import AgentControl
from cua_jev.desktop_snapshot_encoding import compact_snapshot_ancestors
from cua_jev.osworld_backend import OSWorldBackend


def canonical(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def expand(encoded):
    result = canonical(encoded)
    table = result.pop("ancestor_encoding", None)
    if table is None:
        return result
    assert table["version"] == "shared_paths_v1"

    def path(reference):
        texts, seen = [], set()
        current = reference["path_ref"]
        while current is not None:
            assert current not in seen
            seen.add(current)
            node = table["ancestor_paths"][current]
            texts.append(node["text"])
            current = node["parent"]
        return texts[::-1]

    if table["ancestor_path_default"]:
        result["control_defaults"]["ancestors"] = path(result["control_defaults"]["ancestors"])
    for index in table["ancestor_path_controls"]:
        control = result["controls"][index]
        control["ancestors"] = path(control["ancestors"])
    return result


@pytest.fixture
def dense():
    env = Desktop()
    base = OSWorldBackend(env)._snapshot(env._get_obs())
    controls = tuple(AgentControl(
        f"d{i}", f"Observed {i}", "input", role="textbox", app="Editor", window="Main",
        ancestors=("frame: Main", 'panel: "quoted" \\ path\n' + "\u8bbe\u7f6e" * 300,
                   f"group: {i % 3}"), value="", editable=True, active=True,
        box=(100, i, 200, i + 10), modal_context=i % 2 == 0,
    ) for i in range(100))
    return replace(base, controls=controls, text="Supplementary text remains literal")


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_dense_roundtrip_preserves_order_unicode_full_values_defaults_flags_and_capabilities(
    dense, complete, explicit,
):
    dense = replace(dense, a11y_complete=complete)
    before_fingerprint = dense.fingerprint()
    original = _desktop_snapshot_payload(dense, explicit_box_spaces=explicit)
    untouched = copy.deepcopy(original)
    encoded = compact_snapshot_ancestors(original)
    assert "ancestor_encoding" in encoded
    assert expand(encoded) == canonical(original)
    assert original == untouched and dense.fingerprint() == before_fingerprint
    assert len(json.dumps(encoded)) < len(json.dumps(original)) // 2


def test_recorded_large_vscode_tree_retains_every_public_fact_and_tail_control():
    import xml.etree.ElementTree as ET

    fixture = json.loads((Path(__file__).parent / "fixtures/osworld_vscode_large_observation.json").read_text(
        encoding="utf-8",
    ))
    env = Desktop(ET.fromstring(fixture["xml"]))
    current = OSWorldBackend(env)._snapshot(env._get_obs())
    original = _desktop_snapshot_payload(current, explicit_box_spaces=True)
    encoded = compact_snapshot_ancestors(original)
    assert "ancestor_encoding" in encoded
    assert expand(encoded) == canonical(original)
    assert len(encoded["controls"]) == len(original["controls"])
    assert encoded["a11y_complete"] is original["a11y_complete"]
    assert not env.calls


def capture(current, *, enabled, limit=1, history=None):
    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        answer = ({"intent": "complete", "label": "Offline request capture"} if limit > 1 and not history
                  else {"summary": "Offline request capture", "steps": [], "done": True})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        planner = MultimodalPlanner("https://offline.invalid", "mock", profile="desktop",
                                    desktop_contract="property_intent_v33", client=client,
                                    pool_desktop_ancestors=enabled)
        planner.plan("User goal", current, history or [], max_actions=limit)
    return requests[0]


@pytest.mark.parametrize("limit,history", [
    (1, []), (6, []), (6, [{"recovery_required": True, "feedback": "Inspect current observation"}]),
])
def test_actual_requests_apply_only_the_shared_observation_encoding(dense, limit, history):
    before = capture(dense, enabled=False, limit=limit, history=history)
    after = capture(dense, enabled=True, limit=limit, history=history)
    raw_state = json.loads(before["messages"][1]["content"][0]["text"])
    compact_state = json.loads(after["messages"][1]["content"][0]["text"])
    compact_state["snapshot"] = expand(compact_state["snapshot"])
    assert compact_state == raw_state
    assert after["messages"][1]["content"][1] == before["messages"][1]["content"][1]
    assert "snapshot.ancestor_encoding" in after["messages"][0]["content"]
    assert "snapshot.ancestor_encoding" not in before["messages"][0]["content"]
    assert after["model"] == before["model"] and after["stream"] == before["stream"]


def test_small_requests_stay_byte_identical_and_baseline_equals_hybrid_immediate_recovery():
    env = Desktop()
    current = OSWorldBackend(env)._snapshot(env._get_obs())
    assert capture(current, enabled=True) == capture(current, enabled=False)
    history = [{"recovery_required": True, "feedback": "Inspect current observation"}]
    assert capture(current, enabled=True, limit=1, history=history) == capture(
        current, enabled=True, limit=6, history=history,
    )


def test_pooled_baseline_and_hybrid_immediate_recovery_have_identical_real_requests(dense):
    history = [{"recovery_required": True, "feedback": "Inspect current observation"}]
    assert capture(dense, enabled=True, limit=1, history=history) == capture(
        dense, enabled=True, limit=6, history=history,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_osworld_cli_forwards_the_encoding_to_planner_and_reports_it(monkeypatch, enabled):
    import argparse
    from types import SimpleNamespace

    from test_osworld_driver import driver

    parser = argparse.ArgumentParser()
    driver.add_plan_arguments(parser)
    options = vars(parser.parse_args(["--pool-desktop-ancestors"] if enabled else []))
    args = SimpleNamespace(**options, model_base_url="https://offline.invalid", model="mock",
                           allow_screenshot_upload=True, allow_insecure_model_http=False, use_env_proxy=False)
    captured = []

    def factory(*_args, **kwargs):
        captured.append(kwargs)
        return "offline-planner"

    monkeypatch.setattr(driver, "MultimodalPlanner", factory)
    assert driver.make_desktop_planner(args) == "offline-planner"
    assert captured[0]["pool_desktop_ancestors"] is enabled
    assert driver.experiment_settings(args)["desktop_snapshot_encoding"] == (
        "shared_ancestor_paths_v1_if_saving_4096_bytes" if enabled else "control_defaults_v1"
    )


@pytest.mark.parametrize("value", [1, "true", None])
def test_encoding_flag_is_a_strict_boolean(value):
    with pytest.raises(ValueError, match="pool_desktop_ancestors"):
        MultimodalPlanner("https://offline.invalid", "mock", profile="desktop", pool_desktop_ancestors=value)


def test_browser_requests_cannot_enable_desktop_encoding():
    with pytest.raises(ValueError, match="desktop profile"):
        MultimodalPlanner("https://offline.invalid", "mock", pool_desktop_ancestors=True)


def test_defaults_and_nested_reference_lookalikes_stay_literal(dense):
    original = _desktop_snapshot_payload(dense)
    original["controls"][0]["extra"] = {"ancestors": {"path_ref": "p0"}}
    original["control_defaults"]["ancestors"] = ["explicit default" * 100]
    encoded = compact_snapshot_ancestors(original)
    assert expand(encoded) == canonical(original)
    assert encoded["controls"][0]["extra"] == original["controls"][0]["extra"]

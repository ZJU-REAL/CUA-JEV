"""Bounded public-surface settling; no VM, network, model or GUI input."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cua_jev.osworld_remote import RemoteOSWorldError, _settling_diagnostic

SPEC = importlib.util.spec_from_file_location(
    "settling_driver", Path(__file__).parents[1] / "scripts/run_osworld_smoke.py",
)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def observation(label="Shell", *, tree=True, color="white", clock="Oct 24 10:01"):
    image = io.BytesIO()
    Image.new("RGB", (320, 240), color).save(image, format="PNG")
    xml = (f'<desktop><application name="gnome-shell"><frame name="{label}" active="true">'
           f'<button name="Activities"/><label name="{clock}" toolkit="clutter" screencoord="(100, 0)" '
           'size="(100, 25)"/></frame></application></desktop>')
    return {"screenshot": image.getvalue(), "accessibility_tree": xml if tree is True else tree}


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    perf_counter = monotonic

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(driver, "time", value)
    return value


def settle(monkeypatch, tmp_path, clock, initial, subsequent, *, read_cost=0, enabled=True):
    queue, calls, report = list(subsequent), [], {}

    def read(_env, remaining):
        calls.append(remaining)
        clock.now += read_cost
        return queue.pop(0)

    monkeypatch.setattr(driver, "_readiness_observe", read)
    env = SimpleNamespace(step=lambda *_: pytest.fail("settling cannot send input"))
    result = driver.wait_for_desktop_settled(env, initial, output=tmp_path, report=report, enabled=enabled)
    return result, report["passive_settling"], calls, queue


def test_delayed_application_and_partial_frames_reset_quiet_window(monkeypatch, tmp_path, clock):
    shell, partial, application = observation(), observation("Loading"), observation("Editor")
    last, report, calls, _ = settle(monkeypatch, tmp_path, clock, shell,
                                   [shell, partial, application, application, application,
                                    application, application])
    assert last is application and report["status"] == "settled"
    assert len(calls) == 7 and report["elapsed_ms"] == 7000
    assert report["stable_samples"] == 5 and report["stable_span_ms"] == 4000
    assert report["application_ready_proven"] is False and report["future_loading_excluded"] is False


def test_quiet_shell_is_legal_but_cannot_exclude_later_application_loading(monkeypatch, tmp_path, clock):
    shell, later = observation(), observation("Editor appears later")
    last, report, calls, queue = settle(monkeypatch, tmp_path, clock, shell, [shell] * 4 + [later])
    assert last is shell and len(calls) == 4 and queue == [later]
    assert report["status"] == "settled" and report["surface_stability"] == "observed_quiet"
    assert not report["application_ready_proven"] and not report["future_loading_excluded"]


def test_existing_narrow_clock_normalization_tolerates_a_ticking_shell(monkeypatch, tmp_path, clock):
    frames = [observation(clock=f"Oct 24 10:0{i}") for i in range(5)]
    last, report, calls, _ = settle(monkeypatch, tmp_path, clock, frames[0], frames[1:])
    assert last is frames[-1] and report["status"] == "settled" and len(calls) == 4


@pytest.mark.parametrize("tree", [None, "<desktop/>", "<broken", "x" * 2_000_001],
                         ids=["unavailable", "empty", "invalid", "oversized"])
def test_no_complete_tree_remains_unproven_without_failing_valid_visual_tasks(
    monkeypatch, tmp_path, clock, tree,
):
    frames = [observation(tree=tree) for _ in range(10)]
    last, report, calls, _ = settle(monkeypatch, tmp_path, clock, frames[0], frames[1:])
    assert last is frames[-1] and len(calls) == 9
    assert report["status"] == "budget_exhausted" and report["reason"] == "sample_limit"
    assert report["surface_stability"] == "unproven" and report["stable_samples"] == 0


def test_dynamic_content_is_not_stripped_to_manufacture_stability(monkeypatch, tmp_path, clock):
    frames = [observation(f"Progress {i}") for i in range(10)]
    last, report, calls, _ = settle(monkeypatch, tmp_path, clock, frames[0], frames[1:])
    assert last is frames[-1] and len(calls) == 9
    assert report["status"] == "budget_exhausted" and report["stable_samples"] == 1


def test_truncated_repeated_tree_cannot_establish_public_surface_stability(monkeypatch, tmp_path, clock):
    tree = '<desktop><application name="editor">' + '<button name="item"/>' * 513 + '</application></desktop>'
    frame = observation(tree=tree)
    _, report, calls, _ = settle(monkeypatch, tmp_path, clock, frame, [frame] * 9)
    assert len(calls) == 9 and report["status"] == "budget_exhausted"
    assert all(not sample["a11y_complete"] for sample in report["samples"])


def test_observation_cost_counts_toward_same_total_budget(monkeypatch, tmp_path, clock):
    frames = [observation(f"Loading {i}") for i in range(5)]
    last, report, calls, queue = settle(monkeypatch, tmp_path, clock, frames[0], frames[1:], read_cost=9)
    assert last is frames[2] and len(calls) == 2 and queue == frames[3:]
    assert report["reason"] == "observation_budget_reserve" and report["elapsed_ms"] == 21_000
    assert report["observation_reserve_ms"] == 18000
    assert calls == [29, 19]


def test_slow_sample_processing_cannot_claim_settled_after_budget(monkeypatch, tmp_path, clock):
    original = driver._settling_structure

    def process(*args):
        result = original(*args)
        clock.now += 10
        return result

    monkeypatch.setattr(driver, "_settling_structure", process)
    frame = observation()
    _last, report, calls, _ = settle(monkeypatch, tmp_path, clock, frame, [frame] * 3)
    assert report["status"] == "budget_exhausted" and len(calls) == 2
    assert report["elapsed_ms"] == 32_000  # A bounded synchronous parse may finish beyond the deadline.


def test_reserve_avoids_starting_a_read_that_prior_latency_says_will_not_fit(monkeypatch, tmp_path, clock):
    calls, report = [], {}
    frames = [observation(f"Loading {i}") for i in range(5)]

    def read(_env, remaining):
        calls.append(remaining)
        if remaining <= 5:
            pytest.fail("a late read would be interrupted; reserve must decline it before starting")
        clock.now += 5
        return frames[len(calls)]

    monkeypatch.setattr(driver, "_readiness_observe", read)
    last = driver.wait_for_desktop_settled(object(), frames[0], output=tmp_path, report=report)
    diagnostic = report["passive_settling"]
    assert last is frames[-1] and calls == [29, 23, 17, 11]
    assert diagnostic["reason"] == "observation_budget_reserve" and diagnostic["elapsed_ms"] == 25_000
    assert diagnostic["surface_stability"] == "unproven" and diagnostic["observation_reserve_ms"] == 10000
    assert not diagnostic["application_ready_proven"] and not diagnostic["future_loading_excluded"]
    assert _settling_diagnostic(driver.settling_summary(diagnostic), last)["version"] == "public_structure_v3"


@pytest.mark.parametrize("successful_read_s", [12.05, 12.75])
def test_recorded_slow_read_cost_does_not_admit_the_refresh_that_previously_timed_out(
    monkeypatch, tmp_path, clock, successful_read_s,
):
    # Original v29a GIMP slots10/11 each had one 12-13s successful read,
    # then admitted a second read near the end of the30s startup budget.
    frame, calls, report = observation("Preferences"), [], {}

    def read(_env, remaining):
        calls.append(remaining)
        assert len(calls) == 1, "do not start a second risky read"
        clock.now += successful_read_s
        return frame

    monkeypatch.setattr(driver, "_readiness_observe", read)
    result = driver.wait_for_desktop_settled(object(), frame, output=tmp_path, report=report)
    diagnostic = report["passive_settling"]
    assert result is frame and len(calls) == 1
    assert diagnostic["status"] == "budget_exhausted"
    assert diagnostic["reason"] == "observation_budget_reserve"
    assert diagnostic["stable_samples"] == 2 and diagnostic["surface_stability"] == "unproven"
    assert diagnostic["elapsed_ms"] < 30000 and diagnostic["observation_reserve_ms"] > 24000
    assert not diagnostic["application_ready_proven"] and not diagnostic["future_loading_excluded"]


@pytest.mark.parametrize("failure", [driver._ReadinessDeadline(), OSError("PRIVATE transport body")])
def test_in_flight_error_still_fails_after_a_successful_read(monkeypatch, tmp_path, clock, failure):
    count, report = 0, {}

    def read(_env, remaining):
        nonlocal count
        count += 1
        assert remaining > 20  # Ample admission budget; this is an actual in-flight failure.
        if count == 2:
            raise failure
        clock.now += 1
        return observation("Loading changed")

    monkeypatch.setattr(driver, "_readiness_observe", read)
    with pytest.raises(driver.VMStartupError) as caught:
        driver.wait_for_desktop_settled(object(), observation(), output=tmp_path, report=report)
    assert count == 2 and report["passive_settling"]["status"] == "failed"
    assert "PRIVATE" not in str(caught.value.diagnostics)


def test_final_placeholder_is_rejected_when_reserve_declines_next_read(monkeypatch, tmp_path, clock):
    frame, blank = observation(), observation(color="black")
    report, calls = {}, []

    def read(_env, remaining):
        calls.append(remaining)
        clock.now += 20
        return blank

    monkeypatch.setattr(driver, "_readiness_observe", read)
    with pytest.raises(driver.VMStartupError) as caught:
        driver.wait_for_desktop_settled(object(), frame, output=tmp_path, report=report)
    assert len(calls) == 1
    assert caught.value.diagnostics["passive_settling"]["reason"] == "final_visual_evidence_unavailable"


@pytest.mark.parametrize("change", [
    {"observation_reserve_ms": 0}, {"observation_reserve_ms": True},
    {"observation_reserve_ms": float("nan")}, {"observation_reserve_ms": 600_001},
    {"elapsed_ms": 1000}, {"samples_observed": 1, "stable_samples": 1},
    {"version": []}, {"version": {}},
])
def test_reserve_diagnostic_cannot_invent_admission_or_accept_malformed_metadata(
    monkeypatch, tmp_path, clock, change,
):
    frames = [observation(f"Loading {i}") for i in range(5)]
    last, report, _, _ = settle(monkeypatch, tmp_path, clock, frames[0], frames[1:], read_cost=9)
    with pytest.raises(RemoteOSWorldError) as caught:
        _settling_diagnostic({**driver.settling_summary(report), **change}, last)
    assert caught.value.code == "invalid_handshake"


def test_legacy_worker_metadata_is_readable_but_cannot_impersonate_new_run_setting(
    monkeypatch, tmp_path, clock,
):
    initial = observation()
    _, diagnostic, _, _ = settle(monkeypatch, tmp_path, clock, initial, [initial] * 4)
    legacy = driver.settling_summary(diagnostic)
    legacy["version"] = "public_structure_v1"
    legacy.pop("observation_reserve_ms")
    assert _settling_diagnostic(legacy, initial)["version"] == "public_structure_v1"
    with pytest.raises(driver.VMStartupError):
        driver.use_worker_settling(SimpleNamespace(passive_settling=legacy), initial,
                                   output=tmp_path, report={}, enabled=True)


@pytest.mark.parametrize("failure,reason", [
    (driver._ReadinessDeadline(), "observation_timeout"),
    (driver._ReadinessBudgetUnavailable("bounded_observer_unavailable"), "bounded_observer_unavailable"),
    (OSError("PRIVATE transport body"), "observation_error"),
])
def test_observation_failure_does_not_continue_on_a_possibly_desynchronized_transport(
    monkeypatch, tmp_path, clock, failure, reason,
):
    def read(*_):
        raise failure

    monkeypatch.setattr(driver, "_readiness_observe", read)
    report = {}
    with pytest.raises(driver.VMStartupError) as caught:
        driver.wait_for_desktop_settled(object(), observation(), output=tmp_path, report=report)
    assert report["passive_settling"]["reason"] == reason
    assert "PRIVATE" not in str(caught.value.diagnostics)
    assert (tmp_path / "settling-last.png").is_file()


def test_a_final_placeholder_is_not_excused_by_a_previous_valid_surface(monkeypatch, tmp_path, clock):
    blank = observation(color="black")
    with pytest.raises(driver.VMStartupError) as caught:
        settle(monkeypatch, tmp_path, clock, observation(), [blank] * 9)
    assert caught.value.diagnostics["passive_settling"]["reason"] == "final_visual_evidence_unavailable"


def test_disabled_records_no_sampling_and_exact_initial_frame(monkeypatch, tmp_path, clock):
    initial = observation()
    last, report, calls, _ = settle(monkeypatch, tmp_path, clock, initial, [], enabled=False)
    assert last is initial and calls == [] and report["samples_observed"] == 0
    assert report["status"] == "disabled" and report["surface_stability"] == "unproven"
    assert _settling_diagnostic(driver.settling_summary(report), initial)["enabled"] is False


@pytest.mark.parametrize("change", [
    {"application_ready_proven": True}, {"future_loading_excluded": True},
    {"elapsed_ms": float("nan")}, {"elapsed_ms": True}, {"stable_samples": 11},
    {"version": "PRIVATE unknown version"}, {"last_screenshot_sha256": "0" * 64},
    {"reason": "PRIVATE exception body"}, {"unexpected": "PRIVATE"},
])
def test_remote_diagnostic_rejects_unbounded_or_invented_evidence(monkeypatch, tmp_path, clock, change):
    initial = observation()
    _, report, _, _ = settle(monkeypatch, tmp_path, clock, initial, [initial] * 4)
    with pytest.raises(RemoteOSWorldError) as caught:
        _settling_diagnostic({**driver.settling_summary(report), **change}, initial)
    assert caught.value.code == "invalid_handshake" and "PRIVATE" not in str(caught.value)


def test_worker_result_is_copied_once_and_unknown_or_mismatched_settings_fail_closed(
    monkeypatch, tmp_path, clock,
):
    initial = observation()
    _, diagnostic, _, _ = settle(monkeypatch, tmp_path, clock, initial, [initial] * 4)
    checked = _settling_diagnostic(driver.settling_summary(diagnostic), initial)
    env = SimpleNamespace(passive_settling=checked, _get_obs=lambda: pytest.fail("no second settling"))
    report = {}
    assert driver.use_worker_settling(env, initial, output=tmp_path, report=report, enabled=True) is initial
    assert report["passive_settling"] == checked
    for value in (None, {**checked, "enabled": False}):
        env.passive_settling = value
        with pytest.raises(driver.VMStartupError):
            driver.use_worker_settling(env, initial, output=tmp_path, report={}, enabled=True)
    assert _settling_diagnostic(None, initial) is None


def test_direct_episode_saves_latest_settled_observation_and_charges_setup_cost(
    monkeypatch, tmp_path, clock,
):
    shell, app = observation(), observation("Editor")
    pending = [shell, app, app, app, app, app]
    events, report = [], {}

    def read(_env, _remaining):
        events.append("observe")
        return pending.pop(0)

    monkeypatch.setattr(driver, "_readiness_observe", read)
    env = SimpleNamespace(
        controller=SimpleNamespace(get_screenshot=lambda: app["screenshot"]),
        evaluate=lambda: events.append("evaluate") or 0.0, close=lambda: events.append("close"),
    )

    class Runner:
        def __init__(self, *_args, **_kwargs):
            assert pending == []
            events.append("agent_started")

        def run(self, _goal, *, success):
            assert success is None
            return SimpleNamespace(to_dict=lambda: {"status": "unverified", "duration_ms": 0})

    monkeypatch.setattr(driver, "AgentRunner", Runner)
    args = SimpleNamespace(output=tmp_path, model="fake", vm_path=tmp_path / "base",
                           no_a11y=False, pause=0.5, max_steps=25, timeout=420, passive_settling=True)
    report = driver.run_episode(
        args, "os/public", {"instruction": "Public instruction"}, "vlm",
        env_factory=lambda **_: env, initialize=lambda *_: shell,
        backend_factory=lambda *_a, **_k: SimpleNamespace(close=lambda: None),
        planner_factory=lambda _: SimpleNamespace(close=lambda: None),
    )
    assert report["setup_wall_ms"] == report["passive_settling"]["elapsed_ms"] == 6000
    assert report["cleanup_status"] == "confirmed"
    assert events == ["observe"] * 6 + ["agent_started", "evaluate", "close"]
    assert (tmp_path / "os-public/vlm/initial-a11y.xml").read_text() == app["accessibility_tree"]

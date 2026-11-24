"""Desktop startup evidence and bounded waiting; no VM, input or evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("readiness_driver", ROOT / "scripts/run_osworld_smoke.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def observation(*, color="black", pointer=False, text=None, tree="<desktop-frame/>"):
    image = Image.new("RGB", (1920, 1080), color)
    drawing = ImageDraw.Draw(image)
    if pointer:
        # The same generic 13x20 arrow silhouette as the recorded blank frame.
        rows = ("#", "##", "###", "####", "#####", "######", "#######", "########",
                "#########", "##########", "###########", "############", "#############",
                "########", "########", "###.#####", "##..#####", "#....####", ".....####",
                "......##")
        for y, row in enumerate(rows):
            for x, value in enumerate(row):
                if value == "#":
                    drawing.point((963 + x, 542 + y), fill="white")
    if text is not None:
        drawing.text((1, 1), text, fill="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"screenshot": buffer.getvalue(), "accessibility_tree": tree}


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    perf_counter = monotonic

    def sleep(self, seconds):
        self.now += seconds


class Signal:
    ITIMER_REAL = 0
    SIGALRM = 14

    def __init__(self):
        self.timer = (0.0, 0.0)
        self.original = object()
        self.handler = self.original
        self.armed = []

    def getitimer(self, _which):
        return self.timer

    def getsignal(self, _which):
        return self.handler

    def signal(self, _which, handler):
        self.handler = handler

    def setitimer(self, _which, seconds):
        self.timer = (seconds, 0.0)
        self.armed.append(seconds)


@pytest.fixture
def budget(monkeypatch):
    clock, signals = Clock(), Signal()
    monkeypatch.setattr(driver, "time", clock)
    monkeypatch.setattr(driver, "signal", signals)
    return clock, signals


@pytest.mark.parametrize("tree,expected_nodes", [
    ("<desktop-frame/>", 1),
    ("<desktop-frame><application><frame><label name='stale'/></frame></application></desktop-frame>", 4),
    (None, 0), ("<invalid", 0),
])
def test_black_pointer_waits_even_with_populated_or_unavailable_accessibility(tree, expected_nodes):
    quality = driver.desktop_observation_quality(observation(pointer=True, tree=tree))
    assert quality["reason"] == "black_placeholder"
    assert quality["a11y_nodes"] == expected_nodes
    assert quality["foreground_pixels"] == 133  # Pixels above the near-black threshold.
    assert quality["foreground_bounds"] == [963, 542, 976, 562]
    assert quality["near_black_fraction"] > 0.9999


@pytest.mark.parametrize("value", [
    {"color": "white", "tree": None},
    {"color": "#181818", "tree": "<invalid"},
    {"text": "$", "tree": None},
    {"text": "user@machine:~$", "tree": "<desktop-frame/>"},
    {"text": "A", "tree": "<desktop-frame/>"},
    {"pointer": True, "text": "$", "tree": None},
])
def test_visual_fallback_and_sparse_terminal_text_are_not_rejected(value):
    assert driver.desktop_observation_quality(observation(**value))["reason"] == "visual_evidence_present"


@pytest.mark.parametrize("color", ["#090909", "#181818", "#222222", "#383838", "#404040", "#102030"])
def test_uniform_dark_background_with_only_arrow_waits_without_color_specific_rule(color):
    quality = driver.desktop_observation_quality(observation(color=color, pointer=True))
    assert quality["reason"] == "dark_pointer_placeholder"
    assert quality["dark_pointer_pixels"] == quality["dark_nonbackground_pixels"] == 133
    assert quality["dark_pointer_bounds"] == [963, 542, 976, 562]
    assert max(quality["dark_background_rgb"]) <= 64


def shadowed_pointer_observation(*, text=None, text_color="white"):
    public = observation(color="#222222", pointer=True)
    image = Image.open(io.BytesIO(public["screenshot"])).convert("RGB")
    silhouette = image.getchannel("R").point(lambda value: 255 if value > 128 else 0)
    # A faint two-pixel cursor shadow exceeds the former global 0.01% threshold.
    shadow = silhouette.filter(ImageFilter.MaxFilter(5))
    image.paste((36, 36, 36), mask=shadow)
    image.paste((255, 255, 255), mask=silhouette)
    if text is not None:
        ImageDraw.Draw(image).text((1, 1), text, fill=text_color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {**public, "screenshot": buffer.getvalue()}


def test_cursor_shadow_is_bounded_by_shape_not_global_pixel_fraction():
    quality = driver.desktop_observation_quality(shadowed_pointer_observation())
    assert quality["reason"] == "dark_pointer_placeholder"
    assert quality["dark_nonbackground_pixels"] / (1920 * 1080) > 0.0001
    assert quality["dark_nonbackground_bounds"] == [961, 540, 978, 564]
    assert quality["dark_pointer_pixels"] == 133


@pytest.mark.parametrize("text", ["$", "A", ">", ".", "|", "i", "~"])
@pytest.mark.parametrize("pointer", [False, True])
def test_single_character_on_dark_background_is_visual_evidence(text, pointer):
    quality = driver.desktop_observation_quality(observation(color="#222222", pointer=pointer, text=text))
    assert quality["reason"] == "visual_evidence_present"


def test_even_faint_text_outside_cursor_halo_is_preserved():
    quality = driver.desktop_observation_quality(shadowed_pointer_observation(text="A", text_color="#232323"))
    assert quality["reason"] == "visual_evidence_present"


@pytest.mark.parametrize("color,pointer", [("#222222", False), ("#414141", True), ("white", True)])
def test_flat_gray_without_arrow_and_non_dark_backgrounds_are_not_guessed_unready(color, pointer):
    # No general application-readiness inference: a flat gray image alone stays
    # usable, and this added branch only covers dark backgrounds plus an arrow.
    quality = driver.desktop_observation_quality(observation(color=color, pointer=pointer))
    assert quality["reason"] == "visual_evidence_present"


def test_populated_accessibility_cannot_override_gray_pointer_placeholder():
    public = observation(color="#222222", pointer=True, tree="<desktop-frame><application/></desktop-frame>")
    quality = driver.desktop_observation_quality(public)
    assert quality["reason"] == "dark_pointer_placeholder"
    assert quality["a11y_status"] == "populated" and quality["a11y_nodes"] == 2


def test_gray_placeholder_uses_existing_bounded_observe_only_recovery(tmp_path, budget):
    _clock, signals = budget
    initial, good, report, calls = observation(color="#222222", pointer=True), observation(text="$"), {}, []
    env = SimpleNamespace(_get_obs=lambda: calls.append("observe") or good)
    assert driver.wait_for_desktop_ready(env, initial, output=tmp_path, report=report) is good
    assert calls == ["observe"] and signals.armed == [29.0, 0]
    assert report["desktop_readiness"]["samples"][0]["reason"] == "dark_pointer_placeholder"
    assert report["desktop_readiness"]["status"] == "ready"


def test_recorded_gray_pointer_sha_when_local_evidence_is_available():
    folder = ROOT / "runs/osworld-fixed8-v11-docker/thunderbird-7b1e1ff9-bb85-49be-b01d-d6424be18cd0/hybrid"
    image = folder / "initial.png"
    if not image.is_file():
        pytest.skip("recorded ignored run artifacts are not part of the source distribution")
    screenshot = image.read_bytes()
    assert hashlib.sha256(screenshot).hexdigest() == (
        "203632f7681f2595f52cd6acc2c22dd251175833e0ebf281e9551d5070280f06"
    )
    quality = driver.desktop_observation_quality({
        "screenshot": screenshot,
        "accessibility_tree": (folder / "initial-a11y.xml").read_text(encoding="utf-8"),
    })
    assert quality["reason"] == "dark_pointer_placeholder" and quality["a11y_nodes"] == 1673
    assert quality["dark_background_rgb"] == [34, 34, 34]
    assert quality["dark_nonbackground_pixels"] == 228
    assert quality["dark_pointer_pixels"] == 126
    assert quality["dark_pointer_bounds"] == [1731, 974, 1744, 994]


@pytest.mark.parametrize("value", [None, object(), {}, {"screenshot": object()},
                                   {"screenshot": b"invalid PNG"}, {"screenshot": b""}])
def test_unavailable_observation_has_static_diagnostic(value):
    assert driver.desktop_observation_quality(value)["reason"] == "observation_unavailable"


@pytest.mark.parametrize("tree", ["<!DOCTYPE root [<!ENTITY e 'x'>]><root>&e;</root>", "x" * 2_000_001],
                         ids=["entities", "oversized"])
def test_unbounded_or_entity_xml_is_not_parsed_and_does_not_block_visual_fallback(tree):
    quality = driver.desktop_observation_quality(observation(color="white", tree=tree))
    assert quality["a11y_status"] == "unavailable" and quality["a11y_nodes"] == 0
    assert quality["reason"] == "visual_evidence_present"


def test_initial_good_frame_is_reused_without_observation_or_budget_changes(tmp_path, budget):
    _clock, signals = budget
    # An active external alarm need not be touched when no waiting is needed.
    signals.timer = (2.0, 0.0)
    initial, report = observation(color="white"), {}
    env = SimpleNamespace(_get_obs=lambda: pytest.fail("initial frame already usable"))
    assert driver.wait_for_desktop_ready(env, initial, output=tmp_path, report=report) is initial
    assert report["desktop_readiness"]["attempts"] == 1
    assert report["desktop_readiness"]["status"] == "ready"
    assert signals.armed == [] and signals.timer == (2.0, 0.0)
    assert not list(tmp_path.iterdir())


def test_black_frame_recovers_with_observation_only_and_returns_exact_new_frame(tmp_path, budget):
    _clock, signals = budget
    initial, good, report, calls = observation(pointer=True), observation(text="Ready >"), {}, []
    env = SimpleNamespace(_get_obs=lambda: calls.append("observe") or good)
    assert driver.wait_for_desktop_ready(env, initial, output=tmp_path, report=report) is good
    diagnostic = report["desktop_readiness"]
    assert calls == ["observe"]
    assert diagnostic["status"] == "ready" and diagnostic["attempts"] == 2
    assert diagnostic["elapsed_ms"] == 1000
    assert signals.armed == [29.0, 0] and signals.handler is signals.original
    assert (tmp_path / "readiness-first.png").read_bytes() == initial["screenshot"]
    assert (tmp_path / "readiness-first-a11y.xml").read_text() == initial["accessibility_tree"]


def test_attempt_limit_saves_last_evidence_and_never_uses_task_goal(tmp_path, budget):
    _clock, signals = budget
    initial = observation()
    last = observation(pointer=True, tree="<desktop-frame><application/></desktop-frame>")
    report, calls = {}, []
    env = SimpleNamespace(_get_obs=lambda: calls.append("observe") or last)
    with pytest.raises(driver.VMStartupError) as failure:
        driver.wait_for_desktop_ready(env, initial, output=tmp_path, report=report, max_attempts=3)
    diagnostic = report["desktop_readiness"]
    assert failure.value.diagnostics == {"desktop_readiness": diagnostic}
    assert diagnostic["status"] == "failed" and diagnostic["reason"] == "attempt_limit"
    assert diagnostic["attempts"] == 3 and len(diagnostic["samples"]) == 3
    assert calls == ["observe", "observe"]
    assert signals.handler is signals.original and signals.timer == (0, 0)
    assert (tmp_path / "readiness-last.png").read_bytes() == last["screenshot"]
    assert (tmp_path / "readiness-last-a11y.xml").read_text() == last["accessibility_tree"]


def test_deadline_escapes_upstream_exception_retry_and_restores_handler(tmp_path, budget):
    clock, signals = budget
    calls = []

    def read():
        calls.append("observe")
        clock.now += signals.timer[0]
        try:
            signals.handler(signals.SIGALRM, None)
        except Exception:
            pytest.fail("deadline must not be swallowed by upstream retry")

    report = {}
    with pytest.raises(driver.VMStartupError):
        driver.wait_for_desktop_ready(SimpleNamespace(_get_obs=read), observation(), output=tmp_path,
                                      report=report, timeout_s=3)
    assert report["desktop_readiness"]["reason"] == "timeout"
    assert report["desktop_readiness"]["elapsed_ms"] == 3000
    assert report["desktop_readiness"]["attempts"] == 2
    assert calls == ["observe"] and signals.handler is signals.original
    assert signals.timer == (0, 0)


def test_active_alarm_is_not_swallowed_or_postponed(tmp_path, budget):
    _clock, signals = budget
    signals.timer = (12, 0)
    env = SimpleNamespace(_get_obs=lambda: pytest.fail("existing alarm conflict"))
    report = {}
    with pytest.raises(driver.VMStartupError):
        driver.wait_for_desktop_ready(env, observation(), output=tmp_path, report=report)
    assert report["desktop_readiness"]["reason"] == "budget_conflict"
    assert signals.armed == [] and signals.timer == (12, 0) and signals.handler is signals.original


def test_observation_error_is_static_and_preserves_last_frame(tmp_path, budget):
    _clock, signals = budget

    def read():
        raise OSError("private connection details")

    report = {}
    with pytest.raises(driver.VMStartupError):
        driver.wait_for_desktop_ready(SimpleNamespace(_get_obs=read), observation(), output=tmp_path,
                                      report=report)
    assert report["desktop_readiness"]["reason"] == "observation_error"
    assert report["desktop_readiness"]["observation_error_type"] == "OSError"
    assert "private connection" not in json.dumps(report)
    assert signals.handler is signals.original


def test_non_posix_unknown_observer_fails_closed_without_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "signal", SimpleNamespace())
    env = SimpleNamespace(_get_obs=lambda: pytest.fail("unbounded observer must not run"))
    report = {}
    with pytest.raises(driver.VMStartupError):
        driver.wait_for_desktop_ready(env, observation(), output=tmp_path, report=report, interval_s=0)
    assert report["desktop_readiness"]["reason"] == "bounded_observer_unavailable"


def test_non_posix_known_rpc_is_bounded_to_remaining_and_restored(monkeypatch):
    from cua_jev.osworld_remote import RemoteOSWorldEnv

    monkeypatch.setattr(driver, "signal", SimpleNamespace())
    env = object.__new__(RemoteOSWorldEnv)
    env.timeout_s = 120
    calls = []
    env._get_obs = lambda: calls.append(env.timeout_s) or {"ready": True}
    assert driver._readiness_observe(env, 7.5) == {"ready": True}
    assert calls == [7.5] and env.timeout_s == 120

    def fail():
        assert env.timeout_s == 3
        raise OSError()

    env._get_obs = fail
    with pytest.raises(OSError):
        driver._readiness_observe(env, 3)
    assert env.timeout_s == 120


def test_gate_failure_is_startup_error_without_agent_input_or_evaluation(tmp_path, monkeypatch, budget):
    _clock, signals = budget
    closed = []
    initial = observation(pointer=True)

    def timeout():
        signals.handler(signals.SIGALRM, None)

    env = SimpleNamespace(
        _get_obs=timeout, close=lambda: closed.append("environment"),
        evaluate=lambda: pytest.fail("readiness failure must not be evaluated"),
    )
    args = SimpleNamespace(output=tmp_path, model="test", vm_path=tmp_path / "base.qcow2",
                           no_a11y=False, pause=0.5, max_steps=5, timeout=60)
    monkeypatch.setattr(driver, "AgentRunner", lambda *_a, **_k: pytest.fail("no agent on black desktop"))
    report = driver.run_episode(
        args, "os/public-goal", {"instruction": "Goal", "evaluator": {"private": "hidden"}}, "vlm",
        env_factory=lambda **_: env, initialize=lambda *_: initial,
        backend_factory=lambda *_a, **_k: pytest.fail("backend must not start"),
        planner_factory=lambda _: SimpleNamespace(close=lambda: closed.append("planner")),
    )
    assert report["run_status"] == report["episode_status"] == "startup_error"
    assert report["evaluation_status"] == "not_run" and report["official_score"] is None
    assert report["cleanup_status"] == "confirmed" and closed == ["environment", "planner"]
    assert report["startup_diagnostics"]["desktop_readiness"]["reason"] == "timeout"
    assert (tmp_path / "os-public-goal/vlm/readiness-last.png").read_bytes() == initial["screenshot"]


@pytest.mark.parametrize("relative,expected", [
    ("os-f9be0997-4b7c-45c5-b05c-4612b44a6118/hybrid", "black_placeholder"),
    ("os-e0df059f-28a6-4169-924f-b9623e7184cc/vlm", "black_placeholder"),
    ("os-f9be0997-4b7c-45c5-b05c-4612b44a6118/vlm", "visual_evidence_present"),
])
def test_recorded_v11_frames_when_local_evidence_is_available(relative, expected):
    folder = ROOT / "runs/osworld-fixed8-v11-docker" / relative
    if not (folder / "initial.png").is_file():
        pytest.skip("recorded ignored run artifacts are not part of the source distribution")
    public = {"screenshot": (folder / "initial.png").read_bytes(),
              "accessibility_tree": (folder / "initial-a11y.xml").read_text(encoding="utf-8")}
    quality = driver.desktop_observation_quality(public)
    assert quality["reason"] == expected
    if expected == "black_placeholder":
        assert quality["foreground_pixels"] == 133
    else:
        assert quality["a11y_nodes"] == 2032


@pytest.mark.parametrize("scenario", ["already_ready", "recovers", "timeout"])
def test_worker_sends_ready_only_after_visual_evidence_and_cleans_failed_startup(
    tmp_path, monkeypatch, budget, scenario,
):
    _clock, signals = budget
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "readiness_worker", ROOT / "scripts/osworld_remote_worker.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    assert worker.wait_for_desktop_ready.__name__ == driver.wait_for_desktop_ready.__name__
    monkeypatch.setattr(worker, "wait_for_desktop_ready", driver.wait_for_desktop_ready)
    project, osworld = tmp_path / "project", tmp_path / "osworld"
    (project / "scripts").mkdir(parents=True)
    examples = osworld / "evaluation_examples/examples/os"
    examples.mkdir(parents=True)
    (examples / "public.json").write_text('{"instruction":"Public task"}', encoding="utf-8")
    args = SimpleNamespace(expected_user="experimenter", osworld_root=osworld,
                           vm_path=tmp_path / "base.qcow2", task="os/public", output=project / "run",
                           asset_cache=None, no_a11y=False, vm_provider="docker", passive_settling=False,
                           lazy_evaluators=False)
    events = []
    good = observation(text="$ Ready")
    initial = good if scenario == "already_ready" else observation(pointer=True)

    def observe():
        events.append("observe")
        if scenario == "timeout":
            signals.handler(signals.SIGALRM, None)
        return good

    env = SimpleNamespace(_get_obs=observe, close=lambda: events.append("environment_close"))
    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(
        getpwuid=lambda _uid: SimpleNamespace(pw_name="experimenter")))
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", SimpleNamespace(DesktopEnv=lambda **_: env))
    monkeypatch.setattr(worker, "__file__", str(project / "scripts/worker.py"))
    monkeypatch.setattr(worker, "parse_args", lambda _argv: args)
    monkeypatch.setattr(worker, "validate_vm_path", lambda path: path)
    monkeypatch.setattr(worker, "load_task", lambda *_: {"instruction": "Public task"})
    monkeypatch.setattr(worker, "source_metadata", lambda *_: {})
    monkeypatch.setattr(worker, "seed_asset_cache", lambda *_a, **_k: [])
    monkeypatch.setattr(worker, "reset_checked", lambda *_: initial)

    @contextmanager
    def context(*_args):
        events.append("context_enter")
        try:
            yield {"actual_provider": "fake"}
        finally:
            events.append("context_close")

    class Session:
        def __init__(self, _env, actual, _output):
            assert actual is good
            events.append("session_start")

        def close(self):
            events.append("session_close")
            env.close()

    class Pipe(io.StringIO):
        def __init__(self, descriptor):
            super().__init__()
            self.descriptor = descriptor

        def fileno(self):
            return self.descriptor

        def close(self):
            self.was_closed = True

    rpc, stdout, stderr = Pipe(3), Pipe(1), Pipe(2)
    # Fake only the worker's OS/sys handles; no descriptor redirects or VM calls.
    monkeypatch.setattr(worker, "os", SimpleNamespace(
        fdopen=lambda *_a, **_k: rpc, dup=lambda value: value, dup2=lambda *_: None,
        getuid=lambda: project.stat().st_uid,
    ))
    monkeypatch.setattr(worker, "sys", SimpleNamespace(
        stdout=stdout, stderr=stderr,
        stdin=io.StringIO(), path=[],
    ))
    monkeypatch.setattr(worker, "signal", SimpleNamespace(SIGTERM=15, signal=lambda *_: None))
    monkeypatch.setattr(worker, "owned_vm_context", context)
    monkeypatch.setattr(worker, "WorkerSession", Session)
    monkeypatch.setattr(worker, "serve", lambda *_: events.append("serve"))
    status = worker.main([])
    frames = [json.loads(line) for line in rpc.getvalue().splitlines()]
    metadata = json.loads((args.output / "metadata.json").read_text(encoding="utf-8"))
    startup_events = [json.loads(line) for line in stderr.getvalue().splitlines()]
    assert startup_events and all(event["event"] == "worker_startup_phase" for event in startup_events)
    assert all(set(event) == {"event", "phase", "elapsed_ms"} for event in startup_events)
    assert "Public task" not in stderr.getvalue() and "screenshot" not in stderr.getvalue()
    assert "worker_startup_phase" not in rpc.getvalue() and stdout.getvalue() == ""
    phases = metadata["startup_timeline"]["phases_ms"]
    assert "environment_close_completed" in phases and "worker_finished" in phases
    assert rpc.was_closed and events[-1] == "context_close"
    assert events.count("environment_close") == 1
    if scenario == "timeout":
        assert status == 1 and [frame["event"] for frame in frames] == ["fatal"]
        assert "session_start" not in events and "serve" not in events
        assert metadata["desktop_readiness"]["status"] == "failed"
        assert "ready_flush_completed" not in phases
        assert (args.output / "readiness-last.png").read_bytes() == initial["screenshot"]
        diagnostic = json.loads((args.output / "worker-error.json").read_text(encoding="utf-8"))
        assert diagnostic["startup_diagnostics"]["desktop_readiness"]["reason"] == "timeout"
    else:
        assert status == 0 and [frame["event"] for frame in frames] == ["ready"]
        assert frames[0]["observation"] == worker.public_observation(good)
        assert metadata["desktop_readiness"]["status"] == "ready"
        assert "ready_flush_completed" in phases
        assert (args.output / "initial.png").read_bytes() == good["screenshot"]
        assert (args.output / "initial-a11y.xml").read_text() == good["accessibility_tree"]
        assert events.count("observe") == (scenario == "recovers")

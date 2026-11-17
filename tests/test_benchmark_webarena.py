from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_agent_browser import browser  # noqa: F401
from test_agent_runner import Planner

from cua_jev.agent_types import AgentPlan
from cua_jev.benchmark_webarena import AnswerPlanner, evaluate_final, run_episode, validate_task_set


def test_source_hashes_support_real_namespace_package(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    from pathlib import Path

    package = tmp_path / "webarena_test_namespace"
    package.mkdir()
    (package / "evaluator.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "webarena_driver_namespace_test", Path(__file__).parents[1] / "scripts/run_webarena_pilot.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assert __import__("webarena_test_namespace").__file__ is None
    assert driver.package_source_hashes("webarena_test_namespace") == {
        "root-0/evaluator.py": hashlib.sha256(b"VALUE = 1\n").hexdigest()}


def test_website_cleanup_is_charged_to_episode_even_if_another_site_fails(monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "webarena_cleanup_test", Path(__file__).parents[1] / "scripts/run_webarena_pilot.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    ticks, calls = iter([10, 12]), []
    monkeypatch.setattr(driver.time, "perf_counter", lambda: next(ticks))

    def failed():
        calls.append("failed")
        raise RuntimeError("test")

    sites = {"one": SimpleNamespace(stop=failed),
             "two": SimpleNamespace(stop=lambda: calls.append("stopped"))}
    row = {"total_wall_ms": 5000, "cleanup_wall_ms": 100}
    assert driver.finish_episode_sites(row, sites) is False
    assert calls == ["failed", "stopped"]
    assert row["total_wall_ms"] == 7000 and row["cleanup_wall_ms"] == 2100
    assert row["site_cleanup_errors"] == [{"site": "one", "error_type": "RuntimeError"}]


def test_task_metadata_never_exposes_evaluator_answers_and_missing_sites_fail():
    config = {"task_id": 7, "sites": ["reddit"], "intent_template_id": 1,
              "eval": {"eval_types": ["program_html"], "reference_answer": "PRIVATE"}}
    metadata = validate_task_set([config], [7], ["reddit"])
    assert metadata == [{"task_id": 7, "sites": ["reddit"],
                         "eval_types": ["program_html"], "intent_template_id": 1}]
    assert "PRIVATE" not in str(metadata)
    with pytest.raises(ValueError, match="not configured"):
        validate_task_set([config], [7], ["gitlab"])
    config["eval"]["reference_answers"] = {"fuzzy_match": ["PRIVATE"]}
    with pytest.raises(ValueError, match="official LLM judge"):
        validate_task_set([config], [7], ["reddit"])


def test_explicit_done_answer_is_not_taken_from_an_intermediate_plan():
    returns = [AgentPlan("intermediate", (), done=False), AgentPlan("Final answer", (), done=True)]
    requests = []
    def plan(goal, *args, **kwargs):
        requests.append(goal)
        return returns.pop(0)
    planner = AnswerPlanner(SimpleNamespace(plan=plan))
    planner.plan("Question", object(), [])
    assert planner.final_answer is None
    planner.plan("Question", object(), [])
    assert planner.final_answer == "Final answer"
    assert all("final user-facing answer in summary" in goal for goal in requests)


def test_final_evaluation_discards_private_messages_and_info():
    calls = []
    def validate(page, messages):
        calls.append(messages)
        return 1.0, True, "PRIVATE evaluator message", {"answer": "PRIVATE"}
    result = evaluate_final(SimpleNamespace(validate=validate), object(), "Observed answer")
    assert calls == [[{"role": "assistant", "message": "Observed answer"}]]
    assert result["evaluator_calls"] == 1 and result["reward"] == 1
    assert "PRIVATE" not in str(result)


@pytest.mark.parametrize("mode", ["vlm", "adaptive_rule"])
def test_evaluator_runs_once_after_agent_and_cannot_change_agent_observations(browser, tmp_path, mode):  # noqa: F811
    events, snapshots = [], []
    class Task:
        viewport = {"width": 500, "height": 320}
        timeout = 3000
        evaluated = False

        def setup(self, page):
            events.append("setup")
            page.set_content('<input aria-label="Name"><input aria-label="City">')
            return "Fill Name and City", {"private": "DO NOT PASS TO AGENT"}

        def validate(self, page, messages):
            self.evaluated = True
            events.append("evaluate")
            assert page.locator("input").evaluate_all("els => els.map(el => el.value)") == ["Ada", "Hangzhou"]
            page.set_content("<h1>PRIVATE evaluator reference page</h1>")
            return 1.0, True, "PRIVATE", {"oracle": "PRIVATE"}

        def teardown(self):
            events.append("teardown")

    task = Task()
    class Model(Planner):
        def plan(self, goal, snapshot, history, **kwargs):
            assert not task.evaluated
            events.append("plan")
            snapshots.append(snapshot)
            result = super().plan(goal, snapshot, history, **kwargs)
            return replace(result, summary="Done" if result.done else result.summary)

    def reset():
        events.append("reset")
        return {"fresh_instance": True}
    row = run_episode(
        browser, lambda **kwargs: task, task_id=7, seed=123, mode=mode, reset=reset,
        allowed_origins=["https://fixture.test"], directory=tmp_path / mode,
        planner_factory=lambda **kwargs: Model(), record_model_exchanges=False,
    )
    assert row["benchmark_success"] and row["official"]["evaluator_calls"] == 1
    assert row["final_answer"] == "Done" and row["reset_completed"]
    assert events[0] == "reset" and events[-2:] == ["evaluate", "teardown"]
    assert all("PRIVATE" not in s.text for s in snapshots)
    assert "PRIVATE" not in str(row)
    assert row["status"] == "unverified"  # official score is separate from a model completion claim


def test_reset_failure_never_creates_a_task_or_fabricates_an_evaluation(tmp_path):
    def reset():
        raise RuntimeError("private infrastructure detail")
    def task_factory(**kwargs):
        pytest.fail("reset failed, task must not start")
    row = run_episode(object(), task_factory, task_id=7, seed=1, mode="vlm", reset=reset,
                      allowed_origins=[], directory=tmp_path / "failed", record_model_exchanges=False)
    assert not row["benchmark_success"] and row["official"] is None
    assert row["status"] == "setup_error" and not row["reset_completed"]
    assert row["infrastructure_error_phase"] == "reset"
    assert "private infrastructure" not in str(row)


def test_task_setup_timeout_records_phase_without_private_error_or_model_calls(tmp_path):
    closed = []
    page = SimpleNamespace(set_default_timeout=lambda value: None)
    context = SimpleNamespace(new_page=lambda: page, close=lambda: closed.append("context"))
    fake_browser = SimpleNamespace(new_context=lambda **kwargs: context)

    def setup(_page):
        raise TimeoutError("private login details must not enter the result")

    task = SimpleNamespace(viewport={}, timeout=10, setup=setup,
                           teardown=lambda: closed.append("task"))
    row = run_episode(fake_browser, lambda **kwargs: task, task_id=7, seed=1, mode="vlm",
                      reset=lambda: {}, allowed_origins=[], directory=tmp_path / "setup-timeout",
                      record_model_exchanges=False)
    assert row["infrastructure_error_type"] == "TimeoutError"
    assert row["infrastructure_error_phase"] == "task_setup"
    assert row["official"] is None and "planner_calls" not in row
    assert closed == ["task", "context"]
    assert "private login" not in str(row)


@pytest.mark.parametrize("setup_fails", [False, True])
def test_trusted_setup_timeout_is_restored_before_agent_or_cleanup(tmp_path, setup_fails):
    events = []
    def target(name, **extra):
        return SimpleNamespace(
            set_default_timeout=lambda value: events.append((name, "general", value)),
            set_default_navigation_timeout=lambda value: events.append((name, "navigation", value)),
            **extra)
    page = target("page")
    context = target("context", new_page=lambda: page, pages=[page],
                     close=lambda: events.append("closed"))

    def setup(_page):
        events.append("setup")
        assert events[-5:-1] == [("context", "general", 60000), ("context", "navigation", 60000),
                                 ("page", "general", 60000), ("page", "navigation", 60000)]
        if setup_fails:
            raise TimeoutError("setup")
        return "task", {}

    task = SimpleNamespace(viewport={}, timeout=10000, setup=setup, teardown=lambda: None)
    row = run_episode(SimpleNamespace(new_context=lambda **kwargs: context), lambda **kwargs: task,
                      task_id=7, seed=1, mode="vlm", reset=lambda: {}, allowed_origins=[],
                      directory=tmp_path / "bounded", record_model_exchanges=False, setup_timeout_s=60)
    index = events.index("setup")
    assert events[index + 1:index + 5] == [
        ("context", "general", 30000), ("context", "navigation", 30000),
        ("page", "general", 10000), ("page", "navigation", 10000)]
    assert events[-1] == "closed"
    # The successful setup fixture intentionally has no browser observation
    # implementation; it must get past setup, with original timeouts restored.
    assert row["infrastructure_error_phase"] == ("task_setup" if setup_fails else "initial_observation")
    assert row["official"] is None

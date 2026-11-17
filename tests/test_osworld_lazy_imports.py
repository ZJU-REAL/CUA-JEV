import importlib
import sys

import pytest

from cua_jev.osworld_lazy_imports import inspect_exports, lazy_evaluator_imports


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    for name in (
        "desktop_env",
        "desktop_env/evaluators",
        "desktop_env/evaluators/metrics",
        "desktop_env/evaluators/getters",
    ):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("")
    base = tmp_path / "desktop_env/evaluators"
    (base / "metrics/__init__.py").write_text(
        "from .unused import heavy\nfrom .first import score\nfrom .second import score\n"
        "def infeasible():\n    pass\n"
    )
    (base / "metrics/unused.py").write_text("raise RuntimeError('unused dependency was imported')\n")
    (base / "metrics/first.py").write_text("def score(value):\n    return 0\n")
    (base / "metrics/second.py").write_text("def score(value):\n    return int(value == 7)\n")
    (base / "getters/__init__.py").write_text("from .value import observe as get_value\n")
    (base / "getters/value.py").write_text("def observe(env):\n    return env['value']\n")
    prior = {k: v for k, v in sys.modules.items() if k == "desktop_env" or k.startswith("desktop_env.")}
    for key in prior:
        del sys.modules[key]
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for key in tuple(sys.modules):
        if key == "desktop_env" or key.startswith("desktop_env."):
            del sys.modules[key]
    sys.modules.update(prior)


def test_lazy_exports_use_exact_original_functions_and_last_import_wins(sdk):
    before = {str(p): p.read_bytes() for p in sdk.rglob("*.py")}
    with lazy_evaluator_imports(sdk) as audit:
        metrics = importlib.import_module("desktop_env.evaluators.metrics")
        getters = importlib.import_module("desktop_env.evaluators.getters")
        assert not audit["loaded_exports"]
        assert metrics.infeasible() is None
        assert "score" in dir(metrics) and "heavy" in metrics.__all__
        value = getters.get_value({"value": 7})
        assert metrics.score(value) == 1
        original = importlib.import_module("desktop_env.evaluators.metrics.second")
        assert metrics.score is original.score
        assert metrics.score(8) == 0
        assert len(audit["loaded_exports"]) == 2
        assert "desktop_env.evaluators.metrics.unused" not in sys.modules
        assert "desktop_env.evaluators.metrics.first" not in sys.modules
        assert audit["evaluator_functions_modified"] is False
        with pytest.raises(AttributeError):
            _ = metrics.no_such_evaluator
    assert {str(p): p.read_bytes() for p in sdk.rglob("*.py")} == before


@pytest.mark.parametrize(
    "source",
    [
        "import os\n",
        "from .value import *\n",
        "from ..value import score\n",
        "from .value import score\nscore = 3\n",
        "from .value import score\nprint('side effect')\n",
        "from .value import score\ndef infeasible():\n    return 1\n",
    ],
)
def test_non_reexport_initializers_are_rejected_without_executing_them(tmp_path, source):
    path = tmp_path / "__init__.py"
    path.write_text(source)
    with pytest.raises(ValueError, match="unsupported"):
        inspect_exports(path)


def test_existing_sdk_cannot_be_replaced_in_a_running_worker(sdk):
    with lazy_evaluator_imports(sdk):
        importlib.import_module("desktop_env.evaluators.metrics")
    with pytest.raises(ValueError, match="fresh worker"):
        with lazy_evaluator_imports(sdk):
            pass

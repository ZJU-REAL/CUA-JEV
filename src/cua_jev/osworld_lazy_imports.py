"""Load the original evaluator functions on demand without editing their code.

OSWorld eagerly imports every application's evaluator, including large image/
audio stacks. On NFS that can exhaust VM startup time before a task starts.
This opt-in loader accepts only simple re-export initializers and the original
empty infeasible function; any other initializer construct is rejected.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import re
import sys
from contextlib import contextmanager
from pathlib import Path

PACKAGES = ("desktop_env.evaluators.metrics", "desktop_env.evaluators.getters")


def inspect_exports(path):
    source = Path(path).read_bytes()
    tree = ast.parse(source, filename=str(path))
    exports = {}
    local = []
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and isinstance(node.module, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", node.module)
            and all(alias.name != "*" for alias in node.names)
        ):
            for alias in node.names:
                exports[alias.asname or alias.name] = (node.module, alias.name)
        elif (
            isinstance(node, ast.FunctionDef)
            and node.name == "infeasible"
            and len(node.body) == 1
            and isinstance(node.body[0], ast.Pass)
            and not node.decorator_list
            and not node.args.args
            and not node.args.posonlyargs
            and not node.args.kwonlyargs
            and node.args.vararg is None
            and node.args.kwarg is None
            and not node.args.defaults
            and not node.args.kw_defaults
            and node.returns is None
        ):
            local.append(node)
        else:
            raise ValueError("unsupported original evaluator initializer; do not rewrite its semantics")
    if not exports:
        raise ValueError("original evaluator initializer has no explicit exports")
    if "infeasible" in exports or len(local) > 1:
        raise ValueError("unsupported overlapping local evaluator definition")
    return exports, local, hashlib.sha256(source).hexdigest()


class _EvaluatorLoader(importlib.abc.Loader):
    def __init__(self, package, path, exports, local, audit):
        self.package, self.path = package, path
        self.exports, self.local, self.audit = exports, local, audit

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.path)
        module.__path__ = [str(self.path.parent)]
        module.__all__ = list(self.exports) + [node.name for node in self.local]
        if self.local:
            exec(
                compile(ast.Module(body=self.local, type_ignores=[]), str(self.path), "exec"), module.__dict__
            )

        def load(name):
            if name not in self.exports:
                raise AttributeError(name)
            submodule, original_name = self.exports[name]
            original = importlib.import_module("." + submodule, self.package)
            value = getattr(original, original_name)
            source = Path(original.__file__).resolve(strict=True)
            if not source.is_relative_to(self.path.parents[3]):
                raise ValueError("evaluator function module escapes selected SDK")
            module.__dict__[name] = value
            self.audit["loaded_exports"].append(
                {
                    "package": self.package,
                    "name": name,
                    "original_module": original.__name__,
                    "original_module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
            return value

        module.__getattr__ = load
        module.__dir__ = lambda: sorted(set(module.__dict__) | set(module.__all__))


class _EvaluatorFinder(importlib.abc.MetaPathFinder):
    def __init__(self, specs, audit):
        self.specs, self.audit = specs, audit

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.specs:
            return None
        file, exports, local = self.specs[fullname]
        loader = _EvaluatorLoader(fullname, file, exports, local, self.audit)
        return importlib.util.spec_from_file_location(
            fullname,
            file,
            loader=loader,
            submodule_search_locations=[str(file.parent)],
        )


@contextmanager
def lazy_evaluator_imports(root):
    """Worker-only opt-in; context must cover setup, execution and final scoring.

    Module cache remains live until this isolated worker exits, just like normal
    imports. This is not a context manager for swapping SDKs in a shared process.
    """
    root = Path(root).resolve(strict=True)
    if "desktop_env" in sys.modules:
        raise ValueError("lazy evaluator imports require a fresh worker process")
    specs, hashes = {}, {}
    for name in PACKAGES:
        file = root.joinpath(*name.split("."), "__init__.py")
        if not file.resolve(strict=True).is_relative_to(root):
            raise ValueError("evaluator initializer escapes the selected SDK")
        exports, local, digest = inspect_exports(file)
        specs[name] = file, exports, local
        hashes[name] = digest
    audit = {
        "version": "original_evaluator_lazy_exports_v1",
        "initializer_sha256": hashes,
        "loaded_exports": [],
        "evaluator_functions_modified": False,
        "claim": "import scheduling only; task setup and final evaluator remain separately required",
    }
    finder = _EvaluatorFinder(specs, audit)
    sys.meta_path.insert(0, finder)
    try:
        yield audit
    finally:
        sys.meta_path.remove(finder)

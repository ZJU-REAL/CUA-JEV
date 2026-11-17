"""Opt-in acquisition order for a strictly recognized owned SDK instance only."""

import ast
import hashlib
import inspect
import stat
import textwrap
from pathlib import Path
from types import CodeType

ORIGINAL_ORDER = "original"
AX_FIRST_ORDER = "ax_then_screenshot_v1"
OBSERVATION_ORDERS = (ORIGINAL_ORDER, AX_FIRST_ORDER)
_SUPPORTED_METHOD = """def _get_obs(self):
    return {
        "screenshot": self.controller.get_screenshot(),
        "accessibility_tree": self.controller.get_accessibility_tree() if self.require_a11y_tree else None,
        "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
        "instruction": self.instruction
    }
"""


class OwnedObservationOrder:
    """Keep original getter as the default and every unsupported fallback path."""

    def __init__(self, env, *, requested=ORIGINAL_ORDER, osworld_root, owner_uid):
        self.env = env
        self.installed = False
        self.report = {
            "version": "owned_supported_SDK_observation_order_v89",
            "requested": requested if requested in OBSERVATION_ORDERS else "invalid",
            "original_order": ["screenshot", "accessibility_tree_if_required", "terminal_if_required"],
            "requested_full_order": ["accessibility_tree", "screenshot"],
            "effective_order": ORIGINAL_ORDER, "support_checked": False,
            "sdk_supported": None, "configuration_supported": None, "installed": False,
            "restored": False, "reason": "default_original",
            "atomic_observation": False,
            "counter_scope": (
                "Installed adapter calls only; uninstalled original reads and HTTP retries are unobserved."
            ),
            "exception_prefix_semantics": (
                "Reordering changes which getters ran before an exception; "
                "no retry or fallback follows a started reordered read."
            ),
            "calls": {"reordered_full": 0, "returned_full": 0, "original_image_only": 0,
                      "original_fallback": 0},
            "reordered_failures": [], "reordered_failure_records_omitted": 0,
            "diagnostic_errors": {},
        }
        if requested == ORIGINAL_ORDER:
            return
        if requested != AX_FIRST_ORDER:
            self.report["reason"] = "invalid_request_original_retained"
            return
        try:
            if not self.supported(osworld_root, owner_uid):
                return
            self.previous = ("_get_obs" in vars(env), vars(env).get("_get_obs"))
            self.original = env._get_obs
            env._get_obs = self.read
            self.installed = True
            self.report.update(installed=True, effective_order=AX_FIRST_ORDER, reason="installed")
        except Exception as error:
            self.report["reason"] = "support_or_installation_unavailable_original_retained"
            self.note_error(error)

    def note_error(self, error):
        name = type(error).__name__
        errors = self.report["diagnostic_errors"]
        errors[name] = errors.get(name, 0) + 1

    def supported(self, osworld_root, owner_uid):
        self.report["support_checked"] = True
        cls = type(self.env)
        if cls.__module__ != "desktop_env.desktop_env" or cls.__name__ != "DesktopEnv":
            self.report.update(sdk_supported=False, reason="unknown_SDK_type_original_retained")
            return False
        if "_get_obs" in vars(self.env):
            self.report.update(sdk_supported=False, reason="instance_override_original_retained")
            return False
        method = self.env._get_obs
        function = getattr(method, "__func__", None)
        if (getattr(method, "__self__", None) is not self.env
                or not inspect.isfunction(function)
                or function.__module__ != "desktop_env.desktop_env"
                or function.__qualname__ != "DesktopEnv._get_obs"):
            self.report.update(sdk_supported=False, reason="unknown_SDK_method_original_retained")
            return False
        root = Path(osworld_root)
        if root.is_symlink():
            self.report.update(sdk_supported=False, reason="SDK_root_symlink_original_retained")
            return False
        root = root.resolve(strict=True)
        if root.stat().st_uid != owner_uid:
            self.report.update(sdk_supported=False, reason="SDK_root_owner_mismatch_original_retained")
            return False
        expected = root / "desktop_env/desktop_env.py"
        current = root
        for part in ("desktop_env", "desktop_env.py"):
            current = current / part
            if current.is_symlink() or current.stat().st_uid != owner_uid:
                self.report.update(sdk_supported=False, reason="SDK_file_owner_or_path_original_retained")
                return False
        info = expected.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000:
            self.report.update(sdk_supported=False, reason="SDK_source_unavailable_original_retained")
            return False
        actual = Path(inspect.getsourcefile(function)).resolve(strict=True)
        if actual != expected.resolve(strict=True) or Path(function.__code__.co_filename).resolve() != actual:
            self.report.update(sdk_supported=False, reason="SDK_method_source_origin_original_retained")
            return False
        source = textwrap.dedent(inspect.getsource(function))
        shape = ast.dump(ast.parse(source), include_attributes=False)
        expected_shape = ast.dump(ast.parse(_SUPPORTED_METHOD), include_attributes=False)
        reference = next(code for code in compile(_SUPPORTED_METHOD, "<supported SDK>", "exec",
                         dont_inherit=True).co_consts if isinstance(code, CodeType))
        live = function.__code__
        bytecode_matches = all(getattr(live, name) == getattr(reference, name) for name in (
            "co_code", "co_consts", "co_names", "co_varnames", "co_argcount",
            "co_posonlyargcount", "co_kwonlyargcount", "co_freevars", "co_cellvars",
        ))
        if shape != expected_shape or not bytecode_matches:
            self.report.update(sdk_supported=False, reason="SDK_method_shape_original_retained")
            return False
        self.report.update(sdk_supported=True, SDK_source_path=str(actual),
            SDK_file_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
            method_AST_sha256=hashlib.sha256(shape.encode()).hexdigest(),
            method_AST_and_live_bytecode_match=True, disk_source_not_loaded_module_identity_proof=True)
        if type(getattr(self.env, "require_a11y_tree", None)) is not bool:
            self.report.update(configuration_supported=False, reason="unknown_AX_flag_original_retained")
            return False
        if getattr(self.env, "require_terminal", None) is not False:
            self.report.update(configuration_supported=False, reason="terminal_mode_original_retained")
            return False
        self.report["configuration_supported"] = True
        return True

    def read(self, *args, **kwargs):
        # Original signature errors and unsupported dynamic flags stay on the original path.
        if (args or kwargs or type(getattr(self.env, "require_a11y_tree", None)) is not bool
                or getattr(self.env, "require_terminal", None) is not False):
            self.report["calls"]["original_fallback"] += 1
            return self.original(*args, **kwargs)
        if not self.env.require_a11y_tree:
            self.report["calls"]["original_image_only"] += 1
            return self.original()
        self.report["calls"]["reordered_full"] += 1
        prefix = []
        stage = "accessibility_tree"
        try:
            prefix.append(stage)
            tree = self.env.controller.get_accessibility_tree()
            stage = "screenshot"
            prefix.append(stage)
            screenshot = self.env.controller.get_screenshot()
            stage = "instruction"
            instruction = self.env.instruction
        except BaseException as error:
            if len(self.report["reordered_failures"]) < 64:
                self.report["reordered_failures"].append({
                    "attempted_getter_prefix": prefix, "exception_stage": stage,
                    "exception_type": type(error).__name__, "original_path_retried": False,
                })
            else:
                self.report["reordered_failure_records_omitted"] += 1
            raise
        self.report["calls"]["returned_full"] += 1
        return {"screenshot": screenshot, "accessibility_tree": tree,
                "terminal": None, "instruction": instruction}

    def restore(self):
        if not self.installed:
            return
        try:
            existed, previous = self.previous
            if existed:
                self.env._get_obs = previous
            else:
                del self.env._get_obs
            self.report["restored"] = True
        except Exception as error:
            self.note_error(error)

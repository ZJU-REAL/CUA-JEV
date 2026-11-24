"""Strict flat current-action declarations mapped to the unchanged parser input.

The explicit V55 wire grants no authority and resolves no current or future UI
target. The surrounding adapter checks the provider envelope; the original
parser, compiler and execution guards must still validate every mapped object.
"""

import copy
import hashlib
import json
import re

FORMAT = "semantic_tools_v55"
FAMILIES = {
    "declare_native_current_action": {"click", "fill", "check", "uncheck", "press"},
    "declare_semantic_current_action": {"click", "fill", "check", "uncheck"},
    "declare_visual_current_action": {"visual_click", "visual_fill"},
    "declare_key_page_action": {"type_text", "hotkey", "key_press", "press", "scroll", "back", "wait"},
}
COMPLETION = "declare_current_completion"  # H1 empty advisory only.
GUARDED = {"type_text", "hotkey", "key_press", "press", "scroll", "wait"}
FAMILY_FIELDS = {
    "declare_native_current_action": {"ref", "value", "button", "click_count"},
    "declare_semantic_current_action": {"target", "value", "button", "click_count",
                                        "preconditions", "postconditions"},
    "declare_visual_current_action": {"box", "box_space", "value", "button", "click_count"},
    "declare_key_page_action": {"value", "preconditions", "postconditions"},
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _nonfinite(_value):
    raise ValueError("non-finite argument JSON")


def _text(value, maximum):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _predicates(value):
    return isinstance(value, list) and len(value) <= 8 and all(isinstance(v, dict) for v in value)


def _wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def decode_flat_current(function_name, arguments, *, max_actions, prepare_repair_token=None):
    """Map explicit known parameters; return original-parser input plus safe hashes.

    The surrounding transport rejects absent/multiple/wrong tools, multiple
    choices and refusals, without content fallback or a second request. This
    function accepts argument data only; it does not execute a tool.
    """
    _require(type(max_actions) is int and 1 <= max_actions <= 12, "original horizon required")
    _require(isinstance(arguments, str) and 0 < len(arguments) <= 64000,
             "bounded declaration argument string required")
    _require(isinstance(function_name, str) and (
             function_name in FAMILIES or (max_actions == 1 and function_name == COMPLETION)),
             "unknown current declaration")
    _require(prepare_repair_token is None or (max_actions == 1
             and isinstance(prepare_repair_token, str)
             and re.fullmatch(r"[0-9a-f]{32}", prepare_repair_token)), "original repair offer required")
    try:
        raw = json.loads(arguments, parse_constant=_nonfinite)
        _wire(raw)  # Reject nested overflow to inf as well as NaN/Infinity tokens.
    except (ValueError, TypeError, RecursionError):
        raise ValueError("invalid declaration argument JSON") from None
    _require(isinstance(raw, dict), "object arguments required")
    if function_name == COMPLETION:
        _require({"summary", "done"} <= set(raw)
                 and set(raw) <= {"summary", "done", "completion_conditions"}
                 and _text(raw["summary"], 1000) and raw["done"] is True,
                 "H1 empty advisory completion parameters required")
        _require("completion_conditions" not in raw or _predicates(raw["completion_conditions"]),
                 "original completion conditions required")
        mapped = {"summary": raw["summary"], "steps": [], "done": True}
        if "completion_conditions" in raw:
            mapped["completion_conditions"] = copy.deepcopy(raw["completion_conditions"])
    else:
        mapped = _map_action(function_name, raw, max_actions, prepare_repair_token)
    return mapped, {
        "response_transport": FORMAT, "tool_name": function_name, "input_authority": False,
        "argument_sha256": hashlib.sha256(arguments.encode()).hexdigest(),
        "mapped_object_sha256": hashlib.sha256(_wire(mapped)).hexdigest(),
        "mapping_applied": True,
    }


def _map_action(name, raw, horizon, repair_token):
    immediate = horizon == 1
    required = {"summary", "id", "done", "operation"} if immediate else {"label", "operation"}
    header = {"summary", "id", "done", "depends_on", "completion_conditions"} if immediate else {"label"}
    allowed = header | {"operation", "action_label", "recovery"} | FAMILY_FIELDS[name]
    if immediate and name == "declare_visual_current_action" and repair_token is not None:
        allowed.add("prepare_repair")
    _require(required <= set(raw) and set(raw) <= allowed,
             "only exact fields of this named current-action contract are permitted")
    _require(_text(raw["summary" if immediate else "label"], 1000 if immediate else 240),
             "original bounded nonempty summary/label required")
    op = raw["operation"]
    _require(isinstance(op, str) and op in FAMILIES[name], "operation does not belong to function")
    if immediate:
        identifier = raw["id"]
        _require((type(identifier) is int and identifier >= 0)
                 or _text(identifier, 64), "original step identifier required")
        _require(type(raw["done"]) is bool, "original advisory done flag required")
        _require("completion_conditions" not in raw or _predicates(raw["completion_conditions"]),
                 "original completion condition array required")
    if "action_label" in raw:
        _require(_text(raw["action_label"], 240), "bounded explicit action label required")
    if immediate and "depends_on" in raw:
        dependencies = raw["depends_on"]
        _require(isinstance(dependencies, list) and len(dependencies) <= 12
                 and all((type(item) is int and item >= 0) or _text(item, 64)
                         for item in dependencies), "original bounded dependency identifiers required")
    if "value" in raw:
        _require(isinstance(raw["value"], str) and len(raw["value"]) <= 2000,
                 "literal value must remain a bounded string")
    if op in {"fill", "visual_fill", "type_text", "hotkey", "key_press", "press", "scroll", "wait"}:
        _require("value" in raw, "this operation requires its explicit value")
    _require(op in {"click", "visual_click"} or not ({"button", "click_count"} & set(raw)),
             "click parameters belong only to clicks")
    if "recovery" in raw:
        _require(isinstance(raw["recovery"], str) and raw["recovery"] in {"observe", "replan"},
                 "original recovery enum required")
    pre, post = raw.get("preconditions", []), raw.get("postconditions", [])
    _require(_predicates(pre) and _predicates(post), "original bounded predicate arrays required")
    semantic = name == "declare_semantic_current_action" or bool(pre or post)
    if name == "declare_native_current_action":
        _require(_text(raw.get("ref"), 128), "actual current ref is explicit")
    elif name == "declare_semantic_current_action":
        _require(isinstance(raw.get("target"), dict) and bool(post),
                 "semantic control requires target and nonempty effect predicates")
    elif name == "declare_visual_current_action":
        _require(isinstance(raw.get("box"), list) and len(raw["box"]) == 4
                 and all(type(v) is int for v in raw["box"]), "explicit current image box required")
        _require(isinstance(raw.get("box_space"), str)
                 and raw["box_space"] in {"image_pixels", "normalized_1000"},
                 "explicit current image coordinate space required")
        _require(not immediate or "action_label" in raw, "H1 visual action label required")
    elif semantic:
        _require(op in GUARDED and bool(pre) and bool(post),
                 "guarded key/page action needs both predicates; back is raw only")
    if immediate and raw.get("completion_conditions"):
        _require(semantic, "raw current-ref/visual inputs do not carry whole-goal completion")
    if "prepare_repair" in raw:
        _require(raw["prepare_repair"] == repair_token and op == "visual_fill"
                 and raw["done"] is False and not raw.get("completion_conditions"),
                 "only offered current visual-fill preparation repair is expressible")
    # Copy only declared mapping fields; unknown ones already caused rejection.
    action = {k: copy.deepcopy(v) for k, v in raw.items()
              if k in FAMILY_FIELDS[name] | {"operation", "recovery"}}
    if "action_label" in raw:
        action["label"] = raw["action_label"]
    if not immediate:
        return {"intent": "act_current", "label": raw["label"], "action": action}
    step = {"id": raw["id"], **action}
    if "depends_on" in raw:
        step["depends_on"] = copy.deepcopy(raw["depends_on"])
    mapped = {"summary": raw["summary"], "steps": [step], "done": raw["done"]}
    for key in ("completion_conditions", "prepare_repair"):
        if key in raw:
            mapped[key] = copy.deepcopy(raw[key])
    return mapped

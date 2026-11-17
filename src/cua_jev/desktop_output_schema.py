"""Optional structural output shapes; the original desktop parser stays authoritative.

Derived from the saved v49 five-intent diagnostic. Provider acceptance does not
prove constrained decoding, grounded actions or task success. No normalization
or input authority is supplied here.
"""

from __future__ import annotations


def object_schema(properties, required=()):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def text(maximum=240, *, nonempty=False):
    return {"type": "string", "maxLength": maximum, **({"minLength": 1} if nonempty else {})}


def reference(name):
    return {"$ref": "#/$defs/" + name}


def root_schema(max_actions: int = 6):
    target_fields = {
        name: text(80 if name == "role" else 240) for name in ("name", "role", "app", "window", "ancestor")
    }
    target_fields["match"] = {"enum": ["exact", "contains"]}
    target = object_schema(target_fields)
    target["anyOf"] = [
        {"required": [name], "properties": {name: {"minLength": 1}}} for name in ("name", "role")
    ]
    ref_target = object_schema({"ref": text(128, nonempty=True), **target_fields}, ["ref"])
    scope = object_schema({key: text() for key in ("app", "window", "ancestor")})
    scope["anyOf"] = [{"required": [key], "properties": {key: {"minLength": 1}}} for key in ("app", "window")]

    def condition_schema(*, current_refs=False):
        ordinary_target = (
            {"anyOf": [reference("target"), reference("ref_target")]} if current_refs else reference("target")
        )
        return {
            "anyOf": [
                object_schema(
                    {
                        "kind": {
                            "enum": [
                                "exists",
                                "absent",
                                "focused",
                                "checked",
                                "unchecked",
                                "selected",
                                "expanded",
                            ]
                        },
                        "target": ordinary_target,
                        "value": {"const": ""},
                    },
                    ["kind", "target"],
                ),
                object_schema(
                    {"kind": {"const": "value_equals"}, "target": ordinary_target, "value": text(2000)},
                    ["kind", "target", "value"],
                ),
                object_schema(
                    {
                        "kind": {"enum": ["text_contains", "window_contains"]},
                        "value": text(2000, nonempty=True),
                        "target": {"type": "null"},
                    },
                    ["kind", "value"],
                ),
            ]
        }

    def conditions(*, nonempty=False, current_refs=False):
        return {
            "type": "array",
            "items": reference("guard_condition" if current_refs else "condition"),
            "maxItems": 8,
            **({"minItems": 1} if nonempty else {}),
        }

    common_action = {
        "ref": text(128),
        "target": reference("target"),
        "value": text(2000),
        "label": text(),
        "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        "box_space": {"enum": ["normalized_1000", "image_pixels"]},
        "button": {"enum": ["left", "right"]},
        "click_count": {"type": "integer", "enum": [1, 2]},
        "preconditions": conditions(),
        "postconditions": conditions(),
        "recovery": {"enum": ["observe", "replan"]},
    }
    # Keep all existing immediate defaults, including explicit null target/box.
    # In particular, current keyboard/page actions need not claim reusable guards.
    common_action["target"] = {"anyOf": [reference("target"), {"type": "null"}]}
    common_action["box"] = {"anyOf": [common_action["box"], {"type": "null"}]}
    action = object_schema(
        {
            "operation": {
                "enum": [
                    "click",
                    "fill",
                    "check",
                    "uncheck",
                    "visual_click",
                    "visual_fill",
                    "hotkey",
                    "key_press",
                    "type_text",
                    "press",
                    "scroll",
                    "back",
                    "wait",
                ]
            },
            **common_action,
        },
        ["operation"],
    )
    literal_edit = {
        "anyOf": [
            object_schema(
                {"operation": {"enum": ["fill", "set"]}, "value": text(2000, nonempty=True)},
                ["operation", "value"],
            ),
            object_schema(
                {"operation": {"enum": ["check", "uncheck"]}, "value": {"const": ""}}, ["operation"]
            ),
        ]
    }
    field_edit = {
        "anyOf": [
            object_schema(
                {
                    "operation": {"const": "fill"},
                    "target": reference("target"),
                    "value": text(2000, nonempty=True),
                    "preconditions": conditions(),
                },
                ["operation", "target", "value"],
            ),
            object_schema(
                {
                    "operation": {"enum": ["check", "uncheck"]},
                    "target": reference("target"),
                    "preconditions": conditions(),
                },
                ["operation", "target"],
            ),
        ]
    }
    commit = object_schema(
        {
            "operation": {"enum": ["click", "hotkey", "key_press", "press"]},
            "target": {"anyOf": [reference("target"), reference("ref_target"), {"type": "null"}]},
            "ref": text(128),
            "value": text(2000),
            "label": text(),
            "button": {"enum": ["left", "right"]},
            "click_count": {"type": "integer", "enum": [1, 2]},
            "preconditions": conditions(current_refs=True),
            "postconditions": conditions(nonempty=True),
            "recovery": {"enum": ["observe", "replan"]},
        },
        ["operation"],
    )
    completion = {"enum": ["stage", "goal"]}
    stages = [
        object_schema(
            {
                "intent": {"const": "pursue_property"},
                "label": text(nonempty=True),
                "scope": reference("scope"),
                "edit": reference("literal_edit"),
                "prepare": reference("action"),
                "confirm": object_schema({"label": text(nonempty=True)}, ["label"]),
                "completion": completion,
                "goal_conditions": conditions(nonempty=True),
                "numeric_finalize": object_schema({"key": {"enum": ["enter", "tab"]}}, ["key"]),
            },
            ["intent", "label", "scope", "edit"],
        ),
        object_schema(
            {
                "intent": {"const": "navigate"},
                "label": text(nonempty=True),
                "scope": reference("scope"),
                "arrival": conditions(nonempty=True),
                "completion": completion,
            },
            ["intent", "label", "scope", "arrival"],
        ),
        object_schema(
            {
                "intent": {"const": "edit_fields"},
                "label": text(nonempty=True),
                "edits": {"type": "array", "items": reference("field_edit"), "maxItems": max_actions},
                "completion": completion,
                "commit": reference("commit"),
            },
            ["intent", "label", "edits"],
        ),
        object_schema(
            {"intent": {"const": "act_current"}, "label": text(nonempty=True), "action": reference("action")},
            ["intent", "label", "action"],
        ),
        object_schema(
            {
                "intent": {"const": "complete"},
                "label": text(nonempty=True),
                "conditions": conditions(nonempty=True),
            },
            ["intent", "label"],
        ),
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "anyOf": stages,
        "$defs": {
            "target": target,
            "ref_target": ref_target,
            "scope": scope,
            "condition": condition_schema(),
            "guard_condition": condition_schema(current_refs=True),
            "action": action,
            "literal_edit": literal_edit,
            "field_edit": field_edit,
            "commit": commit,
        },
    }


def immediate_schema(*, prepare_repair_token: str | None = None):
    """One current action or advisory empty completion, with optional owned repair."""
    import re

    if prepare_repair_token is not None and (
        not isinstance(prepare_repair_token, str)
        or re.fullmatch(r"[0-9a-f]{32}", prepare_repair_token) is None
    ):
        raise ValueError("preparation repair schema requires an opaque current token")
    definitions = root_schema()["$defs"]
    identifier = {"anyOf": [text(64, nonempty=True), {"type": "integer", "minimum": 0}]}
    step = object_schema(
        {
            **definitions["action"]["properties"],
            "id": identifier,
            "depends_on": {"type": "array", "items": identifier, "maxItems": 12},
            "navigation": {"const": False},
        },
        ["id", "operation"],
    )
    properties = {
        "summary": text(1000, nonempty=True),
        "steps": {"type": "array", "items": step, "maxItems": 1},
        "done": {"type": "boolean"},
        "completion_conditions": {"type": "array", "items": reference("condition"), "maxItems": 8},
    }
    if prepare_repair_token is not None:
        properties["prepare_repair"] = {"const": prepare_repair_token}
    result = object_schema(properties, ["summary", "steps", "done"])
    result["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    result["$defs"] = {key: definitions[key] for key in ("target", "condition")}
    return result


def explicit_strategy_root_schema(max_actions: int = 6, *, image_uploaded: bool = True):
    """V51 declares every property strategy and can request a fresh visual call."""
    result = root_schema(max_actions)
    result["type"] = "object"
    property_stage = result["anyOf"][0]
    property_stage["required"].append("numeric_finalize")
    property_stage["properties"]["numeric_finalize"]["properties"]["key"]["enum"].append("none")
    if not image_uploaded:
        action = result["$defs"]["action"]["properties"]
        action["operation"]["enum"] = [
            op for op in action["operation"]["enum"] if not op.startswith("visual_")
        ]
        action["box"] = {"type": "null"}
        action.pop("box_space")
        result["anyOf"].append(object_schema({
            "intent": {"const": "inspect_visual"}, "label": text(nonempty=True),
        }, ["intent", "label"]))
    return result


def completion_policy_root_schema(max_actions: int = 6, *, image_uploaded: bool = True):
    """V52 requires an explicit stopping scope and a property confirmation choice."""
    result = explicit_strategy_root_schema(max_actions, image_uploaded=image_uploaded)
    for stage in result["anyOf"][:3]:
        stage["required"].append("completion")
    property_stage = result["anyOf"][0]
    property_stage["required"].append("confirm")
    property_stage["properties"]["confirm"] = {
        "anyOf": [property_stage["properties"]["confirm"], {"const": "none"}],
    }
    return result


def desktop_response_format(max_actions: int = 6, *, prepare_repair_token: str | None = None,
                            property_contract: str = "property_intent_v49", image_uploaded: bool = True):
    """Build a fresh schema for the effective horizon and runner-owned repair offer."""
    if type(max_actions) is not int or not 1 <= max_actions <= 12:
        raise ValueError("desktop output schema requires an original horizon between 1 and 12")
    if max_actions != 1 and prepare_repair_token is not None:
        raise ValueError("preparation repair schema is available only for one current action")
    if (not isinstance(property_contract, str)
            or property_contract not in {"property_intent_v49", "property_intent_v51", "property_intent_v52"}
            or type(image_uploaded) is not bool):
        raise ValueError("desktop schema requires a supported property contract and image provenance")
    strategy = property_contract == "property_intent_v51" and max_actions > 1
    completion_policy = property_contract == "property_intent_v52" and max_actions > 1
    return {
        "type": "json_schema",
        "json_schema": {
            "name": ("cua_desktop_stage_v52" if completion_policy else
                     "cua_desktop_stage_v51" if strategy else
                     "cua_desktop_immediate_v50" if max_actions == 1 else "cua_desktop_stage_v50"),
            "strict": True,
            "schema": immediate_schema(prepare_repair_token=prepare_repair_token)
            if max_actions == 1
            else completion_policy_root_schema(
                max_actions, image_uploaded=image_uploaded,
            ) if completion_policy
            else explicit_strategy_root_schema(max_actions, image_uploaded=image_uploaded) if strategy
            else root_schema(max_actions),
        },
    }

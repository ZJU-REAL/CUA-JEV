'Ordered semantic stage programs with current-state grounding and shared budgets.'

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .agent_types import AgentCondition, AgentPlan, AgentSnapshot

PROGRAM_INTENT = "stage_program"
PROGRAM_TOOL = "declare_stage_program"
STAGE_TOOLS = {
    "declare_navigation_stage": "navigate",
    "declare_property_stage": "pursue_property",
    "declare_field_edits_stage": "edit_fields",
}
ParseStage = Callable[[dict[str, Any], AgentSnapshot, int], AgentPlan]


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class StageDeclaration:
    id: str
    tool_name: str
    arguments_json: str
    # Once-per-stage fresh admission only. Persistent action conditions remain
    # in the original declaration's edit/step preconditions and input guards.
    entry_conditions: tuple[AgentCondition, ...]
    # An upper permission on the original active step's optional bounded
    # observe recovery. Never replace or upgrade the step's own policy.
    recovery: str
    # Declaration-node/numeric-contingency count for diagnostics only. This is
    # neither a lower bound on actual attempts nor a runtime capacity gate.
    declared_slots: int

    def arguments(self) -> dict[str, Any]:
        return json.loads(self.arguments_json)

    def parser_object(self) -> dict[str, Any]:
        arguments = self.arguments()
        if self.tool_name == "declare_property_stage":
            from .desktop_semantic_tools import _property_from_flat

            arguments = _property_from_flat(arguments)
        return {"intent": STAGE_TOOLS[self.tool_name], **arguments}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool_name,
            "arguments": self.arguments(),
            "entry_conditions": [asdict(item) for item in self.entry_conditions],
            "recovery": self.recovery,
            "declared_slots": self.declared_slots,
            "compiled": False,
        }


@dataclass(frozen=True)
class StageProgram:
    summary: str
    stages: tuple[StageDeclaration, ...]
    max_actions: int
    declaration_sha256: str
    # Original finite canonical model declaration, before optional defaults and
    # predicate normalization. This is trace provenance, never another request.
    # Actual parse_stage_program always supplies it; a manual legacy instance
    # without source remains explicitly unauditable rather than reconstructed.
    declaration_json: str = ""
    # Only the explicit v64 codec binds these separate wire/normalized sources.
    # Legacy parsing and its declaration hash retain their original semantics.
    wire_declaration_json: str = ""
    wire_declaration_sha256: str = ""
    codec_version: str = ""
    codec_source_hashes_json: str = ""
    wire_stage_mapping_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        if self.codec_version == "semantic_tools_v64_program_members_stage_v1":
            return {
                "label": self.summary,
                "stages": [stage.to_dict() for stage in self.stages],
                "max_actions": self.max_actions,
                "declaration_sha256": self.declaration_sha256,
                "source_declaration": json.loads(self.declaration_json),
                "source_declaration_status": "codec_normalized_canonical_json",
                "wire_declaration_json": self.wire_declaration_json,
                "wire_declaration_sha256": self.wire_declaration_sha256,
                "normalized_declaration_json": self.declaration_json,
                "normalized_declaration_sha256": self.declaration_sha256,
                "wire_stage_mapping": json.loads(self.wire_stage_mapping_json),
                "codec_version": self.codec_version,
                "codec_source_hashes": json.loads(self.codec_source_hashes_json),
                "final_review": "original_fresh_whole_goal_review_required",
                "future_stages_are_uncompiled_declarations": True,
            }
        return {
            "label": self.summary,
            "stages": [stage.to_dict() for stage in self.stages],
            "max_actions": self.max_actions,
            "declaration_sha256": self.declaration_sha256,
            "source_declaration": json.loads(self.declaration_json) if self.declaration_json else None,
            "source_declaration_status": "original_canonical_json" if self.declaration_json else "missing",
            "final_review": "original_fresh_whole_goal_review_required",
            "future_stages_are_uncompiled_declarations": True,
        }


@dataclass(frozen=True)
class StageProgramPlan:
    program: StageProgram
    initial_plan: AgentPlan


def _reject_future_current_inputs(value: Any) -> None:
    if isinstance(value, dict):
        _require(not {"ref", "box", "box_space", "prepare", "action"} & set(value),
                 "future_stage_requires_semantic_declarations_without_current_inputs")
        for item in value.values():
            _reject_future_current_inputs(item)
    elif isinstance(value, list):
        for item in value:
            _reject_future_current_inputs(item)


def _field_edit_shape(arguments: dict[str, Any], *, max_actions: int, first: bool) -> int:
    """Check inert syntax; current field uniqueness is checked only on activation."""
    from .agent_planner import _conditions, _target, _text

    required = {"label", "edits", "completion"}
    _require(required <= set(arguments) <= required | {"commit"}, "program_field_stage_fields")
    edits = arguments["edits"]
    _require(type(edits) is list and len(edits) <= max_actions, "program_field_edits_bound")
    _require(bool(edits) or "commit" in arguments, "program_empty_field_stage_requires_commit")
    for item in edits:
        _require(type(item) is dict and {"operation", "target"} <= set(item)
                 <= {"operation", "target", "value", "preconditions"}, "program_field_edit_fields")
        operation = item["operation"]
        _require(type(operation) is str and operation in {"fill", "check", "uncheck"},
                 "program_field_edit_operation")
        _require(("value" in item) == (operation == "fill"), "program_field_edit_value_shape")
        _target(item["target"], path="program.field.target")
        _conditions(item.get("preconditions", []), path="program.field.preconditions")
        if operation == "fill":
            value = _text(item["value"], "program.field.value", 2000, required=True)
            _require(value.isascii() and "\x00" not in value, "program_field_literal_text")
    if "commit" in arguments:
        commit = arguments["commit"]
        allowed = {
            "operation", "target", "ref", "value", "label", "preconditions", "postconditions",
            "button", "click_count", "recovery",
        }
        _require(type(commit) is dict and "operation" in commit and set(commit) <= allowed,
                 "program_commit_fields")
        _require(type(commit["operation"]) is str and commit["operation"]
                 in {"click", "hotkey", "key_press", "press"}, "program_commit_operation")
        # Only an initial current commit may retain the original optional
        # current-reference/derived-effect path. Future commits are semantic,
        # explicit and still pass the original owning-field checks on activation.
        if not first:
            _require("postconditions" in commit and bool(commit["postconditions"]),
                     "future_commit_requires_explicit_effect")
        _conditions(commit.get("postconditions", []), path="program.commit.postconditions")
        conditions = commit.get("preconditions", [])
        _require(type(conditions) is list and len(conditions) <= 8, "program_commit_guard_bound")
        if first and any(type(item) is dict and type(item.get("target")) is dict
                         and "ref" in item["target"] for item in conditions):
            # This initial stage is compiled against the actual request frame
            # before the returned program is accepted or any input is possible.
            _require(type(conditions) is list and len(conditions) <= 8, "program_current_commit_guards")
        else:
            _conditions(conditions, path="program.commit.preconditions")
        if commit.get("target") is not None and not (
            first and type(commit["target"]) is dict and "ref" in commit["target"]
        ):
            _target(commit["target"], path="program.commit.target")
    return len(edits) + int("commit" in arguments)


def parse_stage_program(raw: Any, *, max_actions: int) -> StageProgram:
    """Validate all inert declaration structure, without inventing future state.

    Current executable controls, prepared ownership, field uniqueness, candidate
    limits and effects remain the original parser/compiler/validator's job at
    each activation. No synthetic observation is made for future declarations.
    """
    from .agent_planner import _conditions, _navigation_scope, _text
    from .desktop_semantic_tools import _property_from_flat
    from .desktop_semantics import _bounded_decimal

    _require(type(max_actions) is int and 2 <= max_actions <= 12, "program_requires_original_H6_mode")
    _require(type(raw) is dict and set(raw) == {"label", "stages"}, "program_outer_fields")
    summary = _text(raw["label"], "program.label", 240, required=True)
    items = raw["stages"]
    _require(type(items) is list and 2 <= len(items) <= max_actions, "program_stage_count")
    _json(raw)  # Finite JSON only; trusted horizon is never read from the model.
    stages: list[StageDeclaration] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        _require(type(item) is dict and {"id", "tool", "arguments"} <= set(item)
                 <= {"id", "tool", "arguments", "entry_conditions", "recovery"}, "program_stage_fields")
        identifier = _text(item["id"], "program.stage.id", 64, required=True)
        _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier) is not None
                 and identifier not in seen, "program_unique_ordered_stage_ids")
        tool = item["tool"]
        _require(type(tool) is str and tool in STAGE_TOOLS, "program_semantic_stage_tools_only")
        arguments = item["arguments"]
        _require(type(arguments) is dict and arguments.get("completion") == "stage"
                 and "goal_conditions" not in arguments, "program_members_cannot_claim_whole_goal")
        _text(arguments.get("label"), "program.stage.label", 240, required=True)
        if index:
            _reject_future_current_inputs(arguments)
        if tool == "declare_property_stage":
            mapped = _property_from_flat(arguments)
            _navigation_scope(mapped["scope"])
            numeric = arguments["numeric_finalize_key"] != "none"
            if numeric:
                _require(mapped["edit"]["operation"] in {"fill", "set"}
                         and _bounded_decimal(mapped["edit"].get("value", "")) is not None,
                         "program_numeric_contingency_requires_numeric_literal")
            slots = (1 + int("prepare" in arguments)
                     + int(arguments["confirmation_label"] != "none") + int(numeric))
        elif tool == "declare_navigation_stage":
            _require(set(arguments) == {"label", "scope", "arrival", "completion"},
                     "program_navigation_fields")
            _navigation_scope(arguments["scope"])
            _require(bool(_conditions(arguments["arrival"], path="program.navigation.arrival")),
                     "program_navigation_arrival_required")
            slots = 1
        else:
            slots = _field_edit_shape(arguments, max_actions=max_actions, first=index == 0)
        entry = _conditions(item.get("entry_conditions", []), path="program.stage.entry_conditions")
        recovery = item.get("recovery", "observe")
        _require(type(recovery) is str and recovery in {"observe", "replan"}, "program_recovery_policy")
        stages.append(StageDeclaration(identifier, tool, _json(arguments), entry, recovery, slots))
        seen.add(identifier)
    declaration_json = _json(raw)
    digest = hashlib.sha256(declaration_json.encode("utf-8")).hexdigest()
    return StageProgram(summary, tuple(stages), max_actions, digest, declaration_json)


def activate_program_stage(
    program: StageProgram,
    index: int,
    snapshot: AgentSnapshot,
    *,
    remaining_actions: int,
    parse_stage: ParseStage,
) -> AgentPlan:
    """Compile exactly one stage against its actual current full snapshot.

    parse_stage must call original parse_plan + _validate_desktop_execution_plan
    under the original property_intent_v52 flags, without HTTP or fresh budgets.
    The caller must abandon the entire program on any rejected/unknown stage
    admission. Stage recovery=observe grants no new entry-observation ticket.
    """
    from .desktop_semantics import conditions_hold

    _require(isinstance(program, StageProgram) and type(index) is int
             and 0 <= index < len(program.stages), "program_activation_identity")
    _require(type(remaining_actions) is int and 0 <= remaining_actions <= program.max_actions,
             "program_remaining_original_horizon_invalid")
    stage = program.stages[index]
    _require(snapshot.a11y_complete is True, "program_activation_requires_complete_current_observation")
    _require(conditions_hold(stage.entry_conditions, snapshot), "program_entry_conditions_not_true")
    # Crucially use original H6 mode, not remaining_actions==1 as an accidental
    # H1 parser mode. A remaining value of zero permits only the caller's
    # original read-only completion facts, never an input or renewed horizon.
    plan = parse_stage(stage.parser_object(), snapshot, program.max_actions)
    _require(isinstance(plan, AgentPlan) and plan.reusable and bool(plan.steps) and not plan.done,
             "program_activation_requires_original_reusable_stage")
    _require(not plan.completion_conditions and not plan.complete_property_goal,
             "program_active_stage_cannot_claim_whole_goal")
    return plan


def make_program_plan(
    program: StageProgram,
    snapshot: AgentSnapshot,
    *,
    parse_stage: ParseStage,
) -> StageProgramPlan:
    initial = activate_program_stage(
        program, 0, snapshot, remaining_actions=program.max_actions, parse_stage=parse_stage,
    )
    return StageProgramPlan(program, initial)

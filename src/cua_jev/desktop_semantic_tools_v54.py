"""Opt-in54 generation hints; unchanged53 decoding and parser authority remain required."""

import copy

from .desktop_semantic_tools import (
    _digest,
    _require,
)
from .desktop_semantic_tools import (
    decode_semantic_declaration as decode_ancestor,
)
from .desktop_semantic_tools import (
    encode_semantic_declaration as encode_ancestor,
)
from .desktop_semantic_tools import (
    semantic_stage_instructions as ancestor_instructions,
)

FORMAT = "semantic_tools_v54"
DESCRIPTIONS = {
    "declare_property_stage": (
        "Delegate one requested property or text result through legal current preparation, live "
        "navigation/edit and explicitly required submission. The final field may be unseen only "
        "with current legal navigation candidates; runtime guards still decide every input."
    ),
    "declare_navigation_stage": (
        "Reach one meaningful observable view using legal clicks only. This does not type a search "
        "query or perform the final field edit."
    ),
    "declare_field_edits_stage": (
        "Edit currently known, uniquely grounded fields with an explicitly declared commit when needed. "
        "Retain original field-state and committed-result guards."
    ),
    "declare_current_action": (
        "Declare one grounded current action only when no reusable stage fits. "
        "It grants no future action or completion claim."
    ),
    "declare_completion": (
        "Declare that the entire requested goal is already observed complete. "
        "This is advisory and never official task success."
    ),
}


def semantic_stage_instructions(limit, *, visual_box_space="normalized_1000"):
    from .desktop_property_protocol import _shared

    _require(type(limit) is int and 2 <= limit <= 12, "original multi-action horizon required")
    return (
        "Return exactly ONE inert declaration call, not an action execution. The function name supplies "
        "the stage; arguments MUST NOT contain intent. Functions: declare_property_stage, "
        "declare_navigation_stage, declare_field_edits_stage, declare_current_action, declare_completion. "
        "PRIORITIZE one property declaration for an expressible requested property/text result, covering "
        "navigation/edit/required submission rather than separate menu calls. Otherwise prefer reusable "
        "navigation/current-field edits. Use a current action only when no reusable stage fits; do not force "
        "property on unsuitable tasks. Scopes come from current navigation_scopes, not future "
        "windows/fields.\n"
        "For a suitable single property with a known requested result and observed app scope, declare "
        "property directly; an unseen final field requires actual current legal navigation candidates, "
        "not a separate navigate merely to open its form. Retain meaningful navigate for multi-app or "
        "non-property work.\n"
        "Complete text example: call declare_property_stage with arguments "
        '{"label":"Apply requested text","scope":{"app":"CURRENT_APP"},'
        '"prepare":{"operation":"click","ref":"CURRENT_REF"},"edit_operation":"fill",'
        '"edit_value":"FINAL_TEXT","numeric_finalize_key":"none",'
        '"confirmation_label":"Submit requested text change","completion":"goal"}. '
        "Placeholders are syntax, never grounding. Replace with actual current evidence and requested data. "
        "Property uses label,scope,edit_operation,numeric_finalize_key,confirmation_label,completion; "
        "prepare/goal_conditions optional. fill/set REQUIRE edit_value; check/uncheck MUST OMIT it, even "
        "an empty value. No edit/confirm/numeric_finalize wrappers or extra fields. Creation/renaming is "
        "fill(final text), not selecting an existing same-name option. set supports real "
        "fields/options/on-off. "
        "Nonnumeric text/path/on-off/check/uncheck use numeric_finalize_key=none. Never repair an invalid "
        "strategy by silently changing its value.\n"
        "Numeric enter/tab is optional strategy, not text/form submission: only bounded numeric fill/set "
        "may declare it, reserving ONE original action. Declare this contingency before its field appears; "
        "the key is sent only if the later native checks hold. Offer that ONE key only after an actual "
        "unverified "
        "fill receipt on the SAME unique focused native numeric field with matching pending edit buffer "
        "and different committed value. Fresh recheck owner/field/focus/buffer immediately before key; "
        "verify actual committed value afterward, retaining original confirm/goal checks. A bound field "
        "later becoming unknown/closed needs VLM. No guessed key, implicit refill, buffer-as-persistence or "
        "TTL/time/action extension.\n"
        "confirmation_label is exact none for NO confirmation input; otherwise explicitly name the "
        "confirmation intention with a nonempty label. It is not a button/ref/key. Declare required "
        "submission "
        "before the form opens. Jev selects only fresh legal buttons in the receipt-verified field's unique "
        "observed owner with persistent field/result/closure guards. Unknown owner needs VLM, never "
        "guessing. "
        "prepare is only a needed prerequisite distinct from the final property edit; when a current "
        "field supports the requested edit/check/uncheck, declare property without duplicating that "
        "final edit as prepare. Do not drop a necessary prerequisite to avoid its guards. "
        "One action grammar applies to declare_current_action and prepare: an ordinary current native "
        "click/fill/check/uncheck uses ref, no target or pre/postconditions; it needs no declared "
        "postcondition. A semantic target action has no ref/box and MUST declare nonempty postconditions "
        "plus any required preconditions. Guarded keyboard/type/scroll/wait actions keep their original "
        "explicit pre/postconditions and no ref/target/box. A visual action uses label/box_space/box, "
        "no ref/target/conditions. Do not mix these forms, invent effects or rely on automatic removal "
        "of refs or conditions. "
        "A current prepare is one action object, operation direct, no wrapper/id. Retain the property stage; "
        "for search, prepare value is query and edit_value is final data. Bind the real field after "
        "preparation. Native preparation requires that control's listed operation; otherwise ground a "
        "visual action in the uploaded image. "
        "Visual input including prepare requires the actual uploaded current image; no future coordinates.\n"
        "CONDITIONS: ordinary goal_conditions, arrival and completion conditions keep each predicate's "
        "required fields. Their semantic target permits only name,role,app,window,ancestor,match; never "
        "ref,value,state,operation "
        "or coordinates. Put an expected literal on condition.value. Do not guess future native fields or "
        "identities. Property field/confirmation checks bind to actual later receipt evidence; use optional "
        "goal_conditions for additional observable whole-task evidence, without speculating about that "
        "binding. Preserve every explicit user-required outcome and any declared whole-goal conditions; "
        "never force their omission or treat a local field/form opening as whole-task completion.\n"
        "Current refs remain allowed in the original explicit commit-only current-button/persistent "
        "guard form, resolved to unique observed semantic targets; this separate exception never "
        "applies to goal_conditions or arrival and never predicts future refs.\n"
        "declare_navigation_stage arguments: label,scope,arrival,completion. Navigation selects only "
        "current legal click gestures; declaring an arrival does not type a needed search query. "
        "For necessary text entry, explicitly provide a grounded input, such as prepare in the same "
        "property, keeping its query separate from edit_value. Arrival must be a meaningful "
        "unmet observable final view/state. declare_field_edits_stage arguments: label,edits,completion and "
        "optional explicit commit, using the original field/action/condition grammar. edits=[] still "
        "requires "
        "a current commit guarded by persistent field state. Resolve current refs uniquely; preserve owner "
        "and actual closure evidence. Property/navigation/field edits require completion=stage|goal: goal "
        "finishes the ENTIRE task; stage leaves work. Nonempty goal_conditions remain binding. A field, "
        "click or open form alone is not persistence. declare_current_action arguments: label,action for "
        "one current grounded action. declare_completion arguments: label and optional observable "
        "conditions, "
        "only if the whole task is already finished. A declaration never proves success. "
        f"At most {limit} logical actions under the original budgets; tool_choice is auto, no function "
        "executes and no tool results or internal second request follow.\n"
        + _shared(visual_box_space)
    )


def encode_semantic_declaration(body, *, max_actions, profile, contract, observation,
                                visual_box_space="normalized_1000"):
    """Project only the actual declaration schemas; never send a request or repair output."""
    from .desktop_generation_schema import project_parameters

    ancestor, ancestor_provenance = encode_ancestor(
        body, max_actions=max_actions, profile=profile, contract=contract, observation=observation,
        visual_box_space=visual_box_space,
    )
    encoded = copy.deepcopy(ancestor)
    if max_actions > 1:
        original = ancestor_instructions(max_actions, visual_box_space=visual_box_space)
        system = encoded["messages"][0]["content"]
        _require(system.startswith(original), "original53 instruction prefix required")
        encoded["messages"][0]["content"] = (
            semantic_stage_instructions(max_actions, visual_box_space=visual_box_space)
            + system[len(original):]
        )
    before_parameters = {}
    for tool in encoded["tools"]:
        function = tool["function"]
        name = function["name"]
        before_parameters[name] = _digest(function["parameters"])
        function["parameters"] = project_parameters(
            function["parameters"], effective_max_actions=max_actions, format_tag=FORMAT,
        )
        if max_actions > 1:
            _require(name in DESCRIPTIONS, "original semantic declaration name required")
            function["description"] = DESCRIPTIONS[name]
    provenance = {
        "response_transport": FORMAT,
        "ancestor_response_transport": ancestor_provenance["response_transport"],
        "ancestor_request_provenance": ancestor_provenance,
        "image_uploaded": ancestor_provenance["image_uploaded"],
        "ancestor_declaration_schema_sha256": _digest(ancestor["tools"]),
        "declaration_schema_sha256": _digest(encoded["tools"]),
        "declaration_tools_sha256": _digest(encoded["tools"]),
        "ancestor_parameter_schema_sha256": before_parameters,
        "parameter_schema_sha256": {
            t["function"]["name"]: _digest(t["function"]["parameters"]) for t in encoded["tools"]
        },
        "ancestor_system_sha256": _digest(ancestor["messages"][0]["content"]),
        "projected_system_sha256": _digest(encoded["messages"][0]["content"]),
        "generation_projection_format": FORMAT,
    }
    return encoded, provenance


def decode_semantic_declaration(payload, *, max_actions):
    """Do not normalize arguments; use unchanged53 extraction then the original caller parser."""
    raw, ancestor_provenance = decode_ancestor(payload, max_actions=max_actions)
    return raw, {
        **ancestor_provenance,
        "response_transport": FORMAT,
        "ancestor_response_transport": ancestor_provenance["response_transport"],
        "input_authority": False,
    }

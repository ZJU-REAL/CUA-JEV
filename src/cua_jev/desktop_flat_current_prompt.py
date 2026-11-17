"""Exact wire-only rewrites of complete strong ancestor54 system instructions.

Input is the complete strong H1 system message AFTER the original H1 transport
has appended its exact TOOL_GUIDANCE. Both baseline H1 and H6's H1 recovery use
the same entry point. Unmatched focus, owner, geometry, numeric and completion
instructions are retained; this module does not execute any action or request.

Every replacement must match exactly once in the ORIGINAL input. All other
characters, including optional owner/repair/encoding guidance, are copied as-is.
No fallback rewrite or substring-wide label/ref renaming is permitted here.
"""


LEGACY_TOOL_GUIDANCE = (
    "\nDECLARATION OUTPUT TRANSPORT: Return exactly one call to declare_desktop_stage. "
    "Its arguments must be the one JSON plan object described above. This is an inert "
    "declaration, not tool execution. Do not return extra calls or a separate content answer."
)

FLAT_TOOL_GUIDANCE = (
    "\nDECLARATION OUTPUT TRANSPORT: Return exactly one call to one of "
    "declare_native_current_action, declare_semantic_current_action, "
    "declare_visual_current_action, declare_key_page_action, or declare_current_completion. "
    "Its arguments must be the exact flat named parameters described above and in that "
    "function's schema. These are inert declarations, not tool execution. Do not return "
    "extra calls or a separate content answer. Do not wrap parameters in steps or action."
)

_LEGACY_EXAMPLE_SEMANTIC = (
    '{"summary":"Open a dialog","steps":[{"id":"s1","operation":"click",'
    '"target":{"name":"New","role":"push-button","app":"Editor"},'
    '"postconditions":[{"kind":"exists","target":{"name":"Name","window":"New"}}],'
    '"recovery":"observe"}],"done":false}'
)

_LEGACY_EXAMPLE_KEY = (
    '{"summary":"Open the current application command palette","steps":[{"id":"s1",'
    '"operation":"hotkey","value":"ctrl+shift+p",'
    '"preconditions":[{"kind":"window_contains","value":"Editor"}],'
    '"postconditions":[{"kind":"exists","target":{"name":"Command"}}],'
    '"recovery":"observe"}],"done":false}'
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _wire_replacements(visual_box_space):
    """Exact H1 wire fragments, retaining every unlisted instruction slice."""
    legacy_visual = (
        '{"summary":"Click the visible toolbar control","steps":[{"id":"s1",'
        '"operation":"visual_click","ref":"","label":"Open",'
        '"box_space":"' + visual_box_space + '","box":[100,200,300,280],"depends_on":[]'
        '}],"done":false}'
    )
    old_examples = (
        "Complete JSON examples (schema examples, not targets to copy): "
        + _LEGACY_EXAMPLE_SEMANTIC + ". " + _LEGACY_EXAMPLE_KEY + ". "
        + legacy_visual + ". "
    )
    flat_examples = (
        "Complete named-declaration examples (schema examples, not targets to copy): "
        "declare_semantic_current_action arguments "
        '{"summary":"Open a dialog","id":"s1","done":false,"operation":"click",'
        '"target":{"name":"New","role":"push-button","app":"Editor"},'
        '"postconditions":[{"kind":"exists","target":{"name":"Name","window":"New"}}],'
        '"recovery":"observe"}. '
        "declare_key_page_action arguments "
        '{"summary":"Open the current application command palette","id":"s1",'
        '"done":false,"operation":"hotkey","value":"ctrl+shift+p",'
        '"preconditions":[{"kind":"window_contains","value":"Editor"}],'
        '"postconditions":[{"kind":"exists","target":{"name":"Command"}}],'
        '"recovery":"observe"}. '
        "declare_visual_current_action arguments "
        '{"summary":"Click the visible toolbar control","id":"s1","done":false,'
        '"operation":"visual_click","action_label":"Open",'
        '"box_space":"' + visual_box_space + '","box":[100,200,300,280],"depends_on":[]}. '
    )
    return (
        (
            "immediate_empty_done_wire",
            "entire goal already holds and you can declare empty done with completion_conditions. ",
            "entire goal already holds and you can call declare_current_completion with "
            "done=true and completion_conditions. ",
        ),
        (
            "root_and_action_wire",
            "browser DOM. Return ONLY one JSON object with summary (short string), steps (list), "
            "done (boolean), and optional completion_conditions. No other top-level fields. "
            "done is advisory; an external evaluator determines completion. "
            "Set done=true only if the task is already complete now, with steps=[] and observable "
            "completion_conditions covering the entire goal whenever available. If proposing any "
            "action, done MUST be false. "
            "Return at most 1 steps. Each step has a short STRING id such as s1, operation, "
            "optional ref, value, depends_on, label, target, preconditions, postconditions, recovery, "
            "navigation, button/click_count (click or visual_click only), and box/box_space "
            "(visual fallback only). No other fields; "
            "do not add a top-level reusable flag. ",
            "browser DOM. Return ONLY one named declaration with a flat argument object; "
            "do not return a separate JSON content answer, a steps list, a nested action, intent, "
            "navigation, or reusable. For one proposed action, the shared flat header contains "
            "summary (short string), id (short STRING such as s1), done=false, optional depends_on "
            "and optional completion_conditions. done is advisory; an external evaluator determines "
            "completion. Set done=true only when the entire task already holds now: call "
            "declare_current_completion with summary, done=true and observable completion_conditions "
            "covering the entire goal whenever available; this completion declaration has no id "
            "or action fields and sends no input. Action declarations additionally take operation, "
            "optional action_label and recovery, and ONLY their named family's fields: "
            "declare_native_current_action uses an actual current ref, operation "
            "click/fill/check/uncheck/press and value when needed; no target, box or predicates. "
            "declare_semantic_current_action uses operation click/fill/check/uncheck, target, "
            "nonempty postconditions, optional preconditions and value when needed; no ref or box. "
            "declare_visual_current_action uses operation visual_click/visual_fill, action_label, "
            "box, box_space and value for visual_fill; no ref, target or predicates. "
            "An actually offered prepare_repair remains a direct parameter only on its offered "
            "H1 visual_fill schema; retain the original owner/token/value/deadline rules. "
            "declare_key_page_action uses operation type_text/hotkey/key_press/press/scroll/back/wait, "
            "value when needed, and the original raw omitted/empty predicates or both nonempty "
            "preconditions and postconditions for a guarded action; back is raw only. The focus "
            "and shortcut guard guidance below remains in force. No ref, target or box. "
            "Only click/visual_click allow button/click_count. All fields are direct function "
            "arguments, including the explicit header and the action fields; no nested wrapper. "
            "action_label is the action's descriptive label, not an observed control's name. "
            "Only exact parameters allowed by the selected function may be emitted. ",
        ),
        (
            "semantic_navigation_field_wire",
            "ORDINARY SEMANTIC TARGET SCHEMA (navigation omitted or false): ",
            "ORDINARY SEMANTIC TARGET SCHEMA (declare_semantic_current_action; "
            "navigation is omitted): ",
        ),
        (
            "semantic_empty_ref_wire",
            "Semantic steps require empty ref and no box. ",
            "declare_semantic_current_action omits ref and box entirely; the original "
            "mapped semantic step still has no direct ref or box. ",
        ),
        (
            "completion_header_wire",
            "GOAL COMPLETION: optional top-level completion_conditions uses the same predicate schema ",
            "GOAL COMPLETION: optional flat header completion_conditions uses the same predicate schema ",
        ),
        (
            "completion_empty_steps_wire",
            "user goal, or when the entire goal already holds now with done=true and steps=[]. ",
            "user goal, or when the entire goal already holds now using "
            "declare_current_completion with done=true. ",
        ),
        (
            "completion_empty_done_wire",
            "empty done with these observable final-state conditions. ",
            "declare_current_completion with summary, done=true and these observable "
            "final-state conditions. ",
        ),
        (
            "no_navigation_parameter_wire",
            "Omit navigation or set it to false. Future semantic targets are not valid immediate ",
            "Omit navigation; these flat current declarations have no navigation parameter. "
            "Future semantic targets are not valid immediate ",
        ),
        (
            "keyboard_empty_ref_wire",
            'Desktop keyboard operations: type_text, hotkey, key_press. They require ref="", no target '
            "or box, and semantic preconditions/postconditions. ",
            "Desktop keyboard operations: type_text, hotkey, key_press use "
            "declare_key_page_action. They require omission of ref, target and box, "
            "and semantic preconditions/postconditions. ",
        ),
        (
            "visual_ref_and_action_label_wire",
            'operations visual_click and visual_fill require ref="", a visible target '
            'label. click and visual_click support explicit button="left" or "right" and ',
            "operations visual_click and visual_fill use declare_visual_current_action, "
            "omit ref and describe the visible target with action_label. click and "
            'visual_click support explicit button="left" or "right" and ',
        ),
        (
            "visual_required_action_label_wire",
            "label and an explicit box_space. ",
            "action_label and an explicit box_space. ",
        ),
        (
            "guarded_page_empty_ref_wire",
            "Reusable scroll/wait/press require empty ref, no direct target, and both preconditions and "
            "postconditions. ",
            "Guarded scroll/wait/press use declare_key_page_action, omit ref and direct target, "
            "and retain both preconditions and postconditions. ",
        ),
        (
            "box_omission_wire",
            "Only visual operations may carry box_space or box; null is accepted only for an "
            "inapplicable box. ",
            "Only declare_visual_current_action carries box_space and box, both explicit and "
            "applicable. Other flat families omit these fields entirely, including null box. ",
        ),
        (
            "zero_action_completion_wire",
            "Return no steps only when done is true. ",
            "Return a no-action declare_current_completion only when done is true. ",
        ),
        ("three_examples_wire", old_examples, flat_examples),
        ("declaration_transport_wire", LEGACY_TOOL_GUIDANCE, FLAT_TOOL_GUIDANCE),
    )


def rewrite_h1_wire_prompt(original, *, effective_max_actions, visual_box_space):
    """H1-only exact edit retaining all unmatched original byte sequences.

    The caller must supply its actual complete strong H1 system message,
    not an independently reconstructed prompt. Run only after the existing H1
    declaration adapter has appended LEGACY_TOOL_GUIDANCE. No H6 stage prompt is
    accepted; an H6 recovery actually rendered at H1 uses this same entry point.
    """
    _require(type(effective_max_actions) is int and effective_max_actions == 1,
             "flat H1 prompt requires the actual effective horizon1")
    _require(visual_box_space in {"image_pixels", "normalized_1000"},
             "actual original visual coordinate space required")
    _require(isinstance(original, str) and bool(original), "original full H1 prompt required")
    _require(original.count("IMMEDIATE ACTION CONTRACT: ") == 1,
             "original strong immediate-action contract required exactly once")
    _require(original.endswith(LEGACY_TOOL_GUIDANCE),
             "actual original H1 declaration guidance must be the exact suffix")
    edits = []
    for name, old, new in _wire_replacements(visual_box_space):
        _require(original.count(old) == 1, "missing or repeated exact wire fragment: " + name)
        start = original.index(old)
        edits.append((start, start + len(old), new, name))
    edits.sort()
    _require(all(left[1] <= right[0] for left, right in zip(edits, edits[1:], strict=False)),
             "overlapping exact wire fragments")
    # Build from ORIGINAL slices, not sequential broad substitutions. Every
    # character outside the reviewed wire spans is preserved in the same order.
    parts, cursor = [], 0
    for start, end, new, _name in edits:
        parts.extend((original[cursor:start], new))
        cursor = end
    parts.append(original[cursor:])
    rewritten = "".join(parts)
    _require(all(token not in rewritten for token in (
        "declare_desktop_stage", '"steps":[', "steps=[]", "Return ONLY one JSON object",
        "Each step has a short STRING id", '"label":"Open"',
    )), "unreviewed legacy H1 wire residue")
    return rewritten


_LEGACY_PROPERTY_EXAMPLE = (
    "Complete text example: call declare_property_stage with arguments "
    '{"label":"Apply requested text","scope":{"app":"CURRENT_APP"},'
    '"prepare":{"operation":"click","ref":"CURRENT_REF"},"edit_operation":"fill",'
    '"edit_value":"FINAL_TEXT","numeric_finalize_key":"none",'
    '"confirmation_label":"Submit requested text change","completion":"goal"}. '
)

_PROPERTY_EXAMPLES = (
    "prepare is optional: include it only for a distinct prerequisite that must run before "
    "navigation/editing. A directly applicable checkbox needs no prepare; do not drop a necessary "
    "prerequisite or duplicate the final edit as preparation. Complete no-prepare checkbox example: "
    "call declare_property_stage with arguments "
    '{"label":"Enable automatic backups","scope":{"app":"CURRENT_APP"},'
    '"edit_operation":"check","numeric_finalize_key":"none",'
    '"confirmation_label":"none","completion":"goal"}. '
    "Complete necessary-query example: the requested default export name is notes_export and an "
    "observed current query textbox must be filled before its setting is available. Call "
    "declare_property_stage with arguments "
    '{"label":"Set the default export name","scope":{"app":"CURRENT_APP"},'
    '"prepare":{"operation":"fill","ref":"CURRENT_QUERY_REF","value":"export name"},'
    '"edit_operation":"fill","edit_value":"notes_export","numeric_finalize_key":"none",'
    '"confirmation_label":"Apply the default export name","completion":"goal"}. '
    "The first example applies only when the requested checkbox needs no separate submission. "
    "The second applies only when that current query control actually lists fill and the requested "
    "result requires the declared submission. prepare.value is the necessary query, edit_value the "
    "final user data; bind the actual final field after preparation. These are syntax examples, "
    "never task-specific instructions or a reason to invent a prerequisite, field, effect or owner. "
)

_LEGACY_CURRENT_GRAMMAR = (
    "One action grammar applies to declare_current_action and prepare: an ordinary current native "
    "click/fill/check/uncheck uses ref, no target or pre/postconditions; it needs no declared "
    "postcondition. A semantic target action has no ref/box and MUST declare nonempty postconditions "
    "plus any required preconditions. Guarded keyboard/type/scroll/wait actions keep their original "
    "explicit pre/postconditions and no ref/target/box. A visual action uses label/box_space/box, "
    "no ref/target/conditions. Do not mix these forms, invent effects or rely on automatic removal "
    "of refs or conditions. "
)

_FLAT_CURRENT_GRAMMAR = (
    "Property prepare keeps its original nested action grammar: an ordinary current native "
    "click/fill/check/uncheck uses ref, no target or pre/postconditions; it needs no declared "
    "postcondition. A semantic target action has no ref/box and MUST declare nonempty postconditions "
    "plus any required preconditions. Guarded keyboard/type/scroll/wait actions keep their original "
    "explicit pre/postconditions and no ref/target/box. A visual action uses label/box_space/box, "
    "no ref/target/conditions. Do not mix these forms, invent effects or rely on automatic removal "
    "of refs or conditions. Only the separate current-declaration wire is flat: use exactly one of "
    "declare_native_current_action, declare_semantic_current_action, declare_visual_current_action "
    "or declare_key_page_action with direct label and operation, optional action_label/recovery and "
    "only that family's fields. label is the explicit current-action intention, not a future target. "
    "The original compiler supplies the act_current step id; no id, summary, done, intent, action "
    "wrapper, dependencies or navigation parameter is allowed in these H>1 current declarations. "
    "declare_native_current_action uses current ref, click/fill/check/uncheck/press and value when "
    "needed, no target/box/predicates. declare_semantic_current_action uses click/fill/check/uncheck, "
    "target and nonempty postconditions plus any required preconditions/value, no ref/box. "
    "declare_visual_current_action uses visual_click/visual_fill, explicit box/box_space and value "
    "for visual_fill, no ref/target/predicates. Its explicit label or optional action_label describes "
    "the actual visible object; the original parser retains its existing label rule, with no "
    "invented description. declare_key_page_action uses type_text/hotkey/key_press/press/scroll/back/wait "
    "and value when needed; preserve the original raw omitted/empty predicate shape or both nonempty "
    "preconditions and postconditions for a guarded action, back raw only. Original focus and "
    "shortcut guidance still applies. Only click/visual_click permit button/click_count. "
    "Do not send parameters belonging to another family or an empty legacy ref/target/box placeholder. "
)


def rewrite_h6_wire_prompt(original, *, effective_max_actions, visual_box_space):
    """Change current wire and two property examples, retaining all other text."""
    _require(type(effective_max_actions) is int and 2 <= effective_max_actions <= 12,
             "actual original multi-action horizon required")
    _require(visual_box_space in {"image_pixels", "normalized_1000"},
             "actual original visual coordinate space required")
    _require(isinstance(original, str) and bool(original), "original complete H>1 system required")
    replacements = (
        (
            "current_function_names",
            "declare_navigation_stage, declare_field_edits_stage, "
            "declare_current_action, declare_completion. ",
            "declare_navigation_stage, declare_field_edits_stage, declare_native_current_action, "
            "declare_semantic_current_action, declare_visual_current_action, declare_key_page_action, "
            "declare_completion. ",
        ),
        ("optional_property_prepare_examples", _LEGACY_PROPERTY_EXAMPLE, _PROPERTY_EXAMPLES),
        ("flat_current_and_original_prepare", _LEGACY_CURRENT_GRAMMAR, _FLAT_CURRENT_GRAMMAR),
        (
            "current_argument_header",
            "declare_current_action arguments: label,action for one current grounded action. ",
            "Each named current-action function takes one current grounded action with flat label,operation "
            "and only its declared family fields, as above. Property/navigation/field-edits/completion "
            "declarations retain their original parameters and mappings. ",
        ),
        (
            "original_nested_visual_label",
            f"visual_click/visual_fill use label,box_space={visual_box_space},box=[left,top,right,bottom] ",
            f"Original nested actions, including prepare, use visual_click/visual_fill with "
            f"label,box_space={visual_box_space},box=[left,top,right,bottom] ",
        ),
    )
    edits = []
    for name, old, new in replacements:
        _require(original.count(old) == 1, "missing or repeated exact H>1 wire fragment: " + name)
        start = original.index(old)
        edits.append((start, start + len(old), new))
    edits.sort()
    _require(all(left[1] <= right[0] for left, right in zip(edits, edits[1:], strict=False)),
             "overlapping exact H>1 wire fragments")
    parts, cursor = [], 0
    for start, end, new in edits:
        parts.extend((original[cursor:start], new))
        cursor = end
    parts.append(original[cursor:])
    rewritten = "".join(parts)
    _require("declare_current_action" not in rewritten,
             "unreviewed generic current-action wire residue")
    return rewritten


def native_current_ref_guidance(system):
    """Explicit v64 current-H1 projection; legacy wire rewriting stays unchanged."""
    _require(isinstance(system, str) and system.count("IMMEDIATE ACTION CONTRACT: ") == 1,
             "actual strong H1 system required for native current guidance")
    old_example = (
        "declare_semantic_current_action arguments "
        '{"summary":"Open a dialog","id":"s1","done":false,"operation":"click",'
        '"target":{"name":"New","role":"push-button","app":"Editor"},'
        '"postconditions":[{"kind":"exists","target":{"name":"Name","window":"New"}}],'
        '"recovery":"observe"}. '
    )
    new_example = (
        "declare_native_current_action arguments "
        '{"summary":"Activate the observed current control","id":"s1","done":false,'
        '"operation":"click","ref":"CURRENT_REF"}. '
    )
    old_families = (
        "click/fill/check/uncheck/press and value when needed; no target, box or predicates. "
        "declare_semantic_current_action uses operation click/fill/check/uncheck, target, "
        "nonempty postconditions, optional preconditions and value when needed; no ref or box. "
    )
    new_families = (
        "click/fill/check/uncheck/press and value when needed; no target, box or predicates. "
        "Prefer this family when the intended current action is directly expressible by a listed "
        "ref. It needs no prediction of a future view, control name or role. "
        "declare_semantic_current_action uses operation click/fill/check/uncheck, target, "
        "nonempty postconditions, optional preconditions and value when needed; no ref or box. "
        "Use it for genuinely required effect constraints, keeping every declared condition "
        "binding; do not invent future predicates merely to name a current ref-capable control. "
    )
    for old, new in ((old_example, new_example), (old_families, new_families)):
        _require(system.count(old) == 1, "exact current H1 guidance fragment required")
        system = system.replace(old, new, 1)
    return system

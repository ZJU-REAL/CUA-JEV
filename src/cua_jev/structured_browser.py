"""Structured-first browser observations and freshly bound reactive subgoals."""
from dataclasses import asdict, dataclass, replace

from .agent_browser import _CAPTURE, _COLLECT, _ENRICH, _EXTRACT, BrowserAgentBackend, _Binding
from .agent_types import AgentControl, AgentSnapshot
from .browser_subgoals import BrowserBoundStep, BrowserSubgoalStep
from .desktop_semantics import conditions_hold, matching_controls
from .models import Verification
from .vision import browser_viewport_image

# Enumerate explicit semantic controls independently of the bounded heuristic
# walk. A large number of layout divs must not hide the semantic controls after it.
_COLLECT_STRUCTURED = r'''request => {
    const semantic = [...document.querySelectorAll(request.selector)];
    if (semantic.length > 20000) return {elements: [], complete: false};
    const elements = [...semantic], seen = new Set(semantic);
    const walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_ELEMENT);
    let node, examined = 0;
    while ((node = walker.nextNode())) {
        if (++examined > 5000) return {elements, complete: false};
        if (!seen.has(node) && !node.children.length && node.textContent.trim() &&
            !node.parentElement?.closest(request.selector) && getComputedStyle(node).cursor === 'pointer')
            elements.push(node);
    }
    return {elements, complete: true};
}'''
_SCOPED_ENRICH = '(el,item) => { item = (' + _ENRICH + r''')(el,item);
    if (!item) return null;
    const ancestors = [];
    let parent = el.parentElement;
    for (let depth=0; parent && parent !== document.body && depth<8; depth++,parent=parent.parentElement) {
        let label = parent.getAttribute('aria-label') || '';
        if (parent.tagName === 'FIELDSET') label ||= parent.querySelector(':scope > legend')?.innerText || '';
        if (parent.matches('tr,[role="row"]')) label ||= parent.innerText || '';
        label = label.replace(/\s+/g,' ').trim();
        if (label && label.length <= 240 && !ancestors.includes(label)) ancestors.push(label);
    }
    item.ancestors = ancestors;
    return item;
}'''
_CAPTURE_STRUCTURED = (_CAPTURE.replace(_COLLECT, _COLLECT_STRUCTURED)
                       .replace(_ENRICH, _SCOPED_ENRICH)
                       .replace('return {...item, ref: `e${i}`};',
                                'return {...item, ref: `e${i}`, semantic: el.matches(selector)};'))
_EXTRACT_STRUCTURED = 'el => (' + _SCOPED_ENRICH + ')(el, (' + _EXTRACT + ')(el))'
_FIND_STRUCTURED = r'''request => {
    const extract = (''' + _EXTRACT_STRUCTURED + r''');
    let elements = [...document.querySelectorAll(request.selector)];
    if (elements.length > 20000) throw new Error('semantic control budget exceeded');
    if (!request.semantic) {
        const collected = (''' + _COLLECT_STRUCTURED + r''')(request);
        if (!collected.complete) throw new Error('heuristic candidate scan is incomplete');
        elements = collected.elements;
    }
    const matches = elements.filter(el => {
        const c=extract(el);
        return c && JSON.stringify([c.label,c.tag,c.role,c.input_type,c.href,c.ancestors,c.box]) ===
            JSON.stringify(request.identity);
    });
    if (matches.length !== 1) throw new Error('target identity missing or ambiguous');
    return matches[0];
}'''


@dataclass(frozen=True)
class ScopedBrowserControl(AgentControl):
    def identity(self):
        # Current-frame refs can distinguish otherwise identical visible fields
        # by observed geometry. Reusable semantic queries still require a unique
        # semantic match, and preflight rejects any geometry/state change.
        return (*super().identity(), self.ancestors, self.box)


def _control(raw):
    raw = {k: v for k, v in raw.items() if k != 'semantic'}
    return ScopedBrowserControl(**{**raw, 'options': tuple(raw['options']),
                                   'box': tuple(raw['box']) if raw['box'] else None,
                                   'ancestors': tuple(raw.get('ancestors', ()))})


class StructuredBrowserBackend(BrowserAgentBackend):
    verified_structured_post_observations = True

    def __init__(self, page, **kwargs):
        kwargs.setdefault('pointer_routes', False)
        kwargs.setdefault('efficient_observations', True)
        kwargs.setdefault('extended_controls', True)
        kwargs.setdefault('settle_transitions', True)
        if kwargs["extended_controls"] is not True:
            raise ValueError("structured browser requires semantic control extraction")
        self._requires_image = False
        self._semantic_refs = set()
        self._preserved_conditions = ()
        super().__init__(page, **kwargs)

    def observe(self):
        return self._capture(include_image=self._requires_image)

    def prepare_planning_snapshot(self, snapshot, *, reason):
        # Full visual evidence is acquired for recovery/unknown UI, not every
        # successful structured step. The new snapshot replaces the old frame.
        recovery = reason not in {'initial', 'vlm_every_step', 'plan_exhausted', 'horizon_reached'}
        if snapshot.image is None and (recovery or not snapshot.controls or not snapshot.a11y_complete):
            return self._capture(include_image=True)
        return snapshot

    def _capture_snapshot(self, *, include_image):
        self._assert_scope(self.page.url)
        raw = self.page.evaluate(_CAPTURE_STRUCTURED, self._capture_request)
        image = browser_viewport_image(self.page) if include_image else None
        self._assert_scope(raw['url'])
        if self.page.url != raw['url']:
            raise ValueError('page changed while capturing observation')
        # Explicit controls precede heuristic controls. Native HTML/ARIA
        # capability is rechecked independently at binding time below.
        self._semantic_refs = {c['ref'] for c in raw['controls'] if c.get('semantic') is True}
        return AgentSnapshot(raw['url'], raw['title'], raw['text'],
                             tuple(_control(c) for c in raw['controls']), image=image,
                             scroll=tuple(raw['scroll']), semantic_text=tuple(raw['semantic_text']),
                             a11y_complete=raw['a11y_complete'])

    def _handle(self, target):
        handle = self.page.evaluate_handle(_FIND_STRUCTURED, {
            **self._capture_request, 'identity': target.identity(),
            'semantic': target.ref in self._semantic_refs})
        element = handle.as_element()
        if element is None:
            handle.dispose()
            raise ValueError('target no longer an element')
        raw = element.evaluate(_EXTRACT_STRUCTURED)
        if raw is None or _control({**raw, 'ref': target.ref}) != target:
            element.dispose()
            raise ValueError('target changed immediately before execution')
        return element

    def policy_controls(self, snapshot, candidates):
        # The selector sees bound targets and their guards; it does not need
        # another full-page control dump. Execution still uses the full snapshot.
        refs = set()
        for candidate in candidates:
            binding = self._issued.get(candidate.id)
            if binding is None or binding.serialized != candidate.to_dict():
                raise ValueError('selector candidate was not issued')
            if binding.target is not None:
                refs.add(binding.target.ref)
            for condition in (*binding.step.preconditions, *binding.step.postconditions):
                if condition.target is not None:
                    refs.update(c.ref for c in matching_controls(snapshot, condition.target))
        for condition in self._preserved_conditions:
            refs.update(c.ref for c in matching_controls(snapshot, condition.target))
        return tuple(c for c in snapshot.controls if c.ref in refs)

    def candidate_scope_complete(self, candidates):
        # This proves coverage of the offered semantic targets, NOT completeness
        # of all page content. a11y_complete remains false for truncated heuristics.
        return bool(candidates) and all(
            (b := self._issued.get(c.id)) is not None and b.serialized == c.to_dict()
            and b.target is not None and b.target.ref in self._semantic_refs for c in candidates)

    def compile(self, step, planned, current):
        # Protection survives planner retries and immediate-action fallbacks.
        # A later plan must never silently recapture an already changed value.
        if not conditions_hold(self._preserved_conditions, current):
            return ()
        if isinstance(step, BrowserSubgoalStep):
            if not conditions_hold(step.invariants, current):
                return ()
            self._preserved_conditions = tuple(dict.fromkeys(
                (*self._preserved_conditions, *step.invariants)))
        if not isinstance(step, BrowserSubgoalStep):
            candidates = super().compile(step, planned, current)
            allowed = []
            for candidate in candidates:
                binding = self._issued[candidate.id]
                if any(binding.target in matching_controls(current, c.target)
                       for c in self._preserved_conditions):
                    self._issued.pop(candidate.id, None)
                else:
                    allowed.append(candidate)
            return tuple(allowed)
        if not conditions_hold(step.preconditions, current):
            return ()
        result = []
        for alternative in step.alternatives:
            if not conditions_hold(alternative.preconditions, current):
                continue
            matches = matching_controls(current, alternative.target)
            if len(matches) != 1:
                continue
            target = matches[0]
            if any(target in matching_controls(current, c.target) for c in self._preserved_conditions):
                continue
            # Skip satisfied idempotent edits without another input or selector call.
            if (alternative.operation == 'fill' and target.value == alternative.value
                    or alternative.operation == 'check' and target.checked
                    or alternative.operation == 'uncheck' and not target.checked):
                continue
            actual = BrowserBoundStep(**{
                **asdict(alternative), 'target': alternative.target,
                'preconditions': alternative.preconditions,
                'postconditions': alternative.postconditions, 'ref': target.ref,
                'invariants': step.invariants,
            })
            for candidate in super().compile(actual, current, current):
                # Canonical DOM route; route aliases do not manufacture a choice.
                if candidate.arguments['route'] != 'dom':
                    self._issued.pop(candidate.id, None)
                    continue
                candidate = replace(candidate, intent=step.id)
                self._issued[candidate.id] = _Binding(candidate.to_dict(), actual, current, target, 'dom')
                result.append(candidate)
        return tuple(result)

    def _preflight_matches(self, binding, before):
        if (not conditions_hold(self._preserved_conditions, before)
                or any(binding.target in matching_controls(before, c.target)
                       for c in self._preserved_conditions)):
            return False
        if not isinstance(binding.step, BrowserBoundStep):
            return super()._preflight_matches(binding, before)
        target = self._unique(binding.target.identity(), before)
        return (before.url == binding.snapshot.url and before.title == binding.snapshot.title
                and target == binding.target and conditions_hold(binding.step.preconditions, before)
                and not any(target in matching_controls(before, c.target) for c in binding.step.invariants))

    def _verify_observed_effect(self, binding, before, after, verification):
        if not isinstance(binding.step, BrowserBoundStep) and not self._preserved_conditions:
            return verification
        preserved = conditions_hold(self._preserved_conditions, after)
        outcome = conditions_hold(binding.step.postconditions, after)
        return Verification(verification.passed and preserved, 'agent.browser.subgoal', {
            **verification.details, 'constraints_preserved': preserved,
            'postconditions_met': outcome and preserved, 'task_success': False,
            'before_scope': before.fingerprint(), 'after_scope': after.fingerprint(),
        })

    def execute(self, candidate, observation_id, decision_id):
        binding = self._issued.get(candidate.id)
        self._requires_image = binding is not None and binding.route in {'mouse', 'visual'}
        try:
            return super().execute(candidate, observation_id, decision_id)
        finally:
            self._requires_image = False


from .browser_redirects import RedirectingBrowserBackend  # noqa: E402


class StructuredRedirectingBrowserBackend(StructuredBrowserBackend, RedirectingBrowserBackend):
    """The same structured compiler with the existing per-hop CDP origin guard."""

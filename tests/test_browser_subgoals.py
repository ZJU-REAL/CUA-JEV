import json

import httpx
import pytest
from test_agent_browser import browser, page  # noqa: F401

from cua_jev.agent_planner import MultimodalPlanner, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentControl, AgentPlan, AgentSnapshot, AgentStep
from cua_jev.benchmark_miniwob import PlanOrderPolicy
from cua_jev.browser_control_encoding import compact_browser_snapshot, expand_browser_snapshot, wire_size
from cua_jev.browser_subgoals import parse_browser_subgoals
from cua_jev.structured_browser import StructuredBrowserBackend


def form(page):  # noqa: F811
    page.set_content('''<fieldset><legend>First trip</legend><label>Date<input value="old"></label></fieldset>
    <fieldset><legend>Second trip</legend><label>Date<input value="untouched"></label></fieldset>
    <button onclick="document.querySelector('#status').textContent='First trip saved'">Save first</button>
    <p id="status">Pending</p>''')


def request():
    first = {'name': 'Date', 'role': 'textbox', 'ancestor': 'First trip'}
    return {'summary': 'Change first trip and save, preserving second',
            'preserve': [{'name': 'Date', 'role': 'textbox', 'ancestor': 'Second trip'}],
            'subgoals': [{'id': 'save', 'label': 'Save the first trip date', 'actions': [
                {'operation': 'fill', 'target': first, 'value': 'new'},
                {'operation': 'click', 'target': {'name': 'Save first'},
                 'preconditions': [{'kind': 'value_equals', 'target': first, 'value': 'new'}]}],
                'until': [{'kind': 'text_contains', 'value': 'First trip saved'}]}]}


def test_lossless_browser_encoding_preserves_false_empty_and_missing_fields():
    snapshot = AgentSnapshot('https://example.test', 'Example', 'body', tuple(
        AgentControl(f'e{i}', 'Item', 'input', value='' if i else 'kept',
                     disabled=bool(i % 2), box=None if i else (1, 1, 20, 20)) for i in range(100)))
    original = json.loads(json.dumps(snapshot.to_dict()))
    encoded = compact_browser_snapshot(snapshot)
    assert 'control_defaults' in encoded
    assert expand_browser_snapshot(encoded) == original
    assert wire_size(encoded) < wire_size(original) * .45
    assert snapshot.controls[0].value == 'kept'
    sparse = {'controls': [{'ref': str(i), 'selected': False, **({'value': ''} if i else {})}
                           for i in range(20)]}
    assert expand_browser_snapshot(compact_browser_snapshot(sparse)) == sparse


def test_long_nonedit_fallback_keeps_one_valid_action_and_validates_discarded_tail():
    snapshot = AgentSnapshot('https://fixture.test', 'Form', '', tuple(
        AgentControl(f'e{i}', f'Choice {i}', 'input', input_type='checkbox', role='checkbox',
                     box=(i * 30, 0, i * 30 + 20, 20)) for i in range(7)))
    raw = {'summary': 'Select options', 'done': False, 'steps': [
        {'id': f's{i}', 'operation': 'click', 'ref': f'e{i}'} for i in range(7)]}
    plan = parse_plan(raw, snapshot, max_actions=6, allow_browser_edit_prefix=True,
                      allow_browser_subgoals=True)
    assert len(plan.steps) == 1 and plan.steps[0].ref == 'e0' and not plan.done
    with pytest.raises(ValueError, match='non-edit'):
        parse_plan(raw, snapshot, max_actions=6, allow_browser_edit_prefix=True)
    raw['steps'][-1]['ref'] = 'invented'
    with pytest.raises(ValueError):
        parse_plan(raw, snapshot, max_actions=6, allow_browser_edit_prefix=True,
                   allow_browser_subgoals=True)


def test_subgoal_reuses_one_plan_for_edit_then_save_with_preserved_second_trip(page):  # noqa: F811
    form(page)
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    calls = []

    class Planner:
        def plan(self, goal, snapshot, history, *, max_actions):
            calls.append(max_actions)
            if len(calls) == 1:
                return parse_browser_subgoals(request(), snapshot, max_actions=max_actions)
            return AgentPlan('Observed first trip saved', (), done=True)

    try:
        result = AgentRunner(backend, Planner(), policy=PlanOrderPolicy(), config=AgentConfig(
            allow_actions=True, adaptive_dispatch=True, max_steps=6, max_plans=4,
            defer_immediate_plan_refresh=True)).run('Change first trip only')
        assert result.steps == 2 and result.plan_reuses == 1
        assert result.planner_calls == 2  # one plan, one final goal review; no VLM between inputs
        assert result.status == 'unverified'  # never equate subgoal effect with benchmark success
        assert page.locator('input').evaluate_all('els => els.map(e => e.value)') == ['new', 'untouched']
        assert page.locator('#status').inner_text() == 'First trip saved'
        assert backend.observation_metrics['full'] == 0
    finally:
        backend.close()


def test_protected_field_change_blocks_input_and_does_not_accept_success(page):  # noqa: F811
    form(page)
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        before = backend.observe()
        plan = parse_browser_subgoals(request(), before, max_actions=6)
        candidate, = backend.compile(plan.steps[0], before, before)
        page.locator('input').nth(1).fill('external change')
        receipt = backend.execute(candidate, 'obs', 'decision')
        assert not receipt.success
        assert page.locator('input').first.input_value() == 'old'
        assert not backend.verify(candidate, receipt).passed
    finally:
        backend.close()


def test_postcondition_checks_catch_collateral_change_even_when_click_succeeds(page):  # noqa: F811
    form(page)
    page.locator('input').first.fill('new')
    page.locator('button').evaluate("el => el.onclick = () => {"
        "document.querySelectorAll('input')[1].value='bad'; "
        "document.querySelector('#status').textContent='First trip saved'}")
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        before = backend.observe()
        plan = parse_browser_subgoals(request(), before, max_actions=6)
        candidate, = backend.compile(plan.steps[0], before, before)
        receipt = backend.execute(candidate, 'obs', 'decision')
        assert receipt.success
        verification = backend.verify(candidate, receipt)
        assert not verification.passed
        assert verification.details['constraints_preserved'] is False
        assert verification.details['postconditions_met'] is False
    finally:
        backend.close()


def test_protected_baseline_survives_replanning_and_immediate_fallback(page):  # noqa: F811
    form(page)
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        initial = backend.observe()
        plan = parse_browser_subgoals(request(), initial, max_actions=6)
        assert backend.compile(plan.steps[0], initial, initial)
        protected = next(c for c in initial.controls if 'Second trip' in c.ancestors)
        assert not backend.compile(AgentStep('fallback', 'fill', ref=protected.ref, value='bad'),
                                   initial, initial)
        page.locator('input').nth(1).fill('external change')
        changed = backend.observe()
        # Parser can see the new value, but the backend keeps the original guard.
        replanned = parse_browser_subgoals(request(), changed, max_actions=6)
        assert not backend.compile(replanned.steps[0], changed, changed)
        save = next(c for c in changed.controls if c.label == 'Save first')
        assert not backend.compile(AgentStep('save', 'click', ref=save.ref), changed, changed)
    finally:
        backend.close()


def test_immediate_fallback_still_checks_previous_protected_fields_after_input(page):  # noqa: F811
    form(page)
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        initial = backend.observe()
        plan = parse_browser_subgoals(request(), initial, max_actions=6)
        backend.compile(plan.steps[0], initial, initial)
        page.locator('button').evaluate("el => el.onclick = () => {"
            "document.querySelectorAll('input')[1].value='bad'; "
            "document.querySelector('#status').textContent='First trip saved'}")
        current = backend.observe()
        save = next(c for c in current.controls if c.label == 'Save first')
        candidate, = backend.compile(AgentStep('fallback', 'click', ref=save.ref), current, current)
        receipt = backend.execute(candidate, 'obs', 'decision')
        assert receipt.success
        assert not backend.verify(candidate, receipt).passed
    finally:
        backend.close()


def test_semantic_control_after_5000_layout_nodes_still_binds_without_claiming_full_tree(page):  # noqa: F811
    page.set_content('<main>' + '<div>x</div>' * 6000 +
                     '<button onclick="this.textContent=\'Done\'">Continue</button></main>')
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        before = backend.observe()
        assert not before.a11y_complete
        button, = before.controls
        candidates = backend.compile(AgentStep('click', 'click', ref=button.ref), before, before)
        assert backend.candidate_scope_complete(candidates)
        receipt = backend.execute(candidates[0], 'obs', 'decision')
        assert receipt.success and backend.verify(candidates[0], receipt).passed
        assert page.locator('button').inner_text() == 'Done'
    finally:
        backend.close()


def test_current_ref_distinguishes_identically_labeled_controls_without_visual_fallback(page):  # noqa: F811
    page.set_content('<input type="checkbox"><input type="checkbox"><input type="checkbox">')
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        before = backend.observe()
        assert len(before.controls) == 3
        assert len({c.label for c in before.controls}) == 1
        selected = before.controls[1]
        candidate, = backend.compile(AgentStep('second', 'check', ref=selected.ref), before, before)
        receipt = backend.execute(candidate, 'obs', 'decision')
        assert receipt.success and backend.verify(candidate, receipt).passed
        assert page.locator('input').evaluate_all('els => els.map(e => e.checked)') == [False, True, False]
        # A captured ref cannot silently follow a same-labeled field after layout changes.
        current = backend.observe()
        candidate, = backend.compile(
            AgentStep('third', 'check', ref=current.controls[2].ref), current, current)
        page.locator('input').nth(2).evaluate("el => el.style.marginLeft='100px'")
        assert not backend.execute(candidate, 'obs2', 'decision2').success
        assert page.locator('input').nth(2).is_checked() is False
    finally:
        backend.close()


def test_scoped_duplicate_fields_are_distinguished_and_ambiguous_query_is_rejected(page):  # noqa: F811
    form(page)
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        snapshot = backend.observe()
        assert len({c.identity() for c in snapshot.controls if c.label == 'Date'}) == 2
        raw = request()
        raw['preserve'][0].pop('ancestor')
        with pytest.raises(ValueError, match='uniquely'):
            parse_browser_subgoals(raw, snapshot, max_actions=6)
        raw = request()
        raw['subgoals'][0]['actions'][0]['target']['ancestor'] = 'Second trip'
        with pytest.raises(ValueError, match='protected'):
            parse_browser_subgoals(raw, snapshot, max_actions=6)
    finally:
        backend.close()


def test_future_semantic_target_is_rebound_after_page_controls_change(page):  # noqa: F811
    page.set_content('''<button onclick="document.body.innerHTML='<input aria-label=Name>'">Open</button>''')
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    raw = {'summary': 'Open editor and fill', 'preserve': [], 'subgoals': [
        {'id': 'edit', 'label': 'Set the name', 'actions': [
            {'operation': 'click', 'target': {'name': 'Open'}},
            {'operation': 'fill', 'target': {'name': 'Name'}, 'value': 'Ada'}],
         'until': [{'kind': 'value_equals', 'target': {'name': 'Name'}, 'value': 'Ada'}]}]}
    calls = []
    class Planner:
        def plan(self, goal, snapshot, history, *, max_actions):
            calls.append(1)
            return (parse_browser_subgoals(raw, snapshot, max_actions=max_actions) if len(calls) == 1
                    else AgentPlan('Observed name set', (), done=True))
    try:
        result = AgentRunner(backend, Planner(), policy=PlanOrderPolicy(), config=AgentConfig(
            allow_actions=True, adaptive_dispatch=True, max_steps=6, max_plans=4)).run('Set name')
        assert result.steps == 2 and result.plan_reuses == 1 and len(calls) == 2
        assert page.locator('input').input_value() == 'Ada'
    finally:
        backend.close()


def test_real_planner_transport_encodes_losslessly_and_accepts_subgoal_contract():
    snapshot = AgentSnapshot('https://example.test', 'Page', 'text', tuple(
        AgentControl(f'e{i}', f'Field {i}', 'input', role='textbox', input_type='text') for i in range(50)))
    raw = {'summary': 'Set field', 'preserve': [], 'subgoals': [
        {'id': 'g', 'label': 'Set field', 'actions': [{'operation': 'fill',
          'target': {'name': 'Field 0'}, 'value': 'Ada'}],
         'until': [{'kind': 'value_equals', 'target': {'name': 'Field 0'}, 'value': 'Ada'}]}]}
    def handler(req):
        body = json.loads(req.content)
        state = json.loads(body['messages'][1]['content'][0]['text'])
        assert expand_browser_snapshot(state['snapshot']) == json.loads(json.dumps(snapshot.to_dict()))
        assert 'control_defaults' in state['snapshot']
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(raw)}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        planner = MultimodalPlanner('https://example.test/v1', 'test', client=client,
                                    compact_browser_controls=True, browser_subgoals=True)
        plan = planner.plan('Set field', snapshot, [], max_actions=6)
        assert plan.reusable and len(plan.steps) == 1
        with pytest.raises(ValueError):
            parse_plan(raw, snapshot, max_actions=1, allow_browser_subgoals=True)
        assert planner.calls == 1


def test_unrelated_text_change_does_not_invalidate_guarded_structured_action(page):  # noqa: F811
    form(page)
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        before = backend.observe()
        plan = parse_browser_subgoals(request(), before, max_actions=6)
        candidate, = backend.compile(plan.steps[0], before, before)
        page.locator('#status').evaluate("el => el.textContent = 'Still pending, background update'")
        receipt = backend.execute(candidate, 'obs', 'decision')
        assert receipt.success and backend.verify(candidate, receipt).passed
        assert page.locator('input').first.input_value() == 'new'
        assert page.locator('input').nth(1).input_value() == 'untouched'
    finally:
        backend.close()


def test_missing_or_partial_information_requests_visual_recovery(page):  # noqa: F811
    page.set_content('<canvas width="200" height="100"></canvas>')
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        initial = backend.observe()
        assert initial.image is None and not initial.controls
        recovered = backend.prepare_planning_snapshot(initial, reason='initial')
        assert recovered.image is not None
        assert backend.observation_metrics['full'] == 1
    finally:
        backend.close()


@pytest.mark.parametrize('mutation', ['unknown_field', 'future_dependency', 'absent', 'too_many', 'code'])
def test_subgoal_schema_refuses_unbounded_or_unsupported_declarations(page, mutation):  # noqa: F811
    form(page)
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        raw = request()
        stage = raw['subgoals'][0]
        if mutation == 'unknown_field':
            raw['execute'] = 'anything'
        elif mutation == 'future_dependency':
            stage['depends_on'] = ['future']
        elif mutation == 'absent':
            stage['until'] = [{'kind': 'absent', 'target': {'name': 'Date'}}]
        elif mutation == 'too_many':
            stage['actions'] *= 5
        else:
            stage['actions'][0]['operation'] = 'evaluate'
        with pytest.raises(ValueError):
            parse_browser_subgoals(raw, backend.observe(), max_actions=6)
    finally:
        backend.close()


def test_long_subgoal_plan_validates_suffix_but_only_retains_current_horizon(page):  # noqa: F811
    import copy
    form(page)
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        raw = request()
        stage = raw['subgoals'][0]
        raw['subgoals'] = []
        for i in range(8):
            item = copy.deepcopy(stage)
            item['id'] = f'g{i}'
            item['depends_on'] = [f'g{i-1}'] if i else []
            raw['subgoals'].append(item)
        snapshot = backend.observe()
        plan = parse_browser_subgoals(raw, snapshot, max_actions=6)
        assert [s.id for s in plan.steps] == [f'g{i}' for i in range(6)] and not plan.done
        raw['subgoals'][-1]['actions'][0]['operation'] = 'execute_code'
        with pytest.raises(ValueError):
            parse_browser_subgoals(raw, snapshot, max_actions=6)
    finally:
        backend.close()


def test_selector_context_retains_protected_fields_without_unrelated_page_controls(page):  # noqa: F811
    form(page)
    page.evaluate("document.body.insertAdjacentHTML('beforeend','<button>Unrelated</button>')")
    backend = StructuredBrowserBackend(page, allowed_origins=['https://fixture.test'])
    try:
        snapshot = backend.observe()
        plan = parse_browser_subgoals(request(), snapshot, max_actions=6)
        candidates = backend.compile(plan.steps[0], snapshot, snapshot)
        controls = backend.policy_controls(snapshot, candidates)
        assert [c.label for c in controls] == ['Date', 'Date']
        assert any('Second trip' in c.ancestors for c in controls)
        assert len(snapshot.controls) == 4  # execution evidence was not truncated
    finally:
        backend.close()

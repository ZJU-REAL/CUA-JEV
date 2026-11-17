"""Actual Chromium/Edge integration checks; no network models or external sites."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from cua_jev.agent_browser import BrowserAgentBackend
from cua_jev.agent_types import AgentStep
from cua_jev.models import Channel


@pytest.fixture(scope="module")
def browser():
    api = pytest.importorskip("playwright.sync_api")
    with api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except api.Error:
            try:
                browser = playwright.chromium.launch(channel="msedge", headless=True)
            except api.Error as exc:
                pytest.skip(f"Headless Chromium/Edge unavailable: {exc}")
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    page = browser.new_page(viewport={"width": 900, "height": 700})
    yield page
    page.close()


@pytest.fixture
def site():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            parsed = urlparse(self.path)
            if parsed.path == "/disconnect":
                self.connection.shutdown(2)
                self.connection.close()
                return
            if parsed.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", parse_qs(parsed.query)["to"][0])
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"<title>{self.path}</title><h1>{self.path}</h1>".encode())

        def log_message(self, *args):
            pass

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(2)]
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    yield [f"http://127.0.0.1:{server.server_port}" for server in servers], requests
    for server, thread in zip(servers, threads, strict=True):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def control(snapshot, label):
    return next(c for c in snapshot.controls if c.label == label)


@pytest.mark.parametrize("animation", ["css", "javascript"])
def test_transition_wait_takes_one_final_frame_without_replanning_moving_controls(page, animation):
    page.set_content('''
      <button id="expand">Expand</button><div id="panel" style="height:0"></div>
      <button id="next" onclick="this.textContent='Finished'">Next</button>
    ''')
    if animation == "css":
        page.evaluate('''() => {
          panel.style.transition = 'height 200ms linear';
          expand.onclick = () => panel.style.height = '120px';
        }''')
    else:
        page.evaluate('''() => expand.onclick = () => {
          let height = 0;
          const timer = setInterval(() => {
            height += 10; panel.style.height = height + 'px';
            if (height === 120) clearInterval(timer);
          }, 16);
        }''')
    backend = BrowserAgentBackend(page, settle_transitions=True)
    try:
        before = backend.observe()
        candidate = backend.compile(AgentStep('expand', 'click', ref=control(before, 'Expand').ref),
                                    before, before)[0]
        captures = backend.observation_metrics['full']
        receipt = backend.execute(candidate, 'o1', 'd1')
        assert backend.verify(candidate, receipt).passed
        assert backend.observation_metrics['full'] - captures == 2  # preflight + final, no frame polling
        settling = receipt.output['transition_settling']
        assert settling['status'] == 'observed_quiet'
        assert settling['future_changes_excluded'] is False
        assert page.locator('#panel').evaluate('el => el.clientHeight') == 120
        final, _ = backend.take_verified_post_snapshot(candidate, receipt)
        # A new decision can use this captured frame after the animation; input
        # still performs an independent fresh preflight.
        page.wait_for_timeout(80)
        next_candidate = backend.compile(AgentStep('next', 'click', ref=control(final, 'Next').ref),
                                         final, final)[0]
        next_receipt = backend.execute(next_candidate, 'o2', 'd2')
        assert backend.verify(next_candidate, next_receipt).passed
        assert page.locator('#next').inner_text() == 'Finished'
    finally:
        backend.close()


def test_continuous_changes_exhaust_transition_budget_without_claiming_readiness(page):
    page.set_content('''<button id="start">Start</button><div id="clock"></div>
      <script>
      start.onclick = () => setInterval(() => clock.textContent=String(performance.now()), 10);
      </script>
    ''')
    backend = BrowserAgentBackend(page, settle_transitions=True)
    try:
        _, receipt, _ = run(backend, 'click', label='Start')
        assert receipt.output['transition_settling']['status'] == 'budget_exhausted'
        assert receipt.output['transition_settling']['future_changes_excluded'] is False
        assert 500 <= receipt.output['transition_settling']['elapsed_ms'] < 1500
    finally:
        backend.close()


def test_extended_controls_bind_aria_and_script_styled_widgets_without_task_classes(page):
    page.set_content("""
      <h3 role="tab" aria-expanded="false"
          onclick="this.setAttribute('aria-expanded', 'true'); panel.hidden=false">Topic A</h3>
      <div id="panel" hidden><span style="cursor:pointer">Target word</span></div>
      <div id="result"></div>
      <script>document.querySelector('span').addEventListener('click',
          () => document.querySelector('#result').textContent='Clicked target');</script>
    """)
    legacy = BrowserAgentBackend(page)
    assert not legacy.observe().controls
    legacy.close()
    backend = BrowserAgentBackend(page, extended_controls=True)
    try:
        initial = backend.observe()
        assert [c.label for c in initial.controls] == ["Topic A"]
        assert not initial.controls[0].expanded
        _, _, verified = run(backend, "click", label="Topic A")
        assert verified.passed
        opened = backend.observe()
        assert control(opened, "Topic A").expanded
        assert control(opened, "Target word").tag == "span"
        _, _, verified = run(backend, "click", label="Target word")
        assert verified.passed
        assert page.locator("#result").inner_text() == "Clicked target"
    finally:
        backend.close()


def test_extended_controls_exclude_hidden_sensitive_and_nested_duplicate_targets(page):
    page.set_content("""
      <button style="cursor:pointer"><span>One button</span></button>
      <span style="cursor:pointer" hidden>Hidden</span>
      <section inert><span style="cursor:pointer">Inert</span></section>
      <input type="password" tabindex="0" aria-label="Password">
      <input type="file" tabindex="0" aria-label="File">
      <div role="menuitem" aria-disabled="true">Disabled item</div>
    """)
    backend = BrowserAgentBackend(page, extended_controls=True)
    try:
        snapshot = backend.observe()
        assert [c.label for c in snapshot.controls] == ["One button", "Disabled item"]
        assert backend.compile(AgentStep("x", "click", ref=control(snapshot, "Disabled item").ref),
                               snapshot, snapshot) == ()
    finally:
        backend.close()


def test_extended_scan_bound_cannot_authorize_an_ambiguous_prefix_match(page):
    page.set_content('<div role="menuitem">Same item</div>' + '<p>x</p>' * 5001
                     + '<div role="menuitem">Same item</div>')
    backend = BrowserAgentBackend(page, extended_controls=True)
    try:
        snapshot = backend.observe()
        assert snapshot.a11y_complete is False
        candidate = backend.compile(AgentStep("x", "click", ref=control(snapshot, "Same item").ref),
                                    snapshot, snapshot)[0]
        receipt = backend.execute(candidate, "obs", "dec")
        assert receipt.success is False
        assert "incomplete" in receipt.error
    finally:
        backend.close()


def test_extended_state_change_is_detected_before_any_input(page):
    page.set_content('<h3 role="tab" aria-expanded="false" onclick="window.clicked=true">Topic</h3>')
    backend = BrowserAgentBackend(page, extended_controls=True)
    try:
        snapshot = backend.observe()
        candidate = backend.compile(AgentStep("x", "click", ref=snapshot.controls[0].ref),
                                    snapshot, snapshot)[0]
        page.locator("h3").evaluate("el => el.setAttribute('aria-expanded', 'true')")
        receipt = backend.execute(candidate, "obs", "dec")
        assert not receipt.success and page.evaluate("window.clicked") is None
    finally:
        backend.close()


def test_nearby_visible_single_field_labels_enable_binding_without_using_ordinals(page):
    page.set_content("""
      <div><span>First field:</span><input></div>
      <table><tr><th>Second field:</th><td><input></td></tr></table>
      <section><div>Third field:</div><div><textarea>Old text</textarea></div></section>
    """)
    backend = BrowserAgentBackend(page, extended_controls=True)
    try:
        initial = backend.observe()
        assert [c.label for c in initial.controls] == [
            "Nearby: First field:", "Nearby: Second field:", "Nearby: Third field:",
        ]
        for label, value in (("First", "one"), ("Second", "two"), ("Third", "three")):
            _, _, verified = run(backend, "fill", label=f"Nearby: {label} field:", value=value)
            assert verified.passed
        assert [c.value for c in backend.observe().controls] == ["one", "two", "three"]
        assert [c.label for c in backend.observe().controls] == [c.label for c in initial.controls]
    finally:
        backend.close()


def test_nearby_labels_do_not_disambiguate_shared_or_hidden_context(page):
    page.set_content("""
      <div><span>Shared label</span><input><input></div>
      <section><span hidden>Hidden label</span><input></section>
    """)
    backend = BrowserAgentBackend(page, extended_controls=True)
    try:
        snapshot = backend.observe()
        assert all(c.label == "" for c in snapshot.controls)
        assert backend.compile(AgentStep("x", "fill", ref=snapshot.controls[0].ref, value="value"),
                               snapshot, snapshot) == ()
    finally:
        backend.close()


def run(backend, operation, *, label="", value="", route="dom", box=None):
    snapshot = backend.observe()
    ref = control(snapshot, label).ref if label and box is None else ""
    candidates = backend.compile(
        AgentStep("test", operation, ref=ref, value=value, box=box), snapshot, snapshot
    )
    candidate = next(c for c in candidates if c.arguments["route"] == route)
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success, receipt.error
    return candidate, receipt, backend.verify(candidate, receipt)


def test_observe_extracts_labels_values_and_excludes_sensitive_controls(page):
    page.set_content("""
      <title>Form</title><label for="name">Your name</label><input id="name" value="Ada">
      <span id="desc">Biography</span><textarea aria-labelledby="desc">Math</textarea>
      <select aria-label="Language"><option>English</option><option selected>中文</option></select>
      <label><input type="checkbox" checked>Remember</label>
      <div contenteditable="true" aria-label="Notes">draft</div>
      <input type="password" aria-label="Password"><input type="file" aria-label="Upload">
      <input type="hidden" aria-label="Secret"><button hidden>Hidden</button>
      <div aria-hidden="true"><button>Invisible</button></div>
    """)
    backend = BrowserAgentBackend(page)
    snapshot = backend.observe()
    assert {c.label for c in snapshot.controls} == {"Your name", "Biography", "Language", "Remember", "Notes"}
    assert control(snapshot, "Your name").value == "Ada"
    assert control(snapshot, "Biography").value == "Math"
    assert control(snapshot, "Language").options == ("English", "中文")
    assert control(snapshot, "Language").value == "中文"
    assert control(snapshot, "Remember").checked
    assert control(snapshot, "Notes").role == "textbox"
    assert control(snapshot, "Notes").tag == "contenteditable"
    assert snapshot.image is not None and snapshot.image.region == (0, 0, 900, 700)
    assert "jpeg" not in json.dumps(snapshot.to_dict())
    backend.close()
    assert not page.is_closed()


def test_fill_rebinds_ref_after_dom_order_changes(page):
    page.set_content("""<input id="first" aria-label="First"><input id="second" aria-label="Second">
      <script>first.oninput = () => document.body.prepend(second);</script>""")
    backend = BrowserAgentBackend(page)
    planned = backend.observe()
    first = AgentStep("fill-first", "fill", ref=control(planned, "First").ref, value="one")
    second = AgentStep("fill-second", "fill", ref=control(planned, "Second").ref, value="two")
    candidate = backend.compile(first, planned, planned)[0]
    receipt = backend.execute(candidate, "obs1", "dec1")
    assert receipt.success, receipt.error
    assert backend.verify(candidate, receipt).passed
    current = backend.observe()
    assert control(planned, "Second").ref != control(current, "Second").ref
    assert current.scope_key() == planned.scope_key()
    rebound = backend.compile(second, planned, current)[0]
    assert rebound.arguments["ref"] == control(current, "Second").ref
    assert rebound.intent == "fill-second" and rebound.capability == "agent.browser.fill"
    receipt = backend.execute(rebound, "obs2", "dec2")
    assert receipt.success, receipt.error
    assert page.locator("#first").input_value() == "one"
    assert page.locator("#second").input_value() == "two"
    assert backend.verify(rebound, receipt).passed


@pytest.mark.parametrize("change", ["duplicate", "remove", "rename"])
def test_compile_rejects_missing_or_ambiguous_identity(page, change):
    page.set_content('<input aria-label="Target">')
    backend = BrowserAgentBackend(page)
    planned = backend.observe()
    step = AgentStep("fill", "fill", ref=planned.controls[0].ref, value="hello")
    if change == "duplicate":
        page.evaluate("document.body.append(document.querySelector('input').cloneNode())")
    elif change == "remove":
        page.locator("input").evaluate("el => el.remove()")
    else:
        page.locator("input").evaluate("el => el.setAttribute('aria-label','Other')")
    assert backend.compile(step, planned, backend.observe()) == ()


def test_execution_rejects_changed_fingerprint_and_altered_candidate(page):
    page.set_content('<input aria-label="Name"><p>before</p>')
    backend = BrowserAgentBackend(page)
    snapshot = backend.observe()
    step = AgentStep("fill", "fill", ref=snapshot.controls[0].ref, value="hello")
    candidate = backend.compile(step, snapshot, snapshot)[0]
    page.locator("p").evaluate("el => el.textContent = 'changed'")
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success and "stale" in receipt.error
    assert page.locator("input").input_value() == ""
    assert not backend.verify(candidate, receipt).passed
    snapshot = backend.observe()
    candidate = backend.compile(step, snapshot, snapshot)[0]
    altered = replace(candidate, arguments={**candidate.arguments, "value": "evil"})
    receipt = backend.execute(altered, "obs", "dec")
    assert not receipt.success and "altered" in receipt.error
    assert page.locator("input").input_value() == ""


def test_checkbox_select_contenteditable_press_scroll_and_wait(page):
    page.set_content("""<label><input type="checkbox">Remember</label>
      <select aria-label="Language"><option value="en">English</option>
      <option value="zh">Chinese</option></select>
      <div contenteditable="true" aria-label="Notes">draft</div>
      <input aria-label="Name"
      onkeydown="if(event.key==='Enter')document.querySelector('p').textContent='Saved'">
      <p>Pending</p><div style="height:2000px">Long page</div>""")
    backend = BrowserAgentBackend(page)
    _, receipt, verification = run(backend, "check", label="Remember")
    assert verification.passed and receipt.output["postcondition"]["checked"] is True
    assert page.locator("input[type=checkbox]").is_checked()
    _, receipt, verification = run(backend, "uncheck", label="Remember")
    assert verification.passed and receipt.output["postcondition"]["checked"] is False
    assert not page.locator("input[type=checkbox]").is_checked()
    _, receipt, verification = run(backend, "select", label="Language", value="Chinese")
    assert verification.passed and receipt.output["postcondition"]["value"] == "zh"
    assert page.locator("select").input_value() == "zh"
    assert run(backend, "fill", label="Notes", value="updated")[2].passed
    assert page.locator("[contenteditable]").inner_text() == "updated"
    assert run(backend, "press", label="Name", value="Enter")[2].passed
    assert page.locator("p").inner_text() == "Saved"
    assert run(backend, "scroll", value="down", route="page")[2].passed
    assert page.evaluate("scrollY") > 0
    assert run(backend, "wait", value="250", route="page")[2].details["effects"] == ["time_elapsed"]


def test_mouse_route_is_script_and_verifies_value(page):
    page.set_content('<input aria-label="Name">')
    backend = BrowserAgentBackend(page)
    candidate, receipt, verification = run(backend, "fill", label="Name", value="Ada", route="mouse")
    assert candidate.channel == Channel.SCRIPT
    assert candidate.id.endswith(":mouse")
    assert verification.passed and verification.details["task_success"] is False
    assert receipt.output["effects"] == ["value"]
    assert receipt.output["postcondition"] == {
        "identity": ["Name", "input", "textbox", "", ""],
        "value": "Ada",
        "checked": False,
    }
    assert page.locator("input").input_value() == "Ada"
    assert "jpeg" not in json.dumps(receipt.to_dict())


def test_visual_click_and_fill_and_stale_pixels(page):
    page.set_content("""<canvas width="180" height="80"
      onclick="document.querySelector('p').textContent='Clicked'"></canvas>
      <p>Pending</p><input aria-label="Name"><script>
      const ctx = document.querySelector('canvas').getContext('2d');
      ctx.fillStyle='red';ctx.fillRect(0,0,180,80);
      </script>""")
    backend = BrowserAgentBackend(page)
    snapshot = backend.observe()
    step = AgentStep("canvas", "visual_click", box=(10, 15, 200, 110), label="red canvas")
    candidate = backend.compile(step, snapshot, snapshot)[0]
    assert candidate.arguments["route"] == "visual"
    page.evaluate("ctx.fillStyle='blue';ctx.fillRect(0,0,180,80)")
    current = backend.observe()
    assert snapshot.fingerprint() == current.fingerprint()
    assert backend.compile(step, snapshot, current) == ()
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success and "stale screenshot" in receipt.error
    assert page.locator("p").inner_text() == "Pending"
    candidate = backend.compile(step, current, current)[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert receipt.success, receipt.error
    assert backend.verify(candidate, receipt).passed
    assert page.locator("p").inner_text() == "Clicked"
    snapshot = backend.observe()
    box = control(snapshot, "Name").box
    assert run(backend, "visual_fill", box=box, value="Grace", route="visual")[2].passed
    assert page.locator("input").input_value() == "Grace"


def test_visual_rejects_viewport_change_and_password_fill(page):
    page.set_content('<input type="password" aria-label="Password">')
    backend = BrowserAgentBackend(page)
    snapshot = backend.observe()
    step = AgentStep("visual", "visual_fill", value="secret", box=(10, 10, 180, 40))
    candidate = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success
    assert page.locator("input").input_value() == ""
    page.set_viewport_size({"width": 800, "height": 700})
    assert backend.compile(step, snapshot, backend.observe()) == ()


@pytest.mark.parametrize(
    "operation,value",
    [("press", "Control+L"), ("scroll", "diagonal"), ("wait", "99999"), ("evaluate", "alert(1)")],
)
def test_unknown_operations_or_keys_produce_no_candidate(page, operation, value):
    page.set_content('<input aria-label="Name">')
    backend = BrowserAgentBackend(page)
    snapshot = backend.observe()
    ref = snapshot.controls[0].ref if operation == "press" else ""
    assert backend.compile(AgentStep("invalid", operation, ref=ref, value=value), snapshot, snapshot) == ()


def test_receipt_success_alone_is_not_a_verified_effect(page):
    page.set_content("<button>No effect</button>")
    backend = BrowserAgentBackend(page, pointer_routes=False)
    # Remove focus/active styling so the second click has no visible effect.
    page.locator("button").focus()
    _, receipt, verification = run(backend, "click", label="No effect")
    assert receipt.success and not verification.passed


def test_scope_filters_links_blocks_navigation_and_allows_subresources(page, site):
    (allowed, outside), requests = site
    page.goto(allowed + "/start")
    page.set_content(f'''<a href="{outside}/blocked-link">Outside</a>
      <button onclick="location.href='{outside}/blocked-script'">Navigate</button>''')
    backend = BrowserAgentBackend(page)
    page.evaluate(
        "url => {const image = new Image(); image.src = url; document.body.append(image);}",
        outside + "/resource",
    )
    page.wait_for_load_state("networkidle")
    snapshot = backend.observe()
    assert "/resource" in requests
    step = AgentStep("outside", "click", ref=control(snapshot, "Outside").ref)
    assert backend.compile(step, snapshot, snapshot) == ()
    step = AgentStep("navigate", "click", ref=control(snapshot, "Navigate").ref)
    candidate = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success
    assert "/blocked-script" not in requests


def test_back_scope_and_extra_origin_preserve_start_origin(page, site):
    (allowed, outside), requests = site
    page.goto(outside + "/old")
    page.goto(allowed + "/new")
    backend = BrowserAgentBackend(page)
    snapshot = backend.observe()
    candidate = backend.compile(AgentStep("back", "back"), snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success
    assert requests.count("/old") == 1
    backend.close()
    page.goto(allowed + "/again")
    backend = BrowserAgentBackend(page, allowed_origins=[outside])
    assert allowed in backend.allowed_origins and outside in backend.allowed_origins
    page.goto(outside + "/permitted")
    assert backend.observe().url == outside + "/permitted"


def test_same_origin_back_verifies_navigation(page, site):
    (allowed, _), _requests = site
    page.goto(allowed + "/first")
    page.goto(allowed + "/second")
    backend = BrowserAgentBackend(page)
    assert run(backend, "back", route="page")[2].passed
    assert page.url == allowed + "/first"


def test_page_keyboard_radio_and_blank_initialization(page, site):
    (allowed, _), _requests = site
    backend = BrowserAgentBackend(page, allowed_origins=[allowed])
    page.goto(allowed + "/initial")
    page.set_content("""<label><input type="radio" name="choice">A</label>
      <label><input type="radio" name="choice">B</label>
      <input aria-label="First"><input aria-label="Second">""")
    assert run(backend, "check", label="B")[2].passed
    snapshot = backend.observe()
    radio = control(snapshot, "B")
    assert radio.role == "radio" and radio.checked
    assert backend.compile(AgentStep("uncheck", "uncheck", ref=radio.ref), snapshot, snapshot) == ()
    page.locator('[aria-label="First"]').focus()
    assert run(backend, "press", value="Tab", route="page")[2].passed
    assert page.locator('[aria-label="Second"]').evaluate("el => el === document.activeElement")


def test_visual_fill_cannot_type_into_previously_focused_field_from_empty_point(page):
    page.set_content('<input aria-label="Name">')
    page.locator("input").focus()
    backend = BrowserAgentBackend(page)
    snapshot = backend.observe()
    step = AgentStep("visual", "visual_fill", value="wrong", box=(500, 500, 600, 600))
    candidate = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success
    assert page.locator("input").input_value() == ""


def test_page_tab_detects_focus_between_identically_labelled_controls(page):
    page.set_content("<input><input>")
    page.locator("input").first.focus()
    backend = BrowserAgentBackend(page)
    _, _, verification = run(backend, "press", value="Tab", route="page")
    assert verification.passed and "focus" in verification.details["effects"]
    assert page.locator("input").last.evaluate("el => el === document.activeElement")


def test_receipt_reports_measured_action_and_observation_stage_timings(page):
    page.set_content("<p>Ready</p>")
    backend = BrowserAgentBackend(page)
    _, receipt, verification = run(backend, "wait", value="250", route="page")
    timings = receipt.output["timings_ms"]
    assert set(timings) == {"preflight", "action", "effect_check"}
    assert all(value > 0 for value in timings.values())
    assert timings["action"] >= 250
    assert sum(timings.values()) <= receipt.duration_ms + 5
    assert verification.passed


def test_fill_rejects_role_textbox_without_editable_element(page):
    page.set_content('<div role="textbox" aria-label="Read only">Text</div>')
    backend = BrowserAgentBackend(page)
    snapshot = backend.observe()
    step = AgentStep("fill", "fill", ref=control(snapshot, "Read only").ref, value="updated")
    assert backend.compile(step, snapshot, snapshot) == ()


def test_semantic_scope_survives_control_reflow_but_detects_new_error(page):
    page.set_content("""<main><button id="first">Save draft</button> <button id="second">Submit</button>
      <p>Review   the order</p><input aria-label="Name"></main>
      <div hidden>Hidden state</div><div inert>Inert state</div>
      <div aria-hidden="true">Decorative state</div>
      <div style="display:none">Display hidden state</div>
      <div style="visibility:hidden">Visibility hidden state</div>
      <script>const internalState = 'not observation text';</script>
      <style>button {color:black}</style><noscript>Enable scripts</noscript>""")
    backend = BrowserAgentBackend(page)
    before = backend.observe()
    assert before.semantic_text == ("Review the order",)
    assert "semantic_text" not in before.to_dict()
    page.evaluate("""() => {
        const first = document.querySelector('#first');
        const second = document.querySelector('#second');
        first.style.display = second.style.display = 'block';
        document.querySelector('main').prepend(second);
    }""")
    reordered = backend.observe()
    assert before.text != reordered.text
    assert before.controls[0].identity() != reordered.controls[0].identity()
    assert before.scope_key() == reordered.scope_key()
    page.evaluate("""() => {
        const error = document.createElement('p');
        error.textContent = 'Email address is required';
        document.querySelector('main').append(error);
    }""")
    error_state = backend.observe()
    assert "Email address is required" in error_state.semantic_text
    assert error_state.scope_key() != reordered.scope_key()


@pytest.mark.parametrize("external", [False, True])
def test_redirect_target_is_blocked_before_any_target_request(page, site, external):
    (allowed, outside), requests = site
    page.goto(allowed + "/start")
    backend = BrowserAgentBackend(page)
    target = (outside if external else allowed) + "/redirect-target"
    page.set_content(f'<a href="{allowed}/redirect?to={target}">Continue</a>')
    snapshot = backend.observe()
    step = AgentStep("redirect", "click", ref=control(snapshot, "Continue").ref)
    candidate = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success
    assert not backend.verify(candidate, receipt).passed
    assert "/redirect-target" not in requests
    assert any(path.startswith("/redirect?") for path in requests)


def test_popup_first_request_is_blocked_before_external_contact(page, site):
    (allowed, outside), requests = site
    page.goto(allowed + "/start")
    backend = BrowserAgentBackend(page)
    page.set_content(f"""<button onclick="window.open('{outside}/popup-target', '_blank')">Open</button>""")
    snapshot = backend.observe()
    candidate = backend.compile(
        AgentStep("popup", "click", ref=snapshot.controls[0].ref), snapshot, snapshot
    )[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success
    assert not backend.verify(candidate, receipt).passed
    assert "/popup-target" not in requests
    assert len(page.context.pages) == 1
    backend.close()
    page.goto(outside + "/after-close")
    assert "/after-close" in requests


def test_frame_navigation_is_blocked_but_subresources_are_permitted(page, site):
    (allowed, outside), requests = site
    page.goto(allowed + "/start")
    backend = BrowserAgentBackend(page)
    page.set_content(f"""<button onclick="const f=document.createElement('iframe');
      f.src='{outside}/frame-target';document.body.append(f)">Embed</button>""")
    snapshot = backend.observe()
    candidate = backend.compile(
        AgentStep("frame", "click", ref=snapshot.controls[0].ref), snapshot, snapshot
    )[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success
    assert "/frame-target" not in requests
    backend.close()


def test_semantic_scope_detects_swapped_account_context_and_fallback_order(page):
    page.set_content("""<section id="selected">Alice</section><aside id="archived">Bob</aside>
      <button>Delete selected account</button>""")
    backend = BrowserAgentBackend(page)
    before = backend.observe()
    page.evaluate("selected.textContent='Bob';archived.textContent='Alice'")
    after = backend.observe()
    assert before.semantic_text == ("Alice", "Bob")
    assert after.semantic_text == ("Bob", "Alice")
    assert before.controls == after.controls
    assert before.scope_key() != after.scope_key()
    assert replace(before, semantic_text=None).scope_key() != replace(after, semantic_text=None).scope_key()


@pytest.mark.parametrize("attributes", ['type="password"', 'aria-disabled="true"'])
def test_page_keyboard_refuses_sensitive_or_disabled_active_control(page, attributes):
    page.set_content(f'<input {attributes} value="secret">')
    page.locator("input").focus()
    page.keyboard.press("End")
    backend = BrowserAgentBackend(page)
    snapshot = backend.observe()
    candidate = backend.compile(AgentStep("key", "press", value="Backspace"), snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success and "sensitive, hidden or disabled" in receipt.error
    assert page.locator("input").input_value() == "secret"


@pytest.mark.parametrize(
    "markup",
    [
        '<input id="target" type="password">',
        '<input id="target" type="file">',
        '<button id="target" disabled>Disabled</button>',
        '<div role="button" aria-disabled="true"><span id="target">Disabled</span></div>',
        '<label id="target" for="secret">Upload</label><input id="secret" type="file" hidden>',
    ],
)
def test_visual_click_refuses_sensitive_and_disabled_hit_targets(page, markup):
    page.set_content(markup)
    backend = BrowserAgentBackend(page)
    rect = page.locator("#target").bounding_box()
    box = (
        round(rect["x"] / 900 * 1000),
        round(rect["y"] / 700 * 1000),
        round((rect["x"] + rect["width"]) / 900 * 1000),
        round((rect["y"] + rect["height"]) / 700 * 1000),
    )
    snapshot = backend.observe()
    step = AgentStep("visual", "visual_click", box=box, label="target")
    candidate = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success and "sensitive, hidden or disabled" in receipt.error


def test_navigation_fetch_failure_becomes_failed_receipt_without_escaping_handler(page, site):
    (allowed, _), _requests = site
    page.goto(allowed + "/start")
    backend = BrowserAgentBackend(page)
    page.set_content(f'<a href="{allowed}/disconnect">Continue</a>')
    snapshot = backend.observe()
    step = AgentStep("network", "click", ref=control(snapshot, "Continue").ref)
    candidate = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "dec")
    assert not receipt.success
    assert not backend.verify(candidate, receipt).passed

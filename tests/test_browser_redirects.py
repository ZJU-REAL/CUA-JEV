import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from test_agent_browser import browser, page  # noqa: F401

from cua_jev.agent_types import AgentStep
from cua_jev.browser_redirects import RedirectingBrowserBackend


@pytest.fixture
def redirects():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(("POST", self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(303)
            self.send_header("Location", "/final")
            self.end_headers()

        def do_GET(self):
            requests.append(("GET", self.path, None))
            parsed = urlparse(self.path)
            if parsed.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", parse_qs(parsed.query)["to"][0])
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<title>Final</title><h1>Reached final page</h1>")

        def log_message(self, *args):
            pass
    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(2)]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for thread in threads:
        thread.start()
    yield [f"http://127.0.0.1:{s.server_port}" for s in servers], requests
    for server, thread in zip(servers, threads, strict=True):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def click(backend):
    snapshot = backend.observe()
    target = next(c for c in snapshot.controls if c.label == "Continue")
    candidate = backend.compile(AgentStep("next", "click", ref=target.ref), snapshot, snapshot)[0]
    return backend.execute(candidate, "obs", "decision")


def test_allowed_redirect_keeps_real_browser_url_and_never_replays_post(page, redirects):  # noqa: F811
    (allowed, _), requests = redirects
    page.goto(allowed)
    page.set_content('<form action="/submit" method="post"><input name="value" value="one">'
                     '<button>Continue</button></form>')
    backend = RedirectingBrowserBackend(page, allowed_origins=[allowed])
    try:
        receipt = click(backend)
        assert receipt.success and page.url == allowed + "/final"
        assert [r for r in requests if r[0] == "POST"] == [("POST", "/submit", b"value=one")]
        assert sum(r[1] == "/final" for r in requests) == 1
    finally:
        backend.close()


@pytest.mark.parametrize("external", [False, True])
def test_each_redirect_hop_is_checked_before_target_request(page, redirects, external):  # noqa: F811
    (allowed, outside), requests = redirects
    target = (outside if external else allowed) + "/target"
    page.goto(allowed)
    page.set_content(f'<a href="{allowed}/redirect?to={target}">Continue</a>')
    backend = RedirectingBrowserBackend(page, allowed_origins=[allowed])
    try:
        receipt = click(backend)
        assert receipt.success is not external
        assert any(r[1] == "/target" for r in requests) is not external
        if not external:
            assert page.url == target
    finally:
        backend.close()


def test_popup_first_request_remains_blocked(page, redirects):  # noqa: F811
    (allowed, outside), requests = redirects
    page.goto(allowed)
    page.set_content(f'<button onclick="window.open(\'{outside}/popup\')">Continue</button>')
    backend = RedirectingBrowserBackend(page, allowed_origins=[allowed])
    try:
        receipt = click(backend)
        assert not receipt.success
        assert not any(r[1] == "/popup" for r in requests)
    finally:
        backend.close()

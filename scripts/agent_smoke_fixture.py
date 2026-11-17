"""Offline browser tasks for exercising an agent against real page state.

The site supplies goals and an independent terminal evaluator, never actions or
element references. The evaluator is for the caller, not an agent tool.
"""

from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

TASKS = ("form", "search", "canvas")

_STORAGE_KEYS = {
    "form": "cua_jev.smoke.ticket",
    "search": "cua_jev.smoke.shortlist",
    "canvas": "cua_jev.smoke.shapes",
}
_EXPECTED = {
    "form": {"subject": "Printer is offline", "priority": "high", "followUp": True},
    "search": ["aurora-desk-lamp"],
    "canvas": ["blue circle", "green square"],
}
_GOALS = {
    "form": (
        'Create and save a support ticket with subject "Printer is offline", '
        'priority "High", and "Request a follow-up" enabled.'
    ),
    "search": "Find the Aurora Desk Lamp in the catalog and add only that product to the local shortlist.",
    "canvas": "On the shape board, click the blue circle, then click the green square on the next board.",
}

_STYLE = """
* { box-sizing: border-box; }
body { margin: 0; background: #f0fdfa; color: #134e4a;
       font: 19px/1.5 Arial, sans-serif; }
main { max-width: 1000px; margin: 30px auto; padding: 0 24px; }
header { border-bottom: 2px solid #99f6e4; margin-bottom: 24px; padding-bottom: 12px; }
.eyebrow { color: #475569; font-size: 15px; font-weight: bold; letter-spacing: .08em; }
h1 { font-size: 34px; margin: 4px 0; line-height: 1.25; }
h2 { font-size: 23px; margin: 0 0 8px; }
p { margin: 10px 0; }
label { display: block; margin-bottom: 8px; font-weight: bold; }
input[type=text], input[type=search], select { min-height: 48px; width: 100%;
    border: 2px solid #64748b; border-radius: 6px; padding: 9px 12px;
    background: white; color: #134e4a; font: inherit; }
button { min-height: 48px; padding: 10px 20px; border: 2px solid #0f766e;
    border-radius: 6px; background: #0f766e; color: white; font: bold 18px Arial, sans-serif;
    cursor: pointer; }
button:hover { background: #115e59; }
.secondary { background: white; color: #134e4a; }
.secondary:hover { background: #ccfbf1; }
:focus-visible { outline: 3px solid #c2410c; outline-offset: 4px; }
.field { margin-bottom: 22px; }
.card { padding: 24px; background: white; border: 1px solid #99d5ce; border-radius: 10px; }
.form-layout { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }
.check { display: flex; align-items: center; gap: 12px; min-height: 48px; }
input[type=checkbox] { width: 24px; height: 24px; accent-color: #0f766e; }
.toolbar { display: flex; gap: 12px; flex-wrap: wrap; }
#tips { margin-bottom: 18px; }
#status { min-height: 30px; font-weight: bold; }
.hint { color: #475569; }
.search-box { display: flex; gap: 12px; align-items: flex-end; }
.search-box .field { flex: 1; margin-bottom: 0; }
.products { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 24px; }
.product { display: flex; flex-direction: column; }
.product p { flex: 1; }
.product button { width: 100%; }
canvas { display: block; width: 100%; height: auto; border: 2px solid #64748b;
         border-radius: 10px; background: white; cursor: pointer; }
@media (max-width: 720px) {
  main { margin: 20px auto; padding: 0 16px; }
  .form-layout, .products { grid-template-columns: 1fr; }
  .search-box { flex-direction: column; align-items: stretch; }
}
"""


def _document(title: str, content: str, script: str, *, task: str, reset: bool) -> str:
    reset_script = f"localStorage.removeItem({json.dumps(_STORAGE_KEYS[task])});" if reset else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)} | Local workspace</title><style>{_STYLE}</style></head>"
        '<body><main><header><div class="eyebrow">LOCAL WORKSPACE</div>'
        f"<h1>{html.escape(title)}</h1></header>{content}</main>"
        f"<script>(() => {{ {reset_script}\n{script}\n }})();</script></body></html>"
    )


def _form_page(*, reset: bool) -> str:
    content = """
    <div class="form-layout">
      <form id="ticket" class="card">
        <div class="field"><label for="subject">Subject</label>
          <input id="subject" name="subject" type="text" required autocomplete="off"></div>
        <div class="field"><label for="priority">Priority</label>
          <select id="priority" name="priority">
            <option value="low">Low</option><option value="normal" selected>Normal</option>
            <option value="high">High</option>
          </select></div>
        <label class="check field"><input id="follow-up" type="checkbox">Request a follow-up</label>
        <button type="submit">Save ticket</button>
      </form>
      <aside class="card"><h2>Ticket tools</h2>
        <button id="tips" type="button" class="secondary">Show keyboard tips</button>
        <button id="saved" type="button" class="secondary">View saved tickets</button>
        <p class="hint">Tickets are saved in this browser.</p>
      </aside>
    </div><p id="status" role="status"></p>
    """
    script = """
    const form = document.querySelector('#ticket');
    const subject = document.querySelector('#subject');
    const status = document.querySelector('#status');
    const key = 'cua_jev.smoke.ticket';
    // Move one existing, harmless control once. Stable identities survive the reorder.
    subject.addEventListener('input', () => {
      form.insertBefore(document.querySelector('#tips'), form.firstElementChild);
    }, {once: true});
    document.querySelector('#tips').addEventListener('click', () => {
      status.textContent = 'Use Tab to move between fields.';
    });
    document.querySelector('#saved').addEventListener('click', () => {
      status.textContent = localStorage.getItem(key) ? 'One saved ticket.' : 'No saved tickets.';
    });
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const ticket = {subject: subject.value,
        priority: document.querySelector('#priority').value,
        followUp: document.querySelector('#follow-up').checked};
      localStorage.setItem(key, JSON.stringify(ticket));
      status.textContent = 'Ticket saved.';
    });
    """
    return _document("New support ticket", content, script, task="form", reset=reset)


_PRODUCTS = (
    ("aurora-floor-lamp", "Aurora Floor Lamp", "A tall reading light for the living room."),
    ("aurora-desk-lamp", "Aurora Desk Lamp", "An adjustable task light for your desk."),
    ("aurora-desk-mat", "Aurora Desk Mat", "A soft surface for your keyboard and mouse."),
)


def _search_page(*, reset: bool, query: str | None = None) -> str:
    value = html.escape(query or "", quote=True)
    content = f"""
    <form class="search-box card" method="get" action="/search/results">
      <div class="field"><label for="query">Search products</label>
        <input id="query" type="search" name="q" value="{value}"
          placeholder="Product name" autocomplete="off" required></div>
      <button type="submit">Search</button>
    </form>
    """
    if query is None:
        content += '<p class="hint">Search the catalog to find products for your local shortlist.</p>'
    else:
        terms = query.casefold().split()
        products = [item for item in _PRODUCTS if any(term in item[1].casefold() for term in terms)]
        content += f'<p>Results for "{html.escape(query)}"</p><div class="products">'
        for product_id, title, description in products:
            content += (
                f'<article class="card product"><h2>{title}</h2><p>{description}</p>'
                f'<button type="button" data-product="{product_id}" '
                f'aria-label="Add {title} to shortlist">Add to shortlist</button></article>'
            )
        content += "</div>" if products else "<p>No products found. Try another search.</p></div>"
    content += '<p id="status" role="status"></p>'
    script = """
    const key = 'cua_jev.smoke.shortlist';
    document.querySelectorAll('[data-product]').forEach(button => {
      button.addEventListener('click', () => {
        const shortlist = JSON.parse(localStorage.getItem(key) || '[]');
        if (!shortlist.includes(button.dataset.product)) shortlist.push(button.dataset.product);
        localStorage.setItem(key, JSON.stringify(shortlist));
        button.textContent = 'Added to shortlist';
        button.disabled = true;
        document.querySelector('#status').textContent = 'Shortlist saved.';
      });
    });
    """
    return _document("Product catalog", content, script, task="search", reset=reset)


def _canvas_page(*, reset: bool) -> str:
    content = """
    <p>Choose a shape on the board. A correct choice opens the next board.</p>
    <canvas width="900" height="440" aria-label="Shape board"></canvas>
    """
    # Geometry and progress are closure-local; no DOM targets, hidden fields, or global hit map.
    script = """
    const canvas = document.querySelector('canvas');
    const context = canvas.getContext('2d');
    const key = 'cua_jev.smoke.shapes';
    let sequence = [];
    const boards = [
      [{kind:'circle', color:'#2563eb', x:170, y:230},
       {kind:'square', color:'#dc2626', x:450, y:230},
       {kind:'triangle', color:'#15803d', x:730, y:230}],
      [{kind:'triangle', color:'#2563eb', x:170, y:230},
       {kind:'circle', color:'#dc2626', x:450, y:230},
       {kind:'square', color:'#15803d', x:730, y:230}]
    ];
    function path(shape) {
      const p = new Path2D();
      if (shape.kind === 'circle') p.arc(shape.x, shape.y, 72, 0, Math.PI * 2);
      if (shape.kind === 'square') p.rect(shape.x - 72, shape.y - 72, 144, 144);
      if (shape.kind === 'triangle') {
        p.moveTo(shape.x, shape.y - 80); p.lineTo(shape.x + 82, shape.y + 68);
        p.lineTo(shape.x - 82, shape.y + 68); p.closePath();
      }
      return p;
    }
    function draw(message) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = '#134e4a'; context.textAlign = 'center';
      context.font = 'bold 28px Arial';
      context.fillText(sequence.length === 2 ? 'Both shapes selected' :
        `Board ${sequence.length + 1} of 2`, 450, 52);
      if (sequence.length === 2) {
        context.font = '24px Arial'; context.fillText('Your choices have been saved.', 450, 220);
        return;
      }
      for (const shape of boards[sequence.length]) {
        context.fillStyle = shape.color; context.fill(path(shape));
      }
      context.fillStyle = '#475569'; context.font = '22px Arial';
      context.fillText(message || 'Select the requested shape.', 450, 390);
    }
    canvas.addEventListener('click', event => {
      if (sequence.length === 2) return;
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left - canvas.clientLeft) * canvas.width / canvas.clientWidth;
      const y = (event.clientY - rect.top - canvas.clientTop) * canvas.height / canvas.clientHeight;
      const shapes = boards[sequence.length];
      const chosen = shapes.find(shape => context.isPointInPath(path(shape), x, y));
      const expected = sequence.length === 0 ? shapes[0] : shapes[2];
      if (chosen !== expected) { draw('That shape did not advance the board. Try again.'); return; }
      sequence.push(sequence.length === 0 ? 'blue circle' : 'green square');
      localStorage.setItem(key, JSON.stringify(sequence));
      draw();
    });
    draw();
    """
    return _document("Shape board", content, script, task="canvas", reset=reset)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        request = urlsplit(self.path)
        query = parse_qs(request.query)
        reset = query.get("reset") == ["1"]
        if request.path == "/form":
            body = _form_page(reset=reset)
        elif request.path == "/search":
            body = _search_page(reset=reset)
        elif request.path == "/search/results":
            body = _search_page(reset=False, query=query.get("q", [""])[0])
        elif request.path == "/canvas":
            body = _canvas_page(reset=reset)
        else:
            self.send_error(404)
            return
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class SmokeSite:
    """Serve three local tasks until the context exits; never invoke an agent."""

    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._origin: str | None = None

    def __enter__(self) -> SmokeSite:
        if self._server is not None:
            raise RuntimeError("SmokeSite is already running")
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        self._server = server
        self._origin = f"http://127.0.0.1:{server.server_address[1]}"
        self._thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05},
            name="agent-smoke-site", daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
        self._origin = None

    @staticmethod
    def _validate_task(task: str) -> None:
        if task not in TASKS:
            raise ValueError(f"Unknown smoke task: {task!r}")

    def url(self, task: str) -> str:
        """Return the task's fresh-start URL (navigation clears that task's state)."""
        self._validate_task(task)
        if self._origin is None:
            raise RuntimeError("Enter the SmokeSite context before requesting a URL")
        return f"{self._origin}/{task}?reset=1"

    def goal(self, task: str) -> str:
        self._validate_task(task)
        return _GOALS[task]

    def verify(self, task: str, page: Any) -> bool:
        """Read actual origin-local storage, independently of agent reports or UI text."""
        self._validate_task(task)
        if self._origin is None:
            return False
        current = urlsplit(page.url)
        if f"{current.scheme}://{current.netloc}" != self._origin:
            return False
        try:
            raw = page.evaluate("key => localStorage.getItem(key)", _STORAGE_KEYS[task])
            if not isinstance(raw, str):
                return False
            actual = json.loads(raw)
        except (ValueError, TypeError):
            return False
        # Strict JSON equality also distinguishes a real checkbox boolean from the number 1.
        return json.dumps(actual, sort_keys=True) == json.dumps(_EXPECTED[task], sort_keys=True)

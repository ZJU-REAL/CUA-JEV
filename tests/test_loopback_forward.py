import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from cua_jev.loopback_forward import LoopbackForward


def test_real_tcp_forward_preserves_post_once_and_closes_active_connections():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            received.append((self.path, body, self.headers['Host']))
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    forward = LoopbackForward('127.0.0.1', server.server_port)
    idle = socket.create_connection(('127.0.0.1', forward.listen_port), timeout=2)
    try:
        body = b'bounded-stream' * 100_000
        with httpx.Client(trust_env=False) as client:
            response = client.post(f'http://127.0.0.1:{forward.listen_port}/submit', content=body,
                                   headers={'Host': 'owned-site.example'})
        assert response.content == body
        assert received == [('/submit', body, 'owned-site.example')]
    finally:
        forward.close()
        idle.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    forward.close()
    assert not forward._thread.is_alive() and not forward._tasks
    with pytest.raises(OSError):
        socket.create_connection(('127.0.0.1', forward.listen_port), timeout=.2)


def test_port_collision_fails_without_touching_existing_listener():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        with pytest.raises(RuntimeError, match='start_failed'):
            LoopbackForward('127.0.0.1', 80, listen_port=listener.getsockname()[1])
        with socket.create_connection(listener.getsockname(), timeout=1):
            accepted, _ = listener.accept()
            accepted.close()


@pytest.mark.parametrize('host', ['0.0.0.0', '8.8.8.8', 'example.com', '::1'])
def test_forward_never_resolves_names_or_connects_to_public_destinations(host):
    with pytest.raises(ValueError):
        LoopbackForward(host, 80)

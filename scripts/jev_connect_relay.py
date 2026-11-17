"""Authenticated loopback CONNECT relay for one fixed HTTPS service.

The remote caller still verifies api.typesafe.ai's TLS certificate. This relay
does not terminate TLS, see API keys, accept other destinations, or log traffic.
Use through an owned SSH reverse forward; close the process after the run.
"""

import argparse
import base64
import hmac
import json
import select
import socket
import socketserver
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import getproxies


def serve(config, state):
    token = json.loads(config.read_text())["token"]
    expected = "Basic " + base64.b64encode(f"cua-jev:{token}".encode()).decode()
    system_proxy = getproxies().get("https")
    proxy = urlsplit(system_proxy) if system_proxy else None
    if proxy and (proxy.scheme != "http" or proxy.username or proxy.password):
        raise ValueError("only the existing unauthenticated HTTP system proxy is supported")

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(15)
            data = b""
            try:
                while b"\r\n\r\n" not in data and len(data) < 8192:
                    packet = self.request.recv(1024)
                    if not packet:
                        return
                    data += packet
                head, _, extra = data.partition(b"\r\n\r\n")
                lines = head.decode("ascii").split("\r\n")
                headers = dict(line.split(":", 1) for line in lines[1:] if ":" in line)
                authorization = next((v.strip() for k, v in headers.items()
                                      if k.lower() == "proxy-authorization"), "")
                if lines[0] not in ("CONNECT api.typesafe.ai:443 HTTP/1.1",
                                    "CONNECT api.typesafe.ai:443 HTTP/1.0"):
                    self.request.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                    return
                if not hmac.compare_digest(authorization, expected):
                    self.request.sendall(b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                                         b"Content-Length: 0\r\n\r\n")
                    return
                destination = (proxy.hostname, proxy.port or 80) if proxy else ("api.typesafe.ai", 443)
                with socket.create_connection(destination, timeout=15) as upstream:
                    if proxy:
                        upstream.sendall(b"CONNECT api.typesafe.ai:443 HTTP/1.1\r\n"
                                         b"Host: api.typesafe.ai:443\r\n\r\n")
                        response = b""
                        while b"\r\n\r\n" not in response and len(response) < 8192:
                            packet = upstream.recv(1024)
                            if not packet:
                                return
                            response += packet
                        if response.split(b"\r\n", 1)[0].split(b" ")[1:2] != [b"200"]:
                            return
                    upstream.settimeout(60)
                    self.request.settimeout(60)
                    self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                    if extra:
                        upstream.sendall(extra)
                    while True:
                        ready, _, _ = select.select([self.request, upstream], [], [], 60)
                        if not ready:
                            return
                        for source in ready:
                            packet = source.recv(65536)
                            if not packet:
                                return
                            (upstream if source is self.request else self.request).sendall(packet)
            except (OSError, ValueError, UnicodeError):
                return

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True

    with Server(("127.0.0.1", 0), Handler) as server:
        state.write_text(json.dumps({"port": server.server_address[1]}) + "\n")
        print(json.dumps({"event": "ready", "port": server.server_address[1]}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    serve(args.config, args.state)

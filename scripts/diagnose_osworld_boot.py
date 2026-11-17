"""Owned VM infrastructure diagnostic, without importing the OSWorld evaluator."""

import argparse
import json
import os
import signal
import socket
import threading
import time
from pathlib import Path

from cua_jev.osworld_qemu import OwnedQemuProvider


def capture(directory, number):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(5)
            connection.connect(f"/proc/self/fd/{fd}/qmp.sock")
            stream = connection.makefile("rwb")
            json.loads(stream.readline())
            for request in (
                {"execute": "qmp_capabilities"},
                {"execute": "query-status"},
                {"execute": "screendump", "arguments": {
                    "filename": str(directory / f"boot-{number:02}.ppm")}},
            ):
                stream.write(json.dumps(request).encode() + b"\n")
                stream.flush()
                for _ in range(20):
                    response = json.loads(stream.readline())
                    if "return" in response or "error" in response:
                        break
                else:
                    raise RuntimeError("QMP response limit")
                if "error" in response:
                    raise RuntimeError("QMP diagnostic request failed")
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-user", required=True)
    args = parser.parse_args()
    output, base = args.output.resolve(), args.base.resolve()
    if not all(p.is_relative_to(Path.home().resolve()) for p in (output, base)):
        parser.error("all paths must remain within the current account's home")
    output.mkdir(parents=True, exist_ok=False)
    provider = OwnedQemuProvider(base, output, args.expected_user)
    provider.startup_timeout_s = 300
    stopped = threading.Event()
    diagnostics = []

    def take_frames():
        number = 0
        while not stopped.wait(30):
            for directory in output.glob("qemu-*"):
                number += 1
                try:
                    capture(directory, number)
                    diagnostics.append({"frame": number, "captured": True})
                except Exception as error:
                    diagnostics.append({"frame": number, "error_type": type(error).__name__})
                print(json.dumps(diagnostics[-1]), flush=True)

    def terminate(_signum, _frame):
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    thread = threading.Thread(target=take_frames, daemon=True)
    thread.start()
    started = time.perf_counter()
    result = {"kind": "infrastructure_only", "startup_timeout_s": 300, "ready": False}
    try:
        provider.start_emulator(base)
        result["ready"] = True
    except Exception as error:
        result.update(ready=False, error_type=type(error).__name__, code=getattr(error, "code", None))
        if isinstance(error, ModuleNotFoundError):
            result["missing_module"] = error.name
    finally:
        stopped.set()
        thread.join(timeout=10)
        provider.close()
        result.update(elapsed_s=time.perf_counter() - started, runtime=provider.metadata,
                      diagnostics=diagnostics)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("ready", "elapsed_s")}), flush=True)


if __name__ == "__main__":
    main()

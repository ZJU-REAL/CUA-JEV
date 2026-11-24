"""Bound AX acquisition on an owned Unix worker without retrying timed-out reads."""

from __future__ import annotations

import math
import signal
import threading
import time


class _AXDeadline(BaseException):
    # SDK readers catch Exception and retry. This deadline must escape that loop.
    pass


class AccessibilityBudget:
    """Temporarily wrap one controller; missing AX never becomes cached success.

    After an acquisition timeout the circuit stays open until this episode ends.
    Cancelling a client wait does not cancel the guest traversal, so starting more
    requests could overload the guest. Evaluation must run after restore().
    """

    def __init__(self, controller, timeout_s: float):
        if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s)
                or not 0 < timeout_s <= 60):
            raise ValueError("AX budget must be finite and in (0, 60] seconds")
        if (not hasattr(signal, "setitimer")
                or threading.current_thread() is not threading.main_thread()):
            raise ValueError("AX budget requires a Unix worker main thread")
        original = getattr(controller, "get_accessibility_tree", None)
        if not callable(original):
            raise ValueError("controller has no accessibility reader")
        self.controller, self.original, self.timeout_s = controller, original, timeout_s
        self.previous = ("get_accessibility_tree" in vars(controller),
                         vars(controller).get("get_accessibility_tree"))
        self.report = {"version": "ax_budget_v1", "timeout_s": timeout_s, "calls": 0,
                       "timeouts": 0, "unavailable": 0, "skipped": 0, "wall_ms": 0.0,
                       "circuit_open": False, "restored": False}
        self._wrapper = self.read
        controller.get_accessibility_tree = self._wrapper

    def read(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("AX budget used outside its worker main thread")
        if self.report["circuit_open"] or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            # Do not replace an outer startup/shutdown watchdog.
            self.report["skipped"] += 1
            return None
        self.report["calls"] += 1
        began = time.perf_counter()
        previous_handler = signal.getsignal(signal.SIGALRM)

        def expired(_signum, _frame):
            raise _AXDeadline()

        signal.signal(signal.SIGALRM, expired)
        try:
            signal.setitimer(signal.ITIMER_REAL, self.timeout_s)
            result = self.original()
            if result is None:
                self.report["unavailable"] += 1
            return result
        except _AXDeadline:
            self.report["timeouts"] += 1
            self.report["circuit_open"] = True
            return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            self.report["wall_ms"] += (time.perf_counter() - began) * 1000

    def restore(self):
        if self.report["restored"]:
            return
        if self.controller.get_accessibility_tree is not self._wrapper:
            raise RuntimeError("restore observation instrumentation before AX budget")
        existed, value = self.previous
        if existed:
            self.controller.get_accessibility_tree = value
        else:
            del self.controller.get_accessibility_tree
        self.report["restored"] = True

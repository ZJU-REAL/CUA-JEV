"""Opt-in owned boot wait inside the existing remote startup budget only."""

import math
from time import monotonic_ns

LEGACY_BOOT_BUDGET = "legacy_180"
SHARED_BOOT_BUDGET = "shared_startup_remaining_v1"
BOOT_BUDGET_POLICIES = (LEGACY_BOOT_BUDGET, SHARED_BOOT_BUDGET)


class OwnedBootWaitBudget:
    """One VM creation attempt and one SDK wait, with no new timeout authority."""

    def __init__(self, startup_timeout_s, *, worker_entry_ns, clock=monotonic_ns):
        if (type(startup_timeout_s) not in (int, float) or not math.isfinite(startup_timeout_s)
                or not 0 < startup_timeout_s <= 600 or type(worker_entry_ns) is not int
                or worker_entry_ns < 0):
            raise ValueError("invalid inherited startup budget")
        self.timeout_s, self.entry_ns, self.clock = float(startup_timeout_s), worker_entry_ns, clock
        self.report = {
            "version": SHARED_BOOT_BUDGET, "declared_outer_ready_budget_s": self.timeout_s,
            "worker_clock_origin": "before_project_imports", "clock": "monotonic_ns",
            "original_owned_boot_timeout_s": 180, "response_margin_s": 15.0,
            "start_invocations": 0, "wait_admission_attempts": 0, "SDK_wait_invocations": 0,
            "SDK_timeout_semantics_verified": False, "cross_host_deadline_identity_proven": False,
            "outer_ready_timeout_unchanged": True, "agent_timeout_unchanged": True,
            "margin_limit": "15s allowance, not a proved transport/cleanup bound.",
            "retry_policy": "No new start before same-worker terminal and independent owned cleanup.",
            "scope": "Boot wait only; original setup/readiness/settling caps and failure rules remain.",
            "limit": "Worker-local remaining time; existing client ready timeout remains authoritative. "
                     "SDK timeout may overshoot and is not a measured hard wall limit.",
        }

    def remaining(self):
        now = self.clock()
        if type(now) is not int or now < self.entry_ns:
            raise ValueError("invalid monotonic startup clock")
        return now, self.timeout_s - (now - self.entry_ns) / 1_000_000_000

    def begin_start(self):
        self.report["start_invocations"] += 1
        if self.report["start_invocations"] != 1:
            raise RuntimeError("shared startup budget forbids a second VM start")
        now, remaining = self.remaining()
        self.report["factory_start_elapsed_ms"] = (now - self.entry_ns) / 1_000_000
        self.report["factory_start_remaining_s"] = remaining
        if remaining <= self.report["response_margin_s"]:
            raise TimeoutError("inherited startup budget exhausted before VM creation")

    def wait_for_ready(self, original_wait):
        if self.report["start_invocations"] != 1:
            raise RuntimeError("SDK boot wait requires the single inherited VM start")
        self.report["wait_admission_attempts"] += 1
        if self.report["wait_admission_attempts"] != 1:
            raise RuntimeError("shared startup budget forbids a second SDK boot wait")
        started, remaining = self.remaining()
        effective = math.floor(remaining - self.report["response_margin_s"])
        self.report.update(boot_wait_elapsed_before_ms=(started - self.entry_ns) / 1_000_000,
                           boot_wait_remaining_before_s=remaining,
                           effective_SDK_timeout_s=max(0, effective))
        if effective <= 0:
            raise TimeoutError("inherited startup budget exhausted before SDK boot wait")
        self.report["SDK_wait_invocations"] += 1
        try:
            return original_wait(timeout=effective)
        except BaseException as error:
            self.report["SDK_wait_error_type"] = type(error).__name__
            raise
        finally:
            try:
                ended = self.clock()
                self.report["SDK_wait_wall_ms"] = max(0, ended - started) / 1_000_000
                self.report["worker_elapsed_after_wait_ms"] = max(0, ended - self.entry_ns) / 1_000_000
            except Exception as error:
                self.report["SDK_wait_wall_ms"] = None
                self.report["timing_error_type"] = type(error).__name__

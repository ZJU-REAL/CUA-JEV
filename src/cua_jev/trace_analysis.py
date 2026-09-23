"""Aggregate local episode traces without exporting task text or screenshots."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 3)


def analyze_traces(paths: list[str | Path]) -> dict[str, Any]:
    """Report empirical outcomes and route probabilities from JSONL traces.

    Partial runs are counted separately from completed episodes. Probabilities
    are aggregated over candidate IDs belonging to the same channel.
    """
    if not paths:
        raise ValueError("at least one trace path is required")
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_path in paths:
        path = Path(raw_path)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path.name} line {number}") from exc
            if not isinstance(event, dict) or not isinstance(event.get("run_id"), str):
                raise ValueError(f"invalid trace event in {path.name} line {number}")
            runs[event["run_id"]].append(event)

    durations: list[float] = []
    decision_ms: list[float] = []
    confidences: list[float] = []
    entropies: list[float] = []
    chosen_channels: Counter[str] = Counter()
    channel_masses: dict[str, list[float]] = defaultdict(list)
    chosen_routes: Counter[str] = Counter()
    route_masses: dict[str, list[float]] = defaultdict(list)
    statuses: Counter[str] = Counter()
    token_usage: dict[str, Counter[str]] = defaultdict(Counter)
    complete = 0
    decisions = 0
    for events in runs.values():
        candidate_channels: dict[str, str] = {}
        candidate_routes: dict[str, str] = {}
        for event in sorted(events, key=lambda item: item.get("sequence", 0)):
            kind, payload = event.get("kind"), event.get("payload")
            if not isinstance(payload, dict):
                continue
            if kind == "candidates":
                candidate_channels = {
                    item["id"]: item["channel"]
                    for item in payload.get("items", [])
                    if isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                    and isinstance(item.get("channel"), str)
                }
                candidate_routes = {
                    item["id"]: f"{item['channel']}:{item['capability']}"
                    for item in payload.get("items", [])
                    if isinstance(item, dict)
                    and all(isinstance(item.get(key), str) for key in ("id", "channel", "capability"))
                }
            elif kind == "decision":
                if not str(payload.get("model", "")).startswith("jev-"):
                    continue
                probabilities = payload.get("probabilities")
                if not isinstance(probabilities, dict) or not candidate_channels:
                    continue
                masses: dict[str, float] = defaultdict(float)
                route_probabilities: dict[str, float] = defaultdict(float)
                valid = True
                for candidate_id, value in probabilities.items():
                    if (
                        candidate_id not in candidate_channels
                        or type(value) not in (int, float)
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        valid = False
                        break
                    masses[candidate_channels[candidate_id]] += float(value)
                    if candidate_id in candidate_routes:
                        route_probabilities[candidate_routes[candidate_id]] += float(value)
                if not valid or abs(sum(masses.values()) - 1) > 0.02:
                    continue
                decisions += 1
                for channel in set(candidate_channels.values()):
                    channel_masses[channel].append(masses.get(channel, 0.0))
                for route in set(candidate_routes.values()):
                    route_masses[route].append(route_probabilities.get(route, 0.0))
                selected = candidate_channels.get(payload.get("candidate_id"))
                if selected:
                    chosen_channels[selected] += 1
                selected_route = candidate_routes.get(payload.get("candidate_id"))
                if selected_route:
                    chosen_routes[selected_route] += 1
                latency = payload.get("latency_ms")
                confidence = payload.get("confidence")
                if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
                    decision_ms.append(float(latency))
                if type(confidence) in (int, float) and 0 <= confidence <= 1:
                    confidences.append(float(confidence))
                entropies.append(-sum(
                    float(p) * math.log2(float(p)) for p in probabilities.values() if p > 0
                ))
            elif kind == "episode":
                complete += 1
                if isinstance(payload.get("status"), str):
                    statuses[payload["status"]] += 1
                duration = payload.get("duration_ms")
                if type(duration) in (int, float) and math.isfinite(duration) and duration >= 0:
                    durations.append(float(duration))
            elif kind in {"model_usage", "policy_exchange"}:
                role = payload.get("role", "jev" if kind == "policy_exchange" else None)
                usage = payload.get("usage")
                if isinstance(role, str) and isinstance(usage, dict):
                    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = usage.get(name)
                        if type(value) is int and value >= 0:
                            token_usage[role][name] += value

    return {
        "runs_seen": len(runs),
        "completed_episodes": complete,
        "successes": statuses["success"],
        "success_rate": round(statuses["success"] / complete, 4) if complete else None,
        "statuses": dict(statuses),
        "wall_ms_median": _percentile(durations, 0.5),
        "wall_ms_p95": _percentile(durations, 0.95),
        "jev_decisions": decisions,
        "decision_ms_median": _percentile(decision_ms, 0.5),
        "decision_ms_p95": _percentile(decision_ms, 0.95),
        "mean_confidence": round(statistics.mean(confidences), 4) if confidences else None,
        "mean_choice_entropy_bits": round(statistics.mean(entropies), 4) if entropies else None,
        "selected_channels": dict(chosen_channels),
        "selected_routes": dict(chosen_routes),
        "mean_probability_by_available_channel": {
            channel: round(statistics.mean(values), 4)
            for channel, values in sorted(channel_masses.items())
        },
        "mean_probability_by_available_route": {
            route: round(statistics.mean(values), 4)
            for route, values in sorted(route_masses.items())
        },
        "reported_model_tokens": {role: dict(values) for role, values in token_usage.items()},
        "cost_usd": None,  # Requires actual provider usage and a dated rate card.
    }

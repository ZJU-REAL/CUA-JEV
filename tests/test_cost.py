import json
from pathlib import Path

import pytest

from cua_jev.cost import (
    CODEX_ENTERPRISE_USD_PER_MILLION,
    JEV_INPUT_USD_PER_MILLION,
    codex_reference_usd,
    codex_standard_credits,
    jev_model_usd,
)


def test_published_pilot_costs_recompute_from_token_counters() -> None:
    snapshot = Path(__file__).resolve().parents[1] / "website" / "snapshot.json"
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    evidence = data["cost_evidence"]
    assert evidence["jev"]["input_usd_per_million"] == JEV_INPUT_USD_PER_MILLION
    codex_rates = evidence["codex"]["usd_per_million"]
    assert tuple(codex_rates[key] for key in ("uncached_input", "cached_input", "output")) == (
        CODEX_ENTERPRISE_USD_PER_MILLION
    )
    for task, benchmark in data["benchmarks"].items():
        rows = benchmark["rows"]
        jev_row = next(row for row in rows if row["agent"] == "jev" and row["action_space"] == "hybrid")
        codex_row = next(row for row in rows if row["agent"] == "codex_computer_use")
        assert jev_model_usd(evidence["jev"]["median_hybrid_input_tokens"][task]) == pytest.approx(
            jev_row["median_model_cost_usd"]
        )
        tokens = evidence["codex"]["pilot_tokens"][task]
        counts = (tokens["input"], tokens["cached_input"], tokens["output"])
        assert codex_reference_usd(*counts) == pytest.approx(codex_row["median_reference_cost_usd"])
        assert codex_standard_credits(*counts) == pytest.approx(
            codex_row["median_reference_cost_usd"] / 0.04
        )


def test_codex_cost_rejects_impossible_cache_count() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        codex_reference_usd(1, 2, 0)

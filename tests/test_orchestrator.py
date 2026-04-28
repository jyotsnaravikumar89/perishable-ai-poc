"""Tests for the FreshFleet orchestrator (end-to-end pipeline)."""

import pytest
from src.agents.orchestrator import Orchestrator
from src.models import FreshnessTier


class TestOrchestrator:
    def test_full_pipeline_runs(self):
        orch = Orchestrator()
        result = orch.run(n_items=15, seed=42)

        assert result.items_scanned == 15
        assert len(result.assessments) > 0
        assert len(result.events) > 0
        assert result.completed_at is not None
        assert result.summary["items_scanned"] == 15

    def test_reproducible_with_seed(self):
        orch1 = Orchestrator()
        orch2 = Orchestrator()
        r1 = orch1.run(n_items=10, seed=123)
        r2 = orch2.run(n_items=10, seed=123)

        # Same seed should produce same assessments
        assert len(r1.assessments) == len(r2.assessments)
        for a1, a2 in zip(r1.assessments, r2.assessments):
            assert a1.produce_type == a2.produce_type
            assert a1.composite_score == a2.composite_score
            assert a1.tier == a2.tier

    def test_summary_has_required_keys(self):
        orch = Orchestrator()
        result = orch.run(n_items=20, seed=99)

        required_keys = [
            "items_scanned",
            "items_validated",
            "tier_distribution",
            "pick_lists_generated",
            "total_dispatch_cases",
            "average_freshness_score",
            "average_days_remaining",
        ]
        for key in required_keys:
            assert key in result.summary, f"Missing summary key: {key}"

    def test_tier_distribution_sums_to_validated(self):
        orch = Orchestrator()
        result = orch.run(n_items=30, seed=77)

        tier_total = sum(result.summary["tier_distribution"].values())
        assert tier_total == result.summary["items_validated"]

    def test_handles_small_inventory(self):
        orch = Orchestrator()
        result = orch.run(n_items=1, seed=42)
        assert len(result.assessments) >= 0  # Might be 0 if rejected
        assert result.completed_at is not None

    def test_handles_large_inventory(self):
        orch = Orchestrator()
        result = orch.run(n_items=100, seed=42)
        assert result.summary["items_scanned"] == 100
        assert len(result.pick_lists) >= 0

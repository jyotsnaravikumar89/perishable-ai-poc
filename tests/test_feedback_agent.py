"""Tests for the Feedback Loop Agent."""

import yaml
import pytest

from src.agents.feedback_agent import FeedbackLoopAgent, StoreOutcome
from src.agents.orchestrator import Orchestrator
from src.models import FreshnessTier


@pytest.fixture
def config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def feedback_agent(config, tmp_path):
    """Create a feedback agent with temp history file."""
    return FeedbackLoopAgent(config, history_path=str(tmp_path / "test_history.json"))


class TestFeedbackLoopAgent:
    def test_simulate_outcomes(self, feedback_agent):
        """Should generate simulated store outcomes from assessments."""
        orch = Orchestrator()
        result = orch.run(n_items=20, seed=42)

        outcomes = feedback_agent.simulate_outcomes(result.assessments)

        # Should only have outcomes for dispatched items (not STORE tier)
        store_count = sum(1 for a in result.assessments if a.tier == FreshnessTier.STORE)
        assert len(outcomes) == len(result.assessments) - store_count

        for outcome in outcomes:
            assert outcome.actual_condition in ("good", "acceptable", "degraded", "spoiled")
            assert outcome.actual_days_usable >= 0

    def test_compute_accuracy(self, feedback_agent):
        """Should compute accuracy metrics from outcomes."""
        orch = Orchestrator()
        result = orch.run(n_items=30, seed=42)
        outcomes = feedback_agent.simulate_outcomes(result.assessments, noise_level=0.1)

        output = feedback_agent.process(outcomes)

        metrics = output["accuracy_metrics"]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["total_predictions"] == len(outcomes)
        assert metrics["bias"] in ("optimistic", "pessimistic", "neutral")
        assert metrics["mean_absolute_error_days"] >= 0

    def test_weight_adjustments(self, feedback_agent):
        """Should recommend weight adjustments when accuracy is low."""
        # Create deliberately bad predictions
        outcomes = [
            StoreOutcome(
                item_id=f"bad-{i}",
                produce_type="tomato",
                variant="roma",
                predicted_tier=FreshnessTier.STORE,  # Predicted fresh
                predicted_days_remaining=6.0,
                actual_condition="spoiled",           # Actually spoiled
                actual_days_usable=0.2,
            )
            for i in range(10)
        ]

        output = feedback_agent.process(outcomes)

        # Should detect optimistic bias and recommend adjustments
        assert output["accuracy_metrics"]["bias"] == "optimistic"
        assert output["accuracy_metrics"]["accuracy"] < 0.5

    def test_supplier_scores(self, feedback_agent):
        """Should compute per-produce-type reliability scores."""
        orch = Orchestrator()
        result = orch.run(n_items=40, seed=42)
        outcomes = feedback_agent.simulate_outcomes(result.assessments)

        output = feedback_agent.process(outcomes)

        scores = output["supplier_scores"]
        assert len(scores) > 0
        for score in scores:
            assert 0.0 <= score["reliability_score"] <= 1.0
            assert score["trend"] in ("improving", "stable", "declining")

    def test_recommendations_generated(self, feedback_agent):
        """Should always generate at least one recommendation."""
        orch = Orchestrator()
        result = orch.run(n_items=15, seed=42)
        outcomes = feedback_agent.simulate_outcomes(result.assessments)

        output = feedback_agent.process(outcomes)

        assert len(output["recommendations"]) >= 1

    def test_history_persistence(self, config, tmp_path):
        """Should persist and load feedback history."""
        history_path = str(tmp_path / "persist_test.json")

        # First agent writes history
        agent1 = FeedbackLoopAgent(config, history_path=history_path)
        outcomes = [
            StoreOutcome(
                item_id="test-1",
                produce_type="tomato",
                variant="roma",
                predicted_tier=FreshnessTier.SHIP_NOW,
                predicted_days_remaining=1.5,
                actual_condition="degraded",
                actual_days_usable=1.0,
            )
        ]
        agent1.process(outcomes)

        # Second agent should load history
        agent2 = FeedbackLoopAgent(config, history_path=history_path)
        assert len(agent2._history) == 1

    def test_emits_events(self, feedback_agent):
        """Should emit processing events."""
        orch = Orchestrator()
        result = orch.run(n_items=10, seed=42)
        outcomes = feedback_agent.simulate_outcomes(result.assessments)

        feedback_agent.process(outcomes)
        events = feedback_agent.get_events()

        assert len(events) >= 1
        assert any(e.event_type == "feedback_processed" for e in events)

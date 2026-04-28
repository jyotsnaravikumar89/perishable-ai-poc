"""Feedback Loop Agent — Closed-Loop Learning from Dispatch Outcomes.

This agent closes the loop between dispatch decisions and real-world
outcomes. When produce arrives at a store, the store reports its actual
condition. This feedback flows back to:

    1. Evaluate classifier accuracy (was our tier prediction correct?)
    2. Adjust scoring weights over time (self-improving system)
    3. Detect systematic biases (are we consistently over/under-estimating?)
    4. Track supplier quality trends (does Supplier X's produce degrade faster?)

This is the agent that transforms FreshFleet from a one-shot pipeline
into a continuously learning system.

Architecture:
    Store POS / inspection → Feedback event → This agent →
    → Accuracy metrics
    → Weight adjustment recommendations
    → Supplier quality scores
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.models import FreshnessAssessment, FreshnessTier


# ── Feedback Data Models ────────────────────────────────────────────────

class StoreOutcome(BaseModel):
    """Feedback from the store about actual produce condition on arrival."""
    item_id: str
    produce_type: str
    variant: str
    predicted_tier: FreshnessTier
    predicted_days_remaining: float
    actual_condition: str = Field(
        description="Reported condition: 'good', 'acceptable', 'degraded', 'spoiled'"
    )
    actual_days_usable: Optional[float] = Field(
        default=None, description="Actual remaining shelf life observed at store"
    )
    hours_in_transit: float = Field(default=4.0)
    store_id: str = Field(default="STORE-001")
    reported_at: datetime = Field(default_factory=datetime.utcnow)


class AccuracyMetrics(BaseModel):
    """Classifier accuracy metrics computed from feedback."""
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    tier_accuracy: dict[str, float] = Field(default_factory=dict)
    mean_absolute_error_days: float = 0.0
    bias: str = Field(default="neutral", description="'optimistic', 'pessimistic', or 'neutral'")
    overestimate_count: int = 0
    underestimate_count: int = 0


class WeightAdjustment(BaseModel):
    """Recommended adjustment to classifier scoring weights."""
    feature: str
    current_weight: float
    recommended_weight: float
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class SupplierScore(BaseModel):
    """Quality tracking per supplier (simulated by produce_type for POC)."""
    supplier_key: str
    total_items: int
    avg_predicted_days: float
    avg_actual_days: float
    reliability_score: float = Field(
        ge=0.0, le=1.0, description="1.0 = predictions match reality perfectly"
    )
    trend: str = Field(description="'improving', 'stable', or 'declining'")


# ── Feedback Loop Agent ─────────────────────────────────────────────────

class FeedbackLoopAgent(BaseAgent):
    """Processes dispatch outcomes and generates learning signals.

    This agent collects store feedback, computes accuracy metrics,
    and recommends weight adjustments to improve classifier performance
    over time.
    """

    def __init__(self, config: dict, history_path: Optional[str] = None):
        super().__init__("FeedbackLoopAgent", config)
        self.history_path = history_path or "data/feedback_history.json"
        self._history: list[dict] = []
        self._load_history()

        # Mapping from actual condition to effective tier
        self._condition_to_tier = {
            "good": FreshnessTier.STORE,
            "acceptable": FreshnessTier.SHIP_SOON,
            "degraded": FreshnessTier.SHIP_NOW,
            "spoiled": FreshnessTier.SHIP_NOW,
        }

    def process(self, outcomes: list[StoreOutcome]) -> dict:
        """Process a batch of store outcomes and generate learning signals.

        Args:
            outcomes: List of store feedback on dispatched items.

        Returns:
            Dict containing accuracy_metrics, weight_adjustments,
            supplier_scores, and recommendations.
        """
        self.clear_events()

        # Store outcomes in history
        for outcome in outcomes:
            self._history.append(outcome.model_dump(mode="json"))

        # Compute metrics
        accuracy = self._compute_accuracy(outcomes)
        adjustments = self._recommend_weight_adjustments(outcomes, accuracy)
        supplier_scores = self._compute_supplier_scores()

        self._save_history()

        self.emit_event(
            "feedback_processed",
            f"🔄 Processed {len(outcomes)} outcomes — "
            f"accuracy: {accuracy.accuracy:.1%}, bias: {accuracy.bias}",
            {
                "total_outcomes": len(outcomes),
                "accuracy": accuracy.accuracy,
                "bias": accuracy.bias,
            },
        )

        return {
            "accuracy_metrics": accuracy.model_dump(),
            "weight_adjustments": [a.model_dump() for a in adjustments],
            "supplier_scores": [s.model_dump() for s in supplier_scores],
            "recommendations": self._generate_recommendations(accuracy, adjustments),
        }

    def simulate_outcomes(
        self,
        assessments: list[FreshnessAssessment],
        noise_level: float = 0.15,
    ) -> list[StoreOutcome]:
        """Simulate store outcomes from assessments for POC testing.

        Generates realistic feedback by adding noise to predictions.
        In production, this would be replaced by real store POS data.

        Args:
            assessments: Original classifier assessments.
            noise_level: How much randomness to add (0.0 = perfect predictions).

        Returns:
            List of simulated StoreOutcome objects.
        """
        import random

        outcomes = []
        for assessment in assessments:
            # Only dispatched items have outcomes (not STORE tier)
            if assessment.tier == FreshnessTier.STORE:
                continue

            # Add noise to the predicted days remaining
            actual_days = max(
                0.0,
                assessment.estimated_days_remaining
                + random.gauss(0, noise_level * assessment.estimated_days_remaining + 0.5)
            )

            # Map actual days to condition
            if actual_days <= 0.5:
                condition = "spoiled"
            elif actual_days <= 1.5:
                condition = "degraded"
            elif actual_days <= 4.0:
                condition = "acceptable"
            else:
                condition = "good"

            outcome = StoreOutcome(
                item_id=assessment.item_id,
                produce_type=assessment.produce_type,
                variant=assessment.variant,
                predicted_tier=assessment.tier,
                predicted_days_remaining=assessment.estimated_days_remaining,
                actual_condition=condition,
                actual_days_usable=round(actual_days, 1),
                hours_in_transit=round(random.uniform(2, 8), 1),
            )
            outcomes.append(outcome)

        return outcomes

    def _compute_accuracy(self, outcomes: list[StoreOutcome]) -> AccuracyMetrics:
        """Compute classification accuracy from store outcomes."""
        if not outcomes:
            return AccuracyMetrics()

        correct = 0
        total = len(outcomes)
        tier_correct: dict[str, int] = defaultdict(int)
        tier_total: dict[str, int] = defaultdict(int)
        day_errors = []
        overestimates = 0
        underestimates = 0

        for outcome in outcomes:
            actual_tier = self._condition_to_tier.get(
                outcome.actual_condition, FreshnessTier.SHIP_SOON
            )

            tier_total[outcome.predicted_tier.value] += 1

            if outcome.predicted_tier == actual_tier:
                correct += 1
                tier_correct[outcome.predicted_tier.value] += 1

            if outcome.actual_days_usable is not None:
                error = outcome.predicted_days_remaining - outcome.actual_days_usable
                day_errors.append(abs(error))

                if error > 0.5:
                    overestimates += 1
                elif error < -0.5:
                    underestimates += 1

        accuracy = correct / total if total > 0 else 0.0
        mae = sum(day_errors) / len(day_errors) if day_errors else 0.0

        # Determine bias direction
        if overestimates > underestimates * 1.5:
            bias = "optimistic"  # Predicting more days than reality
        elif underestimates > overestimates * 1.5:
            bias = "pessimistic"  # Predicting fewer days than reality
        else:
            bias = "neutral"

        tier_accuracy = {
            tier: tier_correct.get(tier, 0) / tier_total[tier]
            for tier in tier_total
            if tier_total[tier] > 0
        }

        return AccuracyMetrics(
            total_predictions=total,
            correct_predictions=correct,
            accuracy=round(accuracy, 3),
            tier_accuracy={k: round(v, 3) for k, v in tier_accuracy.items()},
            mean_absolute_error_days=round(mae, 2),
            bias=bias,
            overestimate_count=overestimates,
            underestimate_count=underestimates,
        )

    def _recommend_weight_adjustments(
        self,
        outcomes: list[StoreOutcome],
        metrics: AccuracyMetrics,
    ) -> list[WeightAdjustment]:
        """Generate weight adjustment recommendations based on accuracy."""
        adjustments = []
        current_weights = self.config.get("vision", {}).get("weights", {})

        # If optimistic bias: increase weight on degradation indicators
        if metrics.bias == "optimistic" and metrics.accuracy < 0.8:
            if current_weights.get("ethylene_level", 0.15) < 0.25:
                adjustments.append(WeightAdjustment(
                    feature="ethylene_level",
                    current_weight=current_weights.get("ethylene_level", 0.15),
                    recommended_weight=min(0.25, current_weights.get("ethylene_level", 0.15) + 0.03),
                    reason="Optimistic bias detected — increasing ethylene weight "
                           "to catch degradation earlier",
                    confidence=0.7,
                ))

            if current_weights.get("firmness_score", 0.25) < 0.35:
                adjustments.append(WeightAdjustment(
                    feature="firmness_score",
                    current_weight=current_weights.get("firmness_score", 0.25),
                    recommended_weight=min(0.35, current_weights.get("firmness_score", 0.25) + 0.03),
                    reason="Firmness is a strong early indicator of degradation — "
                           "increasing weight to reduce overestimation",
                    confidence=0.65,
                ))

        # If pessimistic bias: increase weight on positive indicators
        if metrics.bias == "pessimistic" and metrics.accuracy < 0.8:
            if current_weights.get("color_score", 0.30) < 0.40:
                adjustments.append(WeightAdjustment(
                    feature="color_score",
                    current_weight=current_weights.get("color_score", 0.30),
                    recommended_weight=min(0.40, current_weights.get("color_score", 0.30) + 0.03),
                    reason="Pessimistic bias detected — increasing color weight "
                           "to better recognize items that are still fresh",
                    confidence=0.65,
                ))

        # High MAE: suggest overall recalibration
        if metrics.mean_absolute_error_days > 2.0:
            adjustments.append(WeightAdjustment(
                feature="temperature_delta",
                current_weight=current_weights.get("temperature_delta", 0.10),
                recommended_weight=min(0.18, current_weights.get("temperature_delta", 0.10) + 0.03),
                reason=f"High prediction error (MAE: {metrics.mean_absolute_error_days:.1f} days) — "
                       f"temperature deviations may be underweighted",
                confidence=0.5,
            ))

        return adjustments

    def _compute_supplier_scores(self) -> list[SupplierScore]:
        """Compute quality scores by produce type (proxy for supplier)."""
        by_type: dict[str, list[dict]] = defaultdict(list)

        for record in self._history:
            by_type[record["produce_type"]].append(record)

        scores = []
        for produce_type, records in by_type.items():
            predicted_days = [r["predicted_days_remaining"] for r in records]
            actual_days = [
                r["actual_days_usable"] for r in records
                if r.get("actual_days_usable") is not None
            ]

            if not actual_days:
                continue

            avg_predicted = sum(predicted_days) / len(predicted_days)
            avg_actual = sum(actual_days) / len(actual_days)

            # Reliability = 1 - normalized MAE
            errors = [
                abs(r["predicted_days_remaining"] - r["actual_days_usable"])
                for r in records
                if r.get("actual_days_usable") is not None
            ]
            mae = sum(errors) / len(errors) if errors else 0
            reliability = max(0.0, 1.0 - mae / (avg_predicted + 0.01))

            # Simple trend: compare first half vs second half
            mid = len(actual_days) // 2
            if mid > 0:
                first_half_avg = sum(actual_days[:mid]) / mid
                second_half_avg = sum(actual_days[mid:]) / (len(actual_days) - mid)
                if second_half_avg > first_half_avg * 1.1:
                    trend = "improving"
                elif second_half_avg < first_half_avg * 0.9:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            scores.append(SupplierScore(
                supplier_key=produce_type,
                total_items=len(records),
                avg_predicted_days=round(avg_predicted, 1),
                avg_actual_days=round(avg_actual, 1),
                reliability_score=round(reliability, 3),
                trend=trend,
            ))

        return sorted(scores, key=lambda s: s.reliability_score)

    def _generate_recommendations(
        self,
        metrics: AccuracyMetrics,
        adjustments: list[WeightAdjustment],
    ) -> list[str]:
        """Generate human-readable recommendations."""
        recs = []

        if metrics.accuracy >= 0.85:
            recs.append(
                f"Classifier accuracy is strong at {metrics.accuracy:.0%}. "
                f"Continue monitoring with current weights."
            )
        elif metrics.accuracy >= 0.70:
            recs.append(
                f"Classifier accuracy at {metrics.accuracy:.0%} — room for improvement. "
                f"Consider applying the {len(adjustments)} weight adjustments below."
            )
        else:
            recs.append(
                f"Classifier accuracy at {metrics.accuracy:.0%} is below target. "
                f"Recommend retraining the model or significant weight recalibration."
            )

        if metrics.bias == "optimistic":
            recs.append(
                "System is optimistic — predicting more shelf life than reality. "
                "Risk: dispatching items too late. Action: tighten tier thresholds."
            )
        elif metrics.bias == "pessimistic":
            recs.append(
                "System is pessimistic — dispatching items earlier than necessary. "
                "This reduces spoilage but may increase logistics costs."
            )

        if metrics.mean_absolute_error_days > 1.5:
            recs.append(
                f"Average prediction error is {metrics.mean_absolute_error_days:.1f} days. "
                f"Consider adding time-series tracking for degradation curve fitting."
            )

        return recs

    def _load_history(self):
        """Load feedback history from disk."""
        path = Path(self.history_path)
        if path.exists():
            with open(path) as f:
                self._history = json.load(f)

    def _save_history(self):
        """Persist feedback history to disk."""
        path = Path(self.history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2, default=str)

    def reset_history(self):
        """Clear all feedback history."""
        self._history.clear()
        path = Path(self.history_path)
        if path.exists():
            path.unlink()

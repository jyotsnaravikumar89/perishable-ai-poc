"""Prioritizer Agent — Dispatch Urgency Scoring and Ranking.

This agent takes freshness assessments and computes a dispatch urgency
score that considers multiple factors beyond just freshness:

    - Freshness urgency (50%): How close to spoilage
    - Quantity factor (20%): Larger batches get higher priority (economies of dispatch)
    - Ethylene risk (15%): High-ethylene producers near sensitive items
    - Demand signal (15%): Simulated store demand pull

Items are then ranked by urgency score within their tier, creating a
fully ordered dispatch queue.
"""

from __future__ import annotations

import random
from typing import Optional

from src.agents.base import BaseAgent
from src.models import DispatchScore, FreshnessAssessment, FreshnessTier, ScanResult


class PrioritizerAgent(BaseAgent):
    """Scores and ranks assessed produce items for dispatch priority."""

    def __init__(self, config: dict):
        super().__init__("PrioritizerAgent", config)
        dispatch_config = config.get("dispatch", {})
        self.priority_weights = dispatch_config.get("priority_weights", {})
        self.catalog = {}

        # Load produce catalog for ethylene risk assessment
        import yaml
        with open("config/settings.yaml") as f:
            full_config = yaml.safe_load(f)
        self.catalog = full_config.get("produce_catalog", {})

    def process(
        self,
        assessments: list[FreshnessAssessment],
        scan_results: list[ScanResult],
    ) -> list[DispatchScore]:
        """Score and rank items for dispatch.

        Args:
            assessments: Freshness assessments from ClassifierAgent.
            scan_results: Original scan results (for bay location, case count).

        Returns:
            Sorted list of DispatchScore objects (most urgent first).
        """
        self.clear_events()

        # Build lookup from scan results
        scan_lookup = {s.item_id: s for s in scan_results}

        dispatch_scores = []
        for assessment in assessments:
            scan = scan_lookup.get(assessment.item_id)
            if not scan:
                continue

            urgency = self._compute_urgency(assessment, scan)
            ethylene_risk = self._assess_ethylene_risk(scan)

            score = DispatchScore(
                item_id=assessment.item_id,
                produce_type=assessment.produce_type,
                variant=assessment.variant,
                tier=assessment.tier,
                urgency_score=round(urgency, 3),
                estimated_days_remaining=assessment.estimated_days_remaining,
                case_count=scan.case_count,
                bay_location=scan.bay_location,
                ethylene_risk=ethylene_risk,
            )
            dispatch_scores.append(score)

        # Sort by tier priority first, then by urgency score (descending)
        dispatch_scores.sort(key=lambda s: (s.tier.sort_priority, -s.urgency_score))

        # Assign ranks
        for rank, score in enumerate(dispatch_scores, 1):
            score.dispatch_rank = rank

        self.emit_event(
            "prioritization_complete",
            f"📊 Prioritized {len(dispatch_scores)} items — "
            f"top item: {dispatch_scores[0].produce_type} "
            f"(urgency: {dispatch_scores[0].urgency_score:.2f})" if dispatch_scores else
            f"📊 No items to prioritize",
            {
                "total": len(dispatch_scores),
                "ethylene_risks": sum(1 for s in dispatch_scores if s.ethylene_risk),
            },
        )

        return dispatch_scores

    def _compute_urgency(self, assessment: FreshnessAssessment, scan: ScanResult) -> float:
        """Compute multi-factor dispatch urgency score (0.0-1.0)."""
        w = self.priority_weights

        # Freshness urgency: inverse of composite score (lower freshness = higher urgency)
        freshness_urgency = 1.0 - assessment.composite_score

        # Quantity factor: normalize case count (more cases = slightly higher priority)
        quantity_factor = min(1.0, scan.case_count / 15.0)

        # Ethylene risk: high producers get urgency boost to remove them from storage
        spec = self.catalog.get(scan.produce_type, {})
        ethylene_sens = spec.get("ethylene_sensitivity", "medium")
        ethylene_factor = {"high": 0.8, "medium": 0.4, "low": 0.1}.get(ethylene_sens, 0.4)
        if scan.scan_features.ethylene_ppm > 8.0:
            ethylene_factor = min(1.0, ethylene_factor + 0.3)

        # Demand signal: simulated store demand (in production, this comes from POS/demand API)
        demand_signal = self._simulate_demand(scan.produce_type)

        urgency = (
            freshness_urgency * w.get("freshness_urgency", 0.50)
            + quantity_factor * w.get("quantity_factor", 0.20)
            + ethylene_factor * w.get("ethylene_risk", 0.15)
            + demand_signal * w.get("demand_signal", 0.15)
        )

        return max(0.0, min(1.0, urgency))

    def _assess_ethylene_risk(self, scan: ScanResult) -> bool:
        """Determine if this item poses ethylene cross-contamination risk."""
        spec = self.catalog.get(scan.produce_type, {})
        if spec.get("ethylene_sensitivity") == "high" and scan.scan_features.ethylene_ppm > 6.0:
            return True
        return False

    def _simulate_demand(self, produce_type: str) -> float:
        """Simulate store demand signal (0.0-1.0).

        In production, this would query a demand forecasting service or POS API.
        For the POC, it uses category-based priors with noise.
        """
        base_demand = {
            "tomato": 0.7,
            "strawberry": 0.8,
            "spinach": 0.5,
            "banana": 0.9,
            "lettuce": 0.6,
            "blueberry": 0.7,
            "avocado": 0.8,
            "bell_pepper": 0.5,
        }
        base = base_demand.get(produce_type, 0.5)
        return max(0.0, min(1.0, base + random.gauss(0, 0.1)))

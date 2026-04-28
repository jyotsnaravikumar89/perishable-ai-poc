"""Classifier Agent — 3-Tier Freshness Classification.

This agent takes validated scan features and assigns each item to one
of three freshness tiers based on a weighted composite score:

    T1 (SHIP_NOW):  ≤2 days remaining — immediate dispatch
    T2 (SHIP_SOON): 3-5 days remaining — queue for next dispatch cycle
    T3 (STORE):     6+ days remaining — hold in cold storage

The composite score is a weighted combination of:
    - Color score (30%)
    - Firmness score (25%)
    - Blemish score (20%)
    - Ethylene level (15%)  — normalized and inverted
    - Temperature delta (10%) — deviation from optimal
"""

from __future__ import annotations

import yaml

from src.agents.base import BaseAgent
from src.models import FreshnessAssessment, FreshnessTier, ScanResult


class ClassifierAgent(BaseAgent):
    """Classifies produce into 3-tier freshness tags based on scan features."""

    def __init__(self, config: dict):
        super().__init__("ClassifierAgent", config)
        self.weights = config.get("vision", {}).get("weights", {})
        self.thresholds = config.get("classification", {}).get("score_thresholds", {})

        with open("config/settings.yaml") as f:
            full_config = yaml.safe_load(f)
        self.catalog = full_config.get("produce_catalog", {})

    def process(self, scan_results: list[ScanResult]) -> list[FreshnessAssessment]:
        """Classify each scanned item into a freshness tier.

        Args:
            scan_results: Validated scan results from VisionAgent.

        Returns:
            List of FreshnessAssessment objects with tier tags.
        """
        self.clear_events()
        assessments = []
        tier_counts = {"T1_SHIP_NOW": 0, "T2_SHIP_SOON": 0, "T3_STORE": 0}

        for scan in scan_results:
            composite = self._compute_composite_score(scan)
            days_remaining = self._estimate_remaining_days(scan, composite)
            tier = self._assign_tier(composite, days_remaining)
            risk_factors = self._identify_risks(scan, composite)

            assessment = FreshnessAssessment(
                item_id=scan.item_id,
                produce_type=scan.produce_type,
                variant=scan.variant,
                composite_score=round(composite, 3),
                estimated_days_remaining=round(days_remaining, 1),
                tier=tier,
                confidence=scan.scan_features.scan_confidence,
                risk_factors=risk_factors,
            )
            assessments.append(assessment)
            tier_counts[tier.value] += 1

        self.emit_event(
            "classification_complete",
            f"🏷️  Classified {len(assessments)} items — "
            f"{tier_counts['T1_SHIP_NOW']}x SHIP_NOW | "
            f"{tier_counts['T2_SHIP_SOON']}x SHIP_SOON | "
            f"{tier_counts['T3_STORE']}x STORE",
            {"tier_counts": tier_counts, "total": len(assessments)},
        )

        return assessments

    def _compute_composite_score(self, scan: ScanResult) -> float:
        """Compute weighted composite freshness score (0.0-1.0)."""
        f = scan.scan_features

        # Normalize ethylene: invert so high ethylene = low score
        spec = self.catalog.get(scan.produce_type, {})
        max_ethylene = {"high": 12.0, "medium": 5.0, "low": 1.5}.get(
            spec.get("ethylene_sensitivity", "medium"), 5.0
        )
        ethylene_normalized = max(0.0, 1.0 - (f.ethylene_ppm / (max_ethylene * 1.5)))

        # Temperature delta: how far from optimal (normalized)
        optimal_temp = spec.get("optimal_temp_c", 4.0)
        temp_delta = abs(f.surface_temp_c - optimal_temp)
        temp_score = max(0.0, 1.0 - (temp_delta / 10.0))  # 10°C deviation = score of 0

        # Weighted combination
        w = self.weights
        composite = (
            f.color_score * w.get("color_score", 0.30)
            + f.firmness_score * w.get("firmness_score", 0.25)
            + f.blemish_score * w.get("blemish_score", 0.20)
            + ethylene_normalized * w.get("ethylene_level", 0.15)
            + temp_score * w.get("temperature_delta", 0.10)
        )

        return max(0.0, min(1.0, composite))

    def _estimate_remaining_days(self, scan: ScanResult, composite: float) -> float:
        """Estimate remaining shelf life based on composite score and produce type."""
        spec = self.catalog.get(scan.produce_type, {})
        max_life = spec.get("max_shelf_life_days", 7)

        # Simple mapping: composite score roughly correlates with remaining life
        # A score of 1.0 ≈ full shelf life, 0.0 ≈ expired
        remaining = composite * max_life

        # Adjust for temperature stress
        optimal_temp = spec.get("optimal_temp_c", 4.0)
        temp_delta = abs(scan.scan_features.surface_temp_c - optimal_temp)
        if temp_delta > 3.0:
            remaining *= 0.85  # Temperature stress accelerates degradation

        return max(0.0, remaining)

    def _assign_tier(self, composite: float, days_remaining: float) -> FreshnessTier:
        """Assign freshness tier based on score thresholds and estimated days."""
        critical = self.thresholds.get("critical", 0.35)
        warning = self.thresholds.get("warning", 0.65)

        # Tier assignment uses BOTH score and time estimate (belt and suspenders)
        if composite <= critical or days_remaining <= 2.0:
            return FreshnessTier.SHIP_NOW
        elif composite <= warning or days_remaining <= 5.0:
            return FreshnessTier.SHIP_SOON
        else:
            return FreshnessTier.STORE

    def _identify_risks(self, scan: ScanResult, composite: float) -> list[str]:
        """Identify specific risk factors for this item."""
        risks = []
        f = scan.scan_features

        if f.color_score < 0.3:
            risks.append("severe_color_degradation")
        if f.firmness_score < 0.3:
            risks.append("soft_texture_detected")
        if f.blemish_score < 0.4:
            risks.append("significant_surface_blemishes")
        if f.ethylene_ppm > 10.0:
            risks.append("high_ethylene_emission")

        spec = self.catalog.get(scan.produce_type, {})
        optimal_temp = spec.get("optimal_temp_c", 4.0)
        if abs(f.surface_temp_c - optimal_temp) > 5.0:
            risks.append("cold_chain_breach")

        return risks

"""AI Assurance Module — Explainability, Calibration, Fairness, and Reliability.

This module answers the question every stakeholder asks about AI systems:
"Why should I trust this decision?"

It provides four pillars of AI assurance:

    1. EXPLAINABILITY — Why did the classifier assign this tier?
       Per-item decision explanations in plain language.

    2. CONFIDENCE CALIBRATION — When the model says 90% confident,
       is it right 90% of the time? Miscalibrated confidence means
       the system is overconfident or underconfident.

    3. FAIRNESS — Does the system perform equally well across all
       produce types? Are berries systematically mis-classified
       more than peppers?

    4. RELIABILITY — Under what conditions does the system fail?
       Edge cases, failure mode analysis, uncertainty quantification.

Why this matters in production:
    - Regulatory: Food safety regulators may require explainable decisions
    - Operational: Warehouse managers need to trust the system
    - Financial: Mis-dispatching a truck costs $2,000+; blind trust is expensive
    - Legal: If spoiled produce reaches a consumer, "the AI decided" is not a defense
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.models import (
    FreshnessAssessment,
    FreshnessTier,
    ScanResult,
)


# ── Assurance Models ────────────────────────────────────────────────────

class DecisionExplanation(BaseModel):
    """Human-readable explanation of a single classification decision."""
    item_id: str
    produce_type: str
    tier: FreshnessTier
    composite_score: float
    explanation: str = Field(description="Plain-language explanation of why this tier was assigned")
    top_factors: list[dict] = Field(
        default_factory=list,
        description="Ranked list of factors that most influenced the decision"
    )
    counterfactual: str = Field(
        default="",
        description="What would need to change for a different tier assignment"
    )
    confidence_assessment: str = Field(default="")


class CalibrationMetrics(BaseModel):
    """Confidence calibration analysis."""
    total_items: int = 0
    calibration_error: float = Field(
        default=0.0,
        description="Expected Calibration Error (ECE) — lower is better"
    )
    is_overconfident: bool = False
    is_underconfident: bool = False
    calibration_bins: list[dict] = Field(default_factory=list)
    recommendation: str = ""


class FairnessMetrics(BaseModel):
    """Fairness analysis across produce categories."""
    per_category_accuracy: dict[str, dict] = Field(default_factory=dict)
    most_disadvantaged: str = Field(default="", description="Category with worst performance")
    fairness_gap: float = Field(default=0.0, description="Gap between best and worst category")
    is_fair: bool = Field(default=True, description="True if gap < 10%")
    concerns: list[str] = Field(default_factory=list)


class ReliabilityProfile(BaseModel):
    """System reliability analysis."""
    overall_reliability: float = Field(ge=0.0, le=1.0, default=1.0)
    failure_modes: list[dict] = Field(default_factory=list)
    uncertainty_items: list[dict] = Field(
        default_factory=list,
        description="Items where the system is least certain"
    )
    edge_cases: list[dict] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class AssuranceReport(BaseModel):
    """Complete AI assurance report for a pipeline run."""
    explanations: list[DecisionExplanation] = Field(default_factory=list)
    calibration: CalibrationMetrics = Field(default_factory=CalibrationMetrics)
    fairness: FairnessMetrics = Field(default_factory=FairnessMetrics)
    reliability: ReliabilityProfile = Field(default_factory=ReliabilityProfile)
    overall_trust_score: float = Field(
        ge=0.0, le=1.0, default=1.0,
        description="Composite trust score — should the operator trust this batch?"
    )
    summary: str = ""


# ── AI Assurance Agent ──────────────────────────────────────────────────

class AIAssuranceAgent(BaseAgent):
    """Provides explainability, calibration, fairness, and reliability analysis.

    This agent runs AFTER classification and produces an assurance report
    that operators and auditors can review to understand and trust the
    system's decisions.
    """

    def __init__(self, config: dict):
        super().__init__("AIAssuranceAgent", config)
        self.weights = config.get("vision", {}).get("weights", {})
        self.thresholds = config.get("classification", {}).get("score_thresholds", {})

    def process(
        self,
        assessments: list[FreshnessAssessment],
        scan_results: list[ScanResult],
    ) -> AssuranceReport:
        """Generate a complete AI assurance report.

        Args:
            assessments: Classification results from the classifier agent.
            scan_results: Original scan data for feature-level analysis.

        Returns:
            AssuranceReport with explanations, calibration, fairness, and reliability.
        """
        self.clear_events()

        scan_lookup = {s.item_id: s for s in scan_results}

        report = AssuranceReport()

        # Pillar 1: Explainability
        report.explanations = self._generate_explanations(assessments, scan_lookup)

        # Pillar 2: Confidence calibration
        report.calibration = self._analyze_calibration(assessments)

        # Pillar 3: Fairness
        report.fairness = self._analyze_fairness(assessments)

        # Pillar 4: Reliability
        report.reliability = self._analyze_reliability(assessments, scan_lookup)

        # Overall trust score
        report.overall_trust_score = self._compute_trust_score(report)
        report.summary = self._generate_summary(report)

        self.emit_event(
            "assurance_complete",
            f"🛡️ AI Assurance: trust={report.overall_trust_score:.2f}, "
            f"calibration_error={report.calibration.calibration_error:.3f}, "
            f"fairness_gap={report.fairness.fairness_gap:.1%}, "
            f"{len(report.reliability.uncertainty_items)} uncertain items",
            {
                "trust_score": report.overall_trust_score,
                "calibration_error": report.calibration.calibration_error,
                "fairness_gap": report.fairness.fairness_gap,
            },
        )

        return report

    # ── Pillar 1: Explainability ────────────────────────────────────────

    def _generate_explanations(
        self,
        assessments: list[FreshnessAssessment],
        scan_lookup: dict[str, ScanResult],
    ) -> list[DecisionExplanation]:
        """Generate plain-language explanations for each classification."""
        explanations = []

        for assessment in assessments:
            scan = scan_lookup.get(assessment.item_id)
            if not scan:
                continue

            factors = self._rank_factors(scan, assessment)
            explanation = self._build_explanation(assessment, factors)
            counterfactual = self._build_counterfactual(assessment, factors)
            confidence_note = self._assess_confidence(assessment)

            explanations.append(DecisionExplanation(
                item_id=assessment.item_id,
                produce_type=assessment.produce_type,
                tier=assessment.tier,
                composite_score=assessment.composite_score,
                explanation=explanation,
                top_factors=factors[:3],
                counterfactual=counterfactual,
                confidence_assessment=confidence_note,
            ))

        return explanations

    def _rank_factors(self, scan: ScanResult, assessment: FreshnessAssessment) -> list[dict]:
        """Rank which features contributed most to the classification."""
        f = scan.scan_features
        w = self.weights

        contributions = [
            {
                "feature": "color",
                "value": f.color_score,
                "weight": w.get("color_score", 0.30),
                "contribution": round(f.color_score * w.get("color_score", 0.30), 3),
                "impact": "positive" if f.color_score > 0.5 else "negative",
            },
            {
                "feature": "firmness",
                "value": f.firmness_score,
                "weight": w.get("firmness_score", 0.25),
                "contribution": round(f.firmness_score * w.get("firmness_score", 0.25), 3),
                "impact": "positive" if f.firmness_score > 0.5 else "negative",
            },
            {
                "feature": "blemishes",
                "value": f.blemish_score,
                "weight": w.get("blemish_score", 0.20),
                "contribution": round(f.blemish_score * w.get("blemish_score", 0.20), 3),
                "impact": "positive" if f.blemish_score > 0.5 else "negative",
            },
            {
                "feature": "ethylene",
                "value": f.ethylene_ppm,
                "weight": w.get("ethylene_level", 0.15),
                "contribution": round(max(0, 1.0 - f.ethylene_ppm / 12.0) * w.get("ethylene_level", 0.15), 3),
                "impact": "negative" if f.ethylene_ppm > 6.0 else "positive",
            },
            {
                "feature": "temperature",
                "value": f.surface_temp_c,
                "weight": w.get("temperature_delta", 0.10),
                "contribution": round(max(0, 1.0 - abs(f.surface_temp_c - 4.0) / 10.0) * w.get("temperature_delta", 0.10), 3),
                "impact": "negative" if abs(f.surface_temp_c - 4.0) > 5.0 else "positive",
            },
        ]

        # Sort by absolute contribution — highest impact first
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return contributions

    def _build_explanation(self, assessment: FreshnessAssessment, factors: list[dict]) -> str:
        """Build a plain-language explanation."""
        tier_label = {
            FreshnessTier.SHIP_NOW: "SHIP NOW (immediate dispatch)",
            FreshnessTier.SHIP_SOON: "SHIP SOON (dispatch within 24h)",
            FreshnessTier.STORE: "STORE (hold in cold storage)",
        }[assessment.tier]

        top = factors[0]
        second = factors[1] if len(factors) > 1 else None

        explanation = (
            f"This {assessment.produce_type} ({assessment.variant}) was classified as "
            f"{tier_label} with a freshness score of {assessment.composite_score:.2f} "
            f"and an estimated {assessment.estimated_days_remaining:.1f} days remaining. "
            f"The primary factor was {top['feature']} "
            f"(score: {top['value']:.2f}, contributing {top['contribution']:.3f} "
            f"to the composite)."
        )

        if second and second["impact"] == "negative":
            explanation += (
                f" {second['feature'].capitalize()} was also a concern "
                f"(score: {second['value']:.2f})."
            )

        if assessment.risk_factors:
            explanation += f" Risk factors detected: {', '.join(assessment.risk_factors)}."

        return explanation

    def _build_counterfactual(self, assessment: FreshnessAssessment, factors: list[dict]) -> str:
        """Explain what would need to change for a different classification."""
        critical = self.thresholds.get("critical", 0.35)
        warning = self.thresholds.get("warning", 0.65)

        if assessment.tier == FreshnessTier.SHIP_NOW:
            gap = critical - assessment.composite_score
            return (
                f"To move to SHIP SOON, the composite score would need to increase "
                f"by {abs(gap):.2f} (from {assessment.composite_score:.2f} to above "
                f"{critical:.2f}). This would require improvement in "
                f"{factors[-1]['feature']} (currently {factors[-1]['value']:.2f})."
            )
        elif assessment.tier == FreshnessTier.SHIP_SOON:
            gap_up = warning - assessment.composite_score
            gap_down = assessment.composite_score - critical
            return (
                f"Score is {gap_down:.2f} above SHIP NOW threshold and {gap_up:.2f} "
                f"below STORE threshold. A decline of {gap_down:.2f} in the weakest "
                f"feature ({factors[-1]['feature']}) would trigger immediate dispatch."
            )
        else:  # STORE
            gap = assessment.composite_score - warning
            return (
                f"Score is {gap:.2f} above the SHIP SOON threshold. This item is "
                f"safely in the STORE tier but should be re-scanned in 48 hours as "
                f"it continues to degrade."
            )

    def _assess_confidence(self, assessment: FreshnessAssessment) -> str:
        """Assess how confident we should be in this specific classification."""
        critical = self.thresholds.get("critical", 0.35)
        warning = self.thresholds.get("warning", 0.65)

        # Distance from nearest threshold boundary
        distances = [
            abs(assessment.composite_score - critical),
            abs(assessment.composite_score - warning),
        ]
        min_distance = min(distances)

        if min_distance < 0.05:
            return (
                f"LOW CONFIDENCE: Score {assessment.composite_score:.3f} is very close "
                f"to a tier boundary (within {min_distance:.3f}). Small measurement "
                f"errors could change the classification. Consider manual review."
            )
        elif min_distance < 0.10:
            return (
                f"MODERATE CONFIDENCE: Score is near a tier boundary "
                f"(distance: {min_distance:.3f}). Classification is likely correct "
                f"but should be verified if this is a high-value item."
            )
        else:
            return (
                f"HIGH CONFIDENCE: Score {assessment.composite_score:.3f} is well "
                f"within the {assessment.tier.label} range "
                f"(distance from nearest boundary: {min_distance:.3f})."
            )

    # ── Pillar 2: Confidence Calibration ────────────────────────────────

    def _analyze_calibration(self, assessments: list[FreshnessAssessment]) -> CalibrationMetrics:
        """Analyze whether confidence scores are well-calibrated.

        A well-calibrated model should be right 90% of the time when
        it says it's 90% confident. We use the classifier's confidence
        field as the probability estimate.
        """
        if len(assessments) < 5:
            return CalibrationMetrics(total_items=len(assessments))

        # Bin items by confidence level
        bins = {"0.6-0.7": [], "0.7-0.8": [], "0.8-0.9": [], "0.9-1.0": []}

        for a in assessments:
            conf = a.confidence
            if conf < 0.7:
                bins["0.6-0.7"].append(a)
            elif conf < 0.8:
                bins["0.7-0.8"].append(a)
            elif conf < 0.9:
                bins["0.8-0.9"].append(a)
            else:
                bins["0.9-1.0"].append(a)

        calibration_bins = []
        total_ece = 0.0
        total_items = len(assessments)

        for bin_name, items in bins.items():
            if not items:
                continue

            avg_confidence = statistics.mean(i.confidence for i in items)
            # Proxy for accuracy: items far from tier boundaries are more likely correct
            critical = self.thresholds.get("critical", 0.35)
            warning = self.thresholds.get("warning", 0.65)
            proxy_accuracy = statistics.mean(
                1.0 if min(abs(i.composite_score - critical),
                          abs(i.composite_score - warning)) > 0.1 else 0.5
                for i in items
            )

            bin_error = abs(avg_confidence - proxy_accuracy)
            total_ece += bin_error * (len(items) / total_items)

            calibration_bins.append({
                "bin": bin_name,
                "count": len(items),
                "avg_confidence": round(avg_confidence, 3),
                "proxy_accuracy": round(proxy_accuracy, 3),
                "calibration_error": round(bin_error, 3),
            })

        # Determine if over/underconfident
        high_conf_items = bins.get("0.9-1.0", [])
        if high_conf_items:
            high_conf_boundary_items = [
                i for i in high_conf_items
                if min(abs(i.composite_score - 0.35),
                       abs(i.composite_score - 0.65)) < 0.08
            ]
            is_overconfident = len(high_conf_boundary_items) / len(high_conf_items) > 0.3
        else:
            is_overconfident = False

        recommendation = ""
        if total_ece > 0.15:
            recommendation = (
                "Calibration error is high. Consider applying temperature scaling "
                "or Platt scaling to the confidence scores before using them for "
                "downstream decisions."
            )
        elif is_overconfident:
            recommendation = (
                "System is overconfident — reporting high confidence even for "
                "items near tier boundaries. Reduce confidence for borderline items."
            )

        return CalibrationMetrics(
            total_items=total_items,
            calibration_error=round(total_ece, 4),
            is_overconfident=is_overconfident,
            is_underconfident=False,
            calibration_bins=calibration_bins,
            recommendation=recommendation,
        )

    # ── Pillar 3: Fairness ──────────────────────────────────────────────

    def _analyze_fairness(self, assessments: list[FreshnessAssessment]) -> FairnessMetrics:
        """Analyze classification fairness across produce categories.

        A fair system should classify all produce types with similar
        accuracy. If berries are systematically mis-classified while
        peppers are always correct, the system is biased.
        """
        by_category: dict[str, list[FreshnessAssessment]] = defaultdict(list)
        for a in assessments:
            by_category[a.produce_type].append(a)

        per_category = {}
        for category, items in by_category.items():
            avg_score = statistics.mean(i.composite_score for i in items)
            avg_confidence = statistics.mean(i.confidence for i in items)
            tier_distribution = defaultdict(int)
            for i in items:
                tier_distribution[i.tier.value] += 1

            # Proxy for reliability: items near boundaries are riskier
            boundary_items = sum(
                1 for i in items
                if min(abs(i.composite_score - 0.35),
                       abs(i.composite_score - 0.65)) < 0.08
            )
            boundary_ratio = boundary_items / len(items) if items else 0

            per_category[category] = {
                "count": len(items),
                "avg_score": round(avg_score, 3),
                "avg_confidence": round(avg_confidence, 3),
                "tier_distribution": dict(tier_distribution),
                "boundary_risk_ratio": round(boundary_ratio, 3),
                "risk_factor_rate": round(
                    sum(1 for i in items if i.risk_factors) / len(items), 3
                ),
            }

        # Find fairness gap
        if len(per_category) >= 2:
            confidences = {k: v["avg_confidence"] for k, v in per_category.items()}
            best = max(confidences.values())
            worst = min(confidences.values())
            gap = best - worst
            most_disadvantaged = min(confidences, key=confidences.get)
        else:
            gap = 0.0
            most_disadvantaged = ""

        concerns = []
        for category, metrics in per_category.items():
            if metrics["boundary_risk_ratio"] > 0.4:
                concerns.append(
                    f"{category}: {metrics['boundary_risk_ratio']:.0%} of items are near "
                    f"tier boundaries — classification is uncertain for this category"
                )
            if metrics["avg_confidence"] < 0.80:
                concerns.append(
                    f"{category}: average confidence is {metrics['avg_confidence']:.2f} — "
                    f"below 0.80 threshold. Scanner may need calibration for this produce type."
                )

        return FairnessMetrics(
            per_category_accuracy=per_category,
            most_disadvantaged=most_disadvantaged,
            fairness_gap=round(gap, 4),
            is_fair=gap < 0.10,
            concerns=concerns,
        )

    # ── Pillar 4: Reliability ───────────────────────────────────────────

    def _analyze_reliability(
        self,
        assessments: list[FreshnessAssessment],
        scan_lookup: dict[str, ScanResult],
    ) -> ReliabilityProfile:
        """Analyze system reliability — where does it fail?"""
        failure_modes = []
        uncertainty_items = []
        edge_cases = []

        critical = self.thresholds.get("critical", 0.35)
        warning = self.thresholds.get("warning", 0.65)

        for a in assessments:
            min_dist = min(abs(a.composite_score - critical),
                          abs(a.composite_score - warning))

            # Uncertainty: items near decision boundaries
            if min_dist < 0.05:
                uncertainty_items.append({
                    "item_id": a.item_id,
                    "produce_type": a.produce_type,
                    "score": a.composite_score,
                    "distance_to_boundary": round(min_dist, 4),
                    "assigned_tier": a.tier.value,
                    "note": "Very close to tier boundary — small errors could change classification",
                })

            # Edge cases: contradictory signals
            scan = scan_lookup.get(a.item_id)
            if scan:
                f = scan.scan_features
                if f.color_score > 0.8 and f.firmness_score < 0.3:
                    edge_cases.append({
                        "item_id": a.item_id,
                        "produce_type": a.produce_type,
                        "description": "Surface looks fresh but structure is degraded — "
                                       "possible internal decay not visible to scanner",
                        "recommendation": "Manual inspection recommended",
                    })

                if f.ethylene_ppm > 10.0 and a.tier == FreshnessTier.STORE:
                    edge_cases.append({
                        "item_id": a.item_id,
                        "produce_type": a.produce_type,
                        "description": "Tagged STORE despite high ethylene — "
                                       "may degrade much faster than estimated",
                        "recommendation": "Re-scan in 24h instead of 48h",
                    })

        # Failure modes
        if len(uncertainty_items) > len(assessments) * 0.2:
            failure_modes.append({
                "mode": "boundary_clustering",
                "description": f"{len(uncertainty_items)} items ({len(uncertainty_items)/len(assessments):.0%}) "
                               f"are near tier boundaries. This batch has high classification uncertainty.",
                "mitigation": "Consider adding a 'REVIEW' tier for borderline items, or "
                              "narrowing the SHIP_SOON range to reduce ambiguity.",
            })

        if edge_cases:
            failure_modes.append({
                "mode": "contradictory_features",
                "description": f"{len(edge_cases)} items have contradictory feature readings",
                "mitigation": "Enhanced vision agent's multi-pass scanning helps, "
                              "but some items may need manual inspection.",
            })

        overall = 1.0
        if uncertainty_items:
            overall -= min(0.3, len(uncertainty_items) / len(assessments) * 0.5)
        if edge_cases:
            overall -= min(0.2, len(edge_cases) / len(assessments) * 0.3)

        recommendations = []
        if overall < 0.8:
            recommendations.append(
                "System reliability is below 80% for this batch. "
                "Consider manual spot-checks on flagged items."
            )
        if len(uncertainty_items) > 5:
            recommendations.append(
                f"Review {len(uncertainty_items)} borderline items before dispatch."
            )

        return ReliabilityProfile(
            overall_reliability=round(max(0.0, overall), 3),
            failure_modes=failure_modes,
            uncertainty_items=uncertainty_items[:10],  # Top 10 most uncertain
            edge_cases=edge_cases[:10],
            recommendations=recommendations,
        )

    # ── Trust Score ─────────────────────────────────────────────────────

    def _compute_trust_score(self, report: AssuranceReport) -> float:
        """Compute overall trust score from all four pillars."""
        calibration_score = max(0.0, 1.0 - report.calibration.calibration_error * 3)
        fairness_score = 1.0 if report.fairness.is_fair else max(0.5, 1.0 - report.fairness.fairness_gap)
        reliability_score = report.reliability.overall_reliability

        # Weighted average — reliability matters most
        trust = (
            calibration_score * 0.25
            + fairness_score * 0.25
            + reliability_score * 0.50
        )

        return round(max(0.0, min(1.0, trust)), 3)

    def _generate_summary(self, report: AssuranceReport) -> str:
        """Generate executive summary of the assurance report."""
        trust = report.overall_trust_score

        if trust >= 0.85:
            status = "HIGH TRUST — system decisions can be relied upon for this batch"
        elif trust >= 0.70:
            status = "MODERATE TRUST — most decisions are reliable, review flagged items"
        else:
            status = "LOW TRUST — significant concerns detected, manual review recommended"

        return (
            f"AI Assurance Report: {status}. "
            f"Trust score: {trust:.2f}. "
            f"Calibration error: {report.calibration.calibration_error:.3f}. "
            f"Fairness gap: {report.fairness.fairness_gap:.1%}. "
            f"Reliability: {report.reliability.overall_reliability:.2f}. "
            f"Uncertain items: {len(report.reliability.uncertainty_items)}. "
            f"Edge cases: {len(report.reliability.edge_cases)}."
        )

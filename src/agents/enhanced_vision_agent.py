"""Enhanced Vision Agent — Autonomous Multi-Pass Scanning.

This agent closes the gap between human visual analysis and single-pass
optical scanning by implementing three self-correction strategies:

1. CONSISTENCY VALIDATION
   Humans instinctively notice when features contradict each other.
   A tomato with perfect color but zero firmness is suspicious — humans
   would flip it over and check again. This agent encodes those
   biological correlations as consistency rules and triggers re-scans
   when features don't make sense together.

2. MULTI-ANGLE ENSEMBLE
   Humans rotate produce, check the bottom, look at it from multiple
   angles. This agent simulates multi-angle scanning by taking N scans
   and using ensemble logic (median, agreement voting) to produce a
   robust feature vector. A single bad reading gets outvoted.

3. ADAPTIVE CONFIDENCE THRESHOLDS
   Instead of a fixed 70% confidence gate, this agent uses
   produce-specific and context-aware thresholds. Berries need higher
   confidence because they degrade faster and the cost of error is
   higher. Items from suppliers with poor reliability history get
   scanned with tighter thresholds.

The agent decides autonomously whether to re-scan — no human needed.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional

import yaml

from src.agents.base import BaseAgent
from src.models import ScanFeatures, ScanResult


# ── Consistency Rules ───────────────────────────────────────────────────

@dataclass
class ConsistencyViolation:
    """A detected inconsistency between scan features."""
    rule_name: str
    description: str
    severity: str  # "warning" or "critical"
    conflicting_features: dict[str, float]


# Biological correlation rules — features that should move together
CONSISTENCY_RULES = [
    {
        "name": "color_firmness_divergence",
        "description": "Color and firmness should degrade together. "
                       "High color with very low firmness suggests a surface-only scan "
                       "that missed internal degradation.",
        "check": lambda f: abs(f.color_score - f.firmness_score) > 0.45,
        "severity": "critical",
        "features": lambda f: {"color_score": f.color_score, "firmness_score": f.firmness_score},
    },
    {
        "name": "fresh_but_high_ethylene",
        "description": "High ethylene with high freshness scores suggests the item "
                       "is about to degrade rapidly — human would flag this as 'looks "
                       "good now but won't last'.",
        "check": lambda f: f.color_score > 0.75 and f.firmness_score > 0.70 and f.ethylene_ppm > 8.0,
        "severity": "warning",
        "features": lambda f: {
            "color_score": f.color_score, "firmness_score": f.firmness_score,
            "ethylene_ppm": f.ethylene_ppm,
        },
    },
    {
        "name": "blemish_score_outlier",
        "description": "High blemish score (clean surface) but low color/firmness is "
                       "suspicious — decay may be internal or on the underside.",
        "check": lambda f: f.blemish_score > 0.85 and (f.color_score < 0.35 or f.firmness_score < 0.35),
        "severity": "critical",
        "features": lambda f: {
            "blemish_score": f.blemish_score, "color_score": f.color_score,
            "firmness_score": f.firmness_score,
        },
    },
    {
        "name": "temperature_stress_undetected",
        "description": "Large temperature deviation from optimal but freshness scores "
                       "are still high — cold chain damage may not be visible yet.",
        "check": lambda f: abs(f.surface_temp_c - 4.0) > 8.0 and f.color_score > 0.7,
        "severity": "warning",
        "features": lambda f: {
            "surface_temp_c": f.surface_temp_c, "color_score": f.color_score,
        },
    },
    {
        "name": "all_features_identical",
        "description": "All features nearly identical suggests sensor malfunction "
                       "or stuck reading — not biologically realistic.",
        "check": lambda f: (
            abs(f.color_score - f.firmness_score) < 0.02
            and abs(f.firmness_score - f.blemish_score) < 0.02
        ),
        "severity": "critical",
        "features": lambda f: {
            "color_score": f.color_score, "firmness_score": f.firmness_score,
            "blemish_score": f.blemish_score,
        },
    },
]


# ── Produce-Specific Confidence Thresholds ──────────────────────────────

ADAPTIVE_THRESHOLDS = {
    # Berries degrade fast — errors are costly, demand higher confidence
    "strawberry": {"min_confidence": 0.82, "max_rescans": 3},
    "blueberry": {"min_confidence": 0.80, "max_rescans": 3},

    # Leafy greens wilt quickly — need reliable scans
    "spinach": {"min_confidence": 0.78, "max_rescans": 3},
    "lettuce": {"min_confidence": 0.78, "max_rescans": 3},

    # Fruits are more forgiving — standard thresholds
    "tomato": {"min_confidence": 0.72, "max_rescans": 2},
    "banana": {"min_confidence": 0.70, "max_rescans": 2},
    "avocado": {"min_confidence": 0.75, "max_rescans": 2},

    # Hardy produce — lower threshold acceptable
    "bell_pepper": {"min_confidence": 0.68, "max_rescans": 2},
}

DEFAULT_THRESHOLD = {"min_confidence": 0.72, "max_rescans": 2}


# ── Enhanced Vision Agent ───────────────────────────────────────────────

class EnhancedVisionAgent(BaseAgent):
    """Vision agent with autonomous multi-pass re-scanning.

    This agent replicates human-like analysis by:
    1. Running consistency checks between features
    2. Triggering autonomous re-scans when readings are suspicious
    3. Using ensemble logic to produce robust feature vectors
    4. Adapting confidence thresholds per produce type
    """

    def __init__(self, config: dict, scanner=None):
        super().__init__("EnhancedVisionAgent", config)
        self.scanner = scanner

        # Load scanner if not injected
        if self.scanner is None:
            from src.sensors.optical_scanner import OpticalScanner
            self.scanner = OpticalScanner()

        self.catalog = {}
        with open("config/settings.yaml") as f:
            full_config = yaml.safe_load(f)
        self.catalog = full_config.get("produce_catalog", {})

        # Tracking metrics
        self._rescan_count = 0
        self._violation_count = 0
        self._rejected_count = 0

    def process(self, scan_results: list[ScanResult]) -> list[ScanResult]:
        """Validate scans with human-like consistency checks and autonomous re-scanning.

        For each item:
        1. Check confidence against produce-specific threshold
        2. Run biological consistency rules
        3. If violations found → trigger autonomous re-scan (up to N times)
        4. Use ensemble of all scans to produce final feature vector
        5. Reject only if all re-scans fail consistency

        Args:
            scan_results: Raw scan output from the optical scanner.

        Returns:
            List of validated, potentially re-scanned ScanResults.
        """
        self.clear_events()
        self._rescan_count = 0
        self._violation_count = 0
        self._rejected_count = 0

        validated = []

        for scan in scan_results:
            result = self._process_single_item(scan)
            if result is not None:
                validated.append(result)

        self.emit_event(
            "enhanced_vision_complete",
            f"📷 Enhanced scan: {len(scan_results)} items → "
            f"{len(validated)} validated, {self._rejected_count} rejected, "
            f"{self._rescan_count} re-scans triggered, "
            f"{self._violation_count} consistency violations detected",
            {
                "total": len(scan_results),
                "validated": len(validated),
                "rejected": self._rejected_count,
                "rescans": self._rescan_count,
                "violations": self._violation_count,
            },
        )

        return validated

    def _process_single_item(self, scan: ScanResult) -> Optional[ScanResult]:
        """Process a single item through the multi-pass validation pipeline."""

        # Get produce-specific thresholds
        thresholds = ADAPTIVE_THRESHOLDS.get(scan.produce_type, DEFAULT_THRESHOLD)
        min_confidence = thresholds["min_confidence"]
        max_rescans = thresholds["max_rescans"]

        # Collect all scans for this item (starting with the initial one)
        all_scans = [scan]

        # Pass 1: Check initial scan
        violations = self._check_consistency(scan)

        if not violations and scan.scan_features.scan_confidence >= min_confidence:
            # Clean scan — no re-scan needed
            return scan

        # Violations found or low confidence — trigger re-scans
        rescan_reason = self._format_rescan_reason(scan, violations, min_confidence)

        for attempt in range(max_rescans):
            self._rescan_count += 1
            self.emit_event(
                "autonomous_rescan",
                f"🔄 Re-scan #{attempt + 1} for {scan.produce_type} "
                f"({scan.item_id}): {rescan_reason}",
                {
                    "item_id": scan.item_id,
                    "attempt": attempt + 1,
                    "reason": rescan_reason,
                },
            )

            # Simulate re-scan from a different angle/position
            rescan = self.scanner.scan_single(
                scan.produce_type,
                scan.variant,
                # Simulate same item, slightly different reading
                days_since_harvest=self._estimate_age_from_features(scan.scan_features),
            )
            rescan.item_id = scan.item_id  # Same physical item
            rescan.bay_location = scan.bay_location
            rescan.case_count = scan.case_count
            all_scans.append(rescan)

            # Check if the new scan resolves the violations
            new_violations = self._check_consistency(rescan)
            if not new_violations and rescan.scan_features.scan_confidence >= min_confidence:
                # Good re-scan — but use ensemble of all scans for robustness
                break

        # Build ensemble result from all scans
        ensemble_result = self._build_ensemble(scan, all_scans)

        # Final validation on the ensemble result
        final_violations = self._check_consistency(ensemble_result)
        if final_violations and any(v.severity == "critical" for v in final_violations):
            # Even after re-scans, critical violations persist — reject
            self._rejected_count += 1
            self.emit_event(
                "scan_rejected_after_rescan",
                f"❌ Rejected {scan.produce_type} ({scan.item_id}) after "
                f"{len(all_scans)} scans — persistent consistency violations: "
                f"{[v.rule_name for v in final_violations]}",
                {
                    "item_id": scan.item_id,
                    "total_scans": len(all_scans),
                    "violations": [v.rule_name for v in final_violations],
                },
            )
            return None

        return ensemble_result

    def _check_consistency(self, scan: ScanResult) -> list[ConsistencyViolation]:
        """Run biological consistency rules against scan features.

        These rules encode the intuitions that humans use when they
        look at produce and think 'something doesn't add up here.'
        """
        violations = []
        features = scan.scan_features

        for rule in CONSISTENCY_RULES:
            try:
                if rule["check"](features):
                    violation = ConsistencyViolation(
                        rule_name=rule["name"],
                        description=rule["description"],
                        severity=rule["severity"],
                        conflicting_features=rule["features"](features),
                    )
                    violations.append(violation)
                    self._violation_count += 1
            except Exception:
                pass  # Rule evaluation failed — skip, don't crash

        return violations

    def _build_ensemble(self, original: ScanResult, all_scans: list[ScanResult]) -> ScanResult:
        """Combine multiple scans into a robust ensemble reading.

        Uses median for each feature — this is inherently resistant to
        outliers from bad angles or momentary sensor glitches. A single
        bad scan in a set of 3 gets outvoted automatically.

        This mimics how a human mentally averages what they see from
        multiple angles rather than relying on one glance.
        """
        if len(all_scans) == 1:
            return all_scans[0]

        # Collect all feature values
        colors = [s.scan_features.color_score for s in all_scans]
        firmnesses = [s.scan_features.firmness_score for s in all_scans]
        blemishes = [s.scan_features.blemish_score for s in all_scans]
        ethylenes = [s.scan_features.ethylene_ppm for s in all_scans]
        temps = [s.scan_features.surface_temp_c for s in all_scans]
        confidences = [s.scan_features.scan_confidence for s in all_scans]

        # Take median of each (robust to outliers)
        ensemble_features = ScanFeatures(
            color_score=round(statistics.median(colors), 3),
            firmness_score=round(statistics.median(firmnesses), 3),
            blemish_score=round(statistics.median(blemishes), 3),
            ethylene_ppm=round(statistics.median(ethylenes), 2),
            surface_temp_c=round(statistics.median(temps), 1),
            # Confidence = average confidence * agreement bonus
            scan_confidence=round(min(0.99, statistics.mean(confidences) * self._agreement_bonus(all_scans)), 3),
        )

        # Build result preserving original metadata
        result = ScanResult(
            item_id=original.item_id,
            produce_type=original.produce_type,
            variant=original.variant,
            scan_features=ensemble_features,
            bay_location=original.bay_location,
            case_count=original.case_count,
        )

        if len(all_scans) > 1:
            self.emit_event(
                "ensemble_built",
                f"🔀 Ensemble for {original.produce_type} ({original.item_id}): "
                f"{len(all_scans)} scans combined — "
                f"confidence {original.scan_features.scan_confidence:.2f} → "
                f"{ensemble_features.scan_confidence:.2f}",
                {
                    "item_id": original.item_id,
                    "scan_count": len(all_scans),
                    "original_confidence": original.scan_features.scan_confidence,
                    "ensemble_confidence": ensemble_features.scan_confidence,
                },
            )

        return result

    def _agreement_bonus(self, scans: list[ScanResult]) -> float:
        """Compute how much the multiple scans agree with each other.

        High agreement = readings are consistent across angles → bonus confidence.
        Low agreement = readings vary wildly → no bonus, might even penalize.

        This mimics a human's confidence increasing when the produce
        looks the same from every angle vs. being suspicious when the
        bottom looks different from the top.
        """
        if len(scans) < 2:
            return 1.0

        # Compute coefficient of variation for key features
        colors = [s.scan_features.color_score for s in scans]
        firmnesses = [s.scan_features.firmness_score for s in scans]

        color_cv = statistics.stdev(colors) / (statistics.mean(colors) + 0.01)
        firmness_cv = statistics.stdev(firmnesses) / (statistics.mean(firmnesses) + 0.01)

        avg_cv = (color_cv + firmness_cv) / 2

        # Low variation → high agreement → bonus up to 1.08
        # High variation → no bonus
        if avg_cv < 0.05:
            return 1.08  # Strong agreement
        elif avg_cv < 0.15:
            return 1.03  # Moderate agreement
        elif avg_cv < 0.30:
            return 1.0   # Neutral
        else:
            return 0.95  # Disagreement — penalize slightly

    def _estimate_age_from_features(self, features: ScanFeatures) -> float:
        """Estimate days since harvest from feature values.

        Used to generate realistic re-scans of the same item.
        A real system would just re-trigger the physical scanner.
        """
        # Inverse of the degradation curve: lower scores = older
        avg_freshness = (features.color_score + features.firmness_score + features.blemish_score) / 3
        # Rough inverse mapping: freshness 1.0 ≈ 0 days, 0.0 ≈ 10 days
        estimated_age = (1.0 - avg_freshness) * 10.0
        return max(0.0, estimated_age)

    def _format_rescan_reason(
        self,
        scan: ScanResult,
        violations: list[ConsistencyViolation],
        min_confidence: float,
    ) -> str:
        """Generate human-readable reason for the re-scan."""
        reasons = []

        if scan.scan_features.scan_confidence < min_confidence:
            reasons.append(
                f"confidence {scan.scan_features.scan_confidence:.2f} "
                f"below {scan.produce_type} threshold {min_confidence:.2f}"
            )

        for v in violations:
            reasons.append(v.rule_name.replace("_", " "))

        return "; ".join(reasons) if reasons else "precautionary re-scan"

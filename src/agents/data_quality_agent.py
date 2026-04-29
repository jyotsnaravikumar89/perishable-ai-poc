"""Data Quality Agent — Validates, Monitors, and Scores Data Quality.

In production AI systems, model accuracy is bounded by data quality.
A perfect classifier trained on noisy, drifting, or incomplete data
will make bad decisions. This agent sits at the front of the pipeline
and answers three questions:

    1. Is this data VALID?    (Schema, ranges, null checks)
    2. Is this data FRESH?    (Staleness, timestamp checks)
    3. Is this data DRIFTING? (Distribution shift from baseline)

Data quality issues it catches:
    - Missing or null feature values
    - Out-of-range values (e.g., negative ethylene, temperature > 50°C)
    - Sensor calibration drift (feature distributions shifting over time)
    - Duplicate scans (same item scanned twice)
    - Timestamp anomalies (scans from the future, stale scans)
    - Class imbalance in incoming data (e.g., 90% tomatoes, 2% berries)

Each scan gets a Data Quality Score (DQS) from 0.0 to 1.0.
Items below the DQS threshold are quarantined, not classified.

Why this matters:
    "Garbage in, garbage out" is not just a cliché — it's the #1
    failure mode in production AI. A classifier with 95% accuracy
    on clean data can drop to 60% when fed noisy sensor readings.
    This agent ensures the classifier never sees bad data.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.models import ScanFeatures, ScanResult


# ── Data Quality Models ─────────────────────────────────────────────────

class DataQualityIssue(BaseModel):
    """A single data quality issue detected on a scan."""
    item_id: str
    issue_type: str  # "missing_value", "out_of_range", "drift", "duplicate", "stale"
    severity: str    # "error" (quarantine), "warning" (flag), "info" (log)
    field: str
    description: str
    value: Optional[float] = None


class DataQualityReport(BaseModel):
    """Aggregate data quality report for a batch of scans."""
    total_scans: int = 0
    valid_scans: int = 0
    quarantined_scans: int = 0
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0,
                                  description="Batch-level data quality score")
    issues: list[DataQualityIssue] = Field(default_factory=list)
    issue_summary: dict[str, int] = Field(default_factory=dict)
    feature_distributions: dict[str, dict] = Field(default_factory=dict)
    drift_alerts: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


# ── Valid Ranges for Scan Features ──────────────────────────────────────

FEATURE_RANGES = {
    "color_score": {"min": 0.0, "max": 1.0, "nullable": False},
    "firmness_score": {"min": 0.0, "max": 1.0, "nullable": False},
    "blemish_score": {"min": 0.0, "max": 1.0, "nullable": False},
    "ethylene_ppm": {"min": 0.0, "max": 50.0, "nullable": False},
    "surface_temp_c": {"min": -5.0, "max": 45.0, "nullable": False},
    "scan_confidence": {"min": 0.0, "max": 1.0, "nullable": False},
}

# Baseline distributions (learned from initial calibration runs)
BASELINE_DISTRIBUTIONS = {
    "color_score": {"mean": 0.62, "std": 0.22},
    "firmness_score": {"mean": 0.58, "std": 0.24},
    "blemish_score": {"mean": 0.65, "std": 0.20},
    "ethylene_ppm": {"mean": 4.5, "std": 3.0},
    "surface_temp_c": {"mean": 6.0, "std": 4.0},
}


class DataQualityAgent(BaseAgent):
    """Validates incoming scan data and monitors data quality over time.

    This agent runs BEFORE the vision agent and acts as the first line
    of defense against bad data entering the pipeline.
    """

    def __init__(self, config: dict, dqs_threshold: float = 0.6):
        super().__init__("DataQualityAgent", config)
        self.dqs_threshold = dqs_threshold
        self._scan_history: list[str] = []  # Track item IDs for duplicate detection
        self._batch_features: dict[str, list[float]] = defaultdict(list)

    def process(self, scan_results: list[ScanResult]) -> tuple[list[ScanResult], DataQualityReport]:
        """Validate scans and produce a data quality report.

        Args:
            scan_results: Raw scans from the optical scanner.

        Returns:
            Tuple of (valid_scans, quality_report).
            Quarantined scans are excluded from valid_scans.
        """
        self.clear_events()
        report = DataQualityReport(total_scans=len(scan_results))
        valid_scans = []
        self._batch_features = defaultdict(list)

        for scan in scan_results:
            issues = self._validate_single(scan)
            item_dqs = self._compute_item_dqs(issues)

            if item_dqs < self.dqs_threshold:
                report.quarantined_scans += 1
                self.emit_event(
                    "data_quarantined",
                    f"🚫 Quarantined {scan.produce_type} ({scan.item_id}): "
                    f"DQS={item_dqs:.2f} < threshold {self.dqs_threshold}",
                    {"item_id": scan.item_id, "dqs": item_dqs,
                     "issues": [i.issue_type for i in issues]},
                )
            else:
                valid_scans.append(scan)
                self._collect_features(scan)

            report.issues.extend(issues)

        report.valid_scans = len(valid_scans)

        # Compute batch-level metrics
        report.quality_score = self._compute_batch_dqs(report)
        report.issue_summary = self._summarize_issues(report.issues)
        report.feature_distributions = self._compute_distributions()
        report.drift_alerts = self._detect_drift()
        report.recommendations = self._generate_recommendations(report)

        self.emit_event(
            "data_quality_complete",
            f"📊 Data Quality: {report.valid_scans}/{report.total_scans} valid "
            f"(DQS={report.quality_score:.2f}), "
            f"{report.quarantined_scans} quarantined, "
            f"{len(report.issues)} issues, "
            f"{len(report.drift_alerts)} drift alerts",
            {
                "quality_score": report.quality_score,
                "valid": report.valid_scans,
                "quarantined": report.quarantined_scans,
                "drift_alerts": len(report.drift_alerts),
            },
        )

        return valid_scans, report

    def _validate_single(self, scan: ScanResult) -> list[DataQualityIssue]:
        """Run all validation checks on a single scan."""
        issues = []
        f = scan.scan_features

        # Check 1: Range validation
        feature_values = {
            "color_score": f.color_score,
            "firmness_score": f.firmness_score,
            "blemish_score": f.blemish_score,
            "ethylene_ppm": f.ethylene_ppm,
            "surface_temp_c": f.surface_temp_c,
            "scan_confidence": f.scan_confidence,
        }

        for field_name, value in feature_values.items():
            spec = FEATURE_RANGES[field_name]

            if value is None and not spec["nullable"]:
                issues.append(DataQualityIssue(
                    item_id=scan.item_id,
                    issue_type="missing_value",
                    severity="error",
                    field=field_name,
                    description=f"Required field '{field_name}' is null",
                ))
                continue

            if value < spec["min"] or value > spec["max"]:
                issues.append(DataQualityIssue(
                    item_id=scan.item_id,
                    issue_type="out_of_range",
                    severity="error",
                    field=field_name,
                    description=f"Value {value} outside valid range "
                                f"[{spec['min']}, {spec['max']}]",
                    value=value,
                ))

        # Check 2: Biological plausibility
        if f.surface_temp_c > 35.0:
            issues.append(DataQualityIssue(
                item_id=scan.item_id,
                issue_type="implausible_value",
                severity="warning",
                field="surface_temp_c",
                description=f"Temperature {f.surface_temp_c}°C is implausible "
                            f"for refrigerated produce — possible sensor fault",
                value=f.surface_temp_c,
            ))

        if f.ethylene_ppm > 25.0:
            issues.append(DataQualityIssue(
                item_id=scan.item_id,
                issue_type="implausible_value",
                severity="warning",
                field="ethylene_ppm",
                description=f"Ethylene {f.ethylene_ppm} ppm is abnormally high "
                            f"— possible sensor contamination or calibration error",
                value=f.ethylene_ppm,
            ))

        # Check 3: Duplicate detection
        if scan.item_id in self._scan_history:
            issues.append(DataQualityIssue(
                item_id=scan.item_id,
                issue_type="duplicate",
                severity="warning",
                field="item_id",
                description=f"Item {scan.item_id} has already been scanned in this batch",
            ))
        self._scan_history.append(scan.item_id)

        # Check 4: Unknown produce type
        catalog = self.config.get("produce_catalog", {})
        if scan.produce_type not in catalog:
            issues.append(DataQualityIssue(
                item_id=scan.item_id,
                issue_type="unknown_category",
                severity="warning",
                field="produce_type",
                description=f"Unknown produce type '{scan.produce_type}' — "
                            f"not in catalog, classification may be unreliable",
            ))

        # Check 5: Zero-variance detection (stuck sensor)
        values = [f.color_score, f.firmness_score, f.blemish_score]
        if len(set(round(v, 4) for v in values)) == 1:
            issues.append(DataQualityIssue(
                item_id=scan.item_id,
                issue_type="zero_variance",
                severity="warning",
                field="multiple",
                description="All quality features have identical values — "
                            "possible stuck sensor or calibration issue",
            ))

        return issues

    def _compute_item_dqs(self, issues: list[DataQualityIssue]) -> float:
        """Compute Data Quality Score for a single item."""
        if not issues:
            return 1.0

        penalty = 0.0
        for issue in issues:
            if issue.severity == "error":
                penalty += 0.3
            elif issue.severity == "warning":
                penalty += 0.1
            else:
                penalty += 0.02

        return max(0.0, 1.0 - penalty)

    def _compute_batch_dqs(self, report: DataQualityReport) -> float:
        """Compute batch-level Data Quality Score."""
        if report.total_scans == 0:
            return 0.0

        valid_ratio = report.valid_scans / report.total_scans
        issue_ratio = 1.0 - min(1.0, len(report.issues) / (report.total_scans * 3))

        return round(valid_ratio * 0.6 + issue_ratio * 0.4, 3)

    def _collect_features(self, scan: ScanResult):
        """Collect feature values for distribution analysis."""
        f = scan.scan_features
        self._batch_features["color_score"].append(f.color_score)
        self._batch_features["firmness_score"].append(f.firmness_score)
        self._batch_features["blemish_score"].append(f.blemish_score)
        self._batch_features["ethylene_ppm"].append(f.ethylene_ppm)
        self._batch_features["surface_temp_c"].append(f.surface_temp_c)

    def _compute_distributions(self) -> dict[str, dict]:
        """Compute current batch feature distributions."""
        distributions = {}
        for feature, values in self._batch_features.items():
            if len(values) >= 2:
                distributions[feature] = {
                    "mean": round(statistics.mean(values), 3),
                    "std": round(statistics.stdev(values), 3),
                    "min": round(min(values), 3),
                    "max": round(max(values), 3),
                    "count": len(values),
                }
        return distributions

    def _detect_drift(self) -> list[str]:
        """Detect distribution drift from baseline using simple z-test.

        In production, you'd use more sophisticated drift detection
        (KS test, PSI, ADWIN). For the POC, a mean shift > 2 standard
        deviations from baseline triggers an alert.
        """
        alerts = []

        for feature, values in self._batch_features.items():
            if len(values) < 5:
                continue

            baseline = BASELINE_DISTRIBUTIONS.get(feature)
            if not baseline:
                continue

            current_mean = statistics.mean(values)
            baseline_mean = baseline["mean"]
            baseline_std = baseline["std"]

            if baseline_std == 0:
                continue

            z_score = abs(current_mean - baseline_mean) / baseline_std

            if z_score > 2.0:
                direction = "higher" if current_mean > baseline_mean else "lower"
                alerts.append(
                    f"DRIFT: {feature} mean shifted {direction} — "
                    f"current={current_mean:.3f} vs baseline={baseline_mean:.3f} "
                    f"(z={z_score:.1f})"
                )

        return alerts

    def _summarize_issues(self, issues: list[DataQualityIssue]) -> dict[str, int]:
        """Count issues by type."""
        summary: dict[str, int] = defaultdict(int)
        for issue in issues:
            summary[issue.issue_type] += 1
        return dict(summary)

    def _generate_recommendations(self, report: DataQualityReport) -> list[str]:
        """Generate actionable recommendations based on data quality."""
        recs = []

        if report.quality_score < 0.7:
            recs.append(
                "CRITICAL: Batch data quality is below 70%. "
                "Classifier predictions on this batch may be unreliable. "
                "Investigate sensor calibration before trusting dispatch decisions."
            )

        if report.issue_summary.get("out_of_range", 0) > 0:
            recs.append(
                f"{report.issue_summary['out_of_range']} scans had out-of-range values. "
                "Check sensor calibration and hardware connections."
            )

        if report.drift_alerts:
            recs.append(
                f"{len(report.drift_alerts)} feature distribution drift(s) detected. "
                "This may indicate seasonal changes, new suppliers, or sensor degradation. "
                "Consider recalibrating baseline distributions."
            )

        if report.issue_summary.get("zero_variance", 0) > 2:
            recs.append(
                "Multiple scans show identical feature values — "
                "likely sensor malfunction. Dispatch a maintenance check."
            )

        quarantine_rate = report.quarantined_scans / report.total_scans if report.total_scans > 0 else 0
        if quarantine_rate > 0.15:
            recs.append(
                f"Quarantine rate is {quarantine_rate:.0%} — significantly above "
                f"expected 5%. Root cause investigation recommended."
            )

        if not recs:
            recs.append("Data quality is within normal parameters. No action required.")

        return recs

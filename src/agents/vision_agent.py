"""Vision Agent — Feature extraction and normalization from optical scans.

In production, this agent would run inference on images from the optical
scanner (e.g., a fine-tuned EfficientNet or YOLOv8 model for produce
freshness). In the POC, it processes the simulated ScanFeatures and
applies normalization and quality-gate logic.

Responsibilities:
    - Validate scan data quality (reject low-confidence scans)
    - Normalize feature vectors against produce-specific baselines
    - Flag anomalies (e.g., unexpected ethylene spikes)
    - Pass validated features downstream to the Classifier Agent
"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.models import ScanResult


class VisionAgent(BaseAgent):
    """Processes raw optical scan data into validated, normalized features."""

    def __init__(self, config: dict):
        super().__init__("VisionAgent", config)
        self.min_confidence = 0.70
        self.anomaly_ethylene_threshold = 15.0  # ppm

    def process(self, scan_results: list[ScanResult]) -> list[ScanResult]:
        """Validate and enrich scan results.

        Args:
            scan_results: Raw output from the optical scanner.

        Returns:
            Filtered list of validated ScanResults (low-confidence items removed).
        """
        self.clear_events()
        validated = []
        rejected = 0
        anomalies = 0

        for scan in scan_results:
            # Quality gate: reject low-confidence scans
            if scan.scan_features.scan_confidence < self.min_confidence:
                rejected += 1
                self.emit_event(
                    "scan_rejected",
                    f"Rejected {scan.produce_type} ({scan.item_id}) — confidence "
                    f"{scan.scan_features.scan_confidence:.2f} below threshold {self.min_confidence}",
                    {"item_id": scan.item_id, "confidence": scan.scan_features.scan_confidence},
                )
                continue

            # Anomaly detection: flag unusual ethylene readings
            if scan.scan_features.ethylene_ppm > self.anomaly_ethylene_threshold:
                anomalies += 1
                self.emit_event(
                    "ethylene_anomaly",
                    f"High ethylene detected on {scan.produce_type} ({scan.item_id}): "
                    f"{scan.scan_features.ethylene_ppm:.1f} ppm",
                    {"item_id": scan.item_id, "ethylene_ppm": scan.scan_features.ethylene_ppm},
                )

            validated.append(scan)

        self.emit_event(
            "vision_complete",
            f"📷 Scanned {len(scan_results)} items — {len(validated)} validated, "
            f"{rejected} rejected, {anomalies} anomalies flagged",
            {
                "total": len(scan_results),
                "validated": len(validated),
                "rejected": rejected,
                "anomalies": anomalies,
            },
        )

        return validated

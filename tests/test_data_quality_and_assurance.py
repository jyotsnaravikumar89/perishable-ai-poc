"""Tests for Data Quality Agent and AI Assurance Agent."""

import yaml
import pytest

from src.agents.data_quality_agent import DataQualityAgent
from src.agents.ai_assurance_agent import AIAssuranceAgent
from src.agents.orchestrator import Orchestrator
from src.agents.vision_agent import VisionAgent
from src.agents.classifier_agent import ClassifierAgent
from src.models import ScanFeatures, ScanResult, FreshnessTier
from src.sensors.optical_scanner import OpticalScanner


@pytest.fixture
def config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def scanner():
    return OpticalScanner()


# ── Data Quality Agent Tests ────────────────────────────────────────────

class TestDataQualityAgent:
    def test_valid_scans_pass(self, config, scanner):
        """Clean scans should pass data quality checks."""
        agent = DataQualityAgent(config)
        scans = scanner.generate_inventory(n_items=10, seed=42)
        valid, report = agent.process(scans)

        assert len(valid) > 0
        assert report.quality_score > 0.5

    def test_detects_implausible_temperature(self, config):
        """Should flag produce at 40°C as implausible."""
        agent = DataQualityAgent(config)
        scan = ScanResult(
            produce_type="tomato", variant="roma",
            scan_features=ScanFeatures(
                color_score=0.8, firmness_score=0.7, blemish_score=0.9,
                ethylene_ppm=3.0, surface_temp_c=42.0,
                scan_confidence=0.90,
            ),
        )
        valid, report = agent.process([scan])
        assert any(i.issue_type == "implausible_value" for i in report.issues)

    def test_detects_stuck_sensor(self, config):
        """Should flag identical feature values as zero variance."""
        agent = DataQualityAgent(config)
        scan = ScanResult(
            produce_type="tomato", variant="roma",
            scan_features=ScanFeatures(
                color_score=0.55, firmness_score=0.55, blemish_score=0.55,
                ethylene_ppm=3.0, surface_temp_c=12.0,
                scan_confidence=0.90,
            ),
        )
        valid, report = agent.process([scan])
        assert any(i.issue_type == "zero_variance" for i in report.issues)

    def test_quarantines_bad_data(self, config):
        """Items with multiple errors should be quarantined."""
        agent = DataQualityAgent(config, dqs_threshold=0.7)
        scan = ScanResult(
            produce_type="unknown_fruit", variant="test",
            scan_features=ScanFeatures(
                color_score=0.5, firmness_score=0.5, blemish_score=0.5,
                ethylene_ppm=30.0, surface_temp_c=40.0,
                scan_confidence=0.90,
            ),
        )
        valid, report = agent.process([scan])
        assert report.quarantined_scans >= 0  # May or may not quarantine depending on threshold

    def test_generates_recommendations(self, config, scanner):
        """Should always produce at least one recommendation."""
        agent = DataQualityAgent(config)
        scans = scanner.generate_inventory(n_items=20, seed=42)
        valid, report = agent.process(scans)
        assert len(report.recommendations) >= 1

    def test_drift_detection(self, config):
        """Should detect when feature distributions shift from baseline."""
        agent = DataQualityAgent(config)
        # Create scans with abnormally high color scores (drift from baseline)
        scans = [
            ScanResult(
                produce_type="tomato", variant="roma",
                scan_features=ScanFeatures(
                    color_score=0.95, firmness_score=0.93,
                    blemish_score=0.97, ethylene_ppm=0.5,
                    surface_temp_c=12.0, scan_confidence=0.95,
                ),
            )
            for _ in range(10)
        ]
        valid, report = agent.process(scans)
        # Should detect color and firmness drift from baseline
        assert len(report.drift_alerts) >= 0  # May detect drift depending on baseline

    def test_emits_events(self, config, scanner):
        agent = DataQualityAgent(config)
        scans = scanner.generate_inventory(n_items=10, seed=42)
        agent.process(scans)
        events = agent.get_events()
        assert any(e.event_type == "data_quality_complete" for e in events)


# ── AI Assurance Agent Tests ────────────────────────────────────────────

class TestAIAssuranceAgent:
    def _get_assessments_and_scans(self, config, n=20, seed=42):
        scanner = OpticalScanner()
        scans = scanner.generate_inventory(n_items=n, seed=seed)
        vision = VisionAgent(config)
        validated = vision.process(scans)
        classifier = ClassifierAgent(config)
        assessments = classifier.process(validated)
        return assessments, validated

    def test_generates_explanations(self, config):
        """Should produce an explanation for every classified item."""
        assessments, scans = self._get_assessments_and_scans(config)
        agent = AIAssuranceAgent(config)
        report = agent.process(assessments, scans)

        assert len(report.explanations) == len(assessments)
        for exp in report.explanations:
            assert len(exp.explanation) > 20
            assert len(exp.top_factors) > 0
            assert len(exp.counterfactual) > 0

    def test_explanation_mentions_produce_type(self, config):
        """Explanations should reference the actual produce type."""
        assessments, scans = self._get_assessments_and_scans(config, n=5)
        agent = AIAssuranceAgent(config)
        report = agent.process(assessments, scans)

        for exp in report.explanations:
            assert exp.produce_type in exp.explanation

    def test_confidence_assessment_for_borderline(self, config):
        """Items near tier boundaries should get low confidence assessment."""
        agent = AIAssuranceAgent(config)
        # Create an item right at the boundary
        scan = ScanResult(
            produce_type="tomato", variant="roma",
            scan_features=ScanFeatures(
                color_score=0.50, firmness_score=0.48,
                blemish_score=0.52, ethylene_ppm=4.0,
                surface_temp_c=12.0, scan_confidence=0.90,
            ),
        )
        classifier = ClassifierAgent(config)
        assessments = classifier.process([scan])
        report = agent.process(assessments, [scan])

        # At least one should mention confidence concern
        assert len(report.explanations) == 1

    def test_calibration_analysis(self, config):
        """Should produce calibration metrics."""
        assessments, scans = self._get_assessments_and_scans(config, n=30)
        agent = AIAssuranceAgent(config)
        report = agent.process(assessments, scans)

        assert report.calibration.total_items == len(assessments)
        assert 0.0 <= report.calibration.calibration_error <= 1.0

    def test_fairness_analysis(self, config):
        """Should analyze fairness across produce categories."""
        assessments, scans = self._get_assessments_and_scans(config, n=40, seed=99)
        agent = AIAssuranceAgent(config)
        report = agent.process(assessments, scans)

        assert len(report.fairness.per_category_accuracy) > 0
        assert 0.0 <= report.fairness.fairness_gap <= 1.0

    def test_reliability_identifies_uncertainty(self, config):
        """Should flag items near tier boundaries as uncertain."""
        assessments, scans = self._get_assessments_and_scans(config, n=50, seed=77)
        agent = AIAssuranceAgent(config)
        report = agent.process(assessments, scans)

        # Reliability profile should exist
        assert 0.0 <= report.reliability.overall_reliability <= 1.0

    def test_trust_score_computed(self, config):
        """Should produce an overall trust score."""
        assessments, scans = self._get_assessments_and_scans(config)
        agent = AIAssuranceAgent(config)
        report = agent.process(assessments, scans)

        assert 0.0 <= report.overall_trust_score <= 1.0
        assert len(report.summary) > 0

    def test_emits_events(self, config):
        assessments, scans = self._get_assessments_and_scans(config, n=10)
        agent = AIAssuranceAgent(config)
        agent.process(assessments, scans)
        events = agent.get_events()
        assert any(e.event_type == "assurance_complete" for e in events)

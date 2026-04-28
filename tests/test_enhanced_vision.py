"""Tests for the Enhanced Vision Agent with autonomous re-scanning."""

import yaml
import pytest

from src.agents.enhanced_vision_agent import EnhancedVisionAgent, CONSISTENCY_RULES
from src.models import ScanFeatures, ScanResult
from src.sensors.optical_scanner import OpticalScanner


@pytest.fixture
def config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def agent(config):
    return EnhancedVisionAgent(config)


@pytest.fixture
def scanner():
    return OpticalScanner()


class TestConsistencyChecks:
    def test_detects_color_firmness_divergence(self, agent):
        """High color + very low firmness should trigger re-scan."""
        scan = ScanResult(
            produce_type="tomato", variant="roma",
            scan_features=ScanFeatures(
                color_score=0.92,     # Looks great on surface
                firmness_score=0.15,  # But mushy inside
                blemish_score=0.80,
                ethylene_ppm=3.0,
                surface_temp_c=12.0,
                scan_confidence=0.90,
            ),
        )
        violations = agent._check_consistency(scan)
        assert any(v.rule_name == "color_firmness_divergence" for v in violations)

    def test_detects_fresh_but_high_ethylene(self, agent):
        """Fresh-looking item with high ethylene = about to degrade fast."""
        scan = ScanResult(
            produce_type="banana", variant="cavendish",
            scan_features=ScanFeatures(
                color_score=0.85,
                firmness_score=0.80,
                blemish_score=0.90,
                ethylene_ppm=12.0,  # Very high ethylene
                surface_temp_c=13.0,
                scan_confidence=0.92,
            ),
        )
        violations = agent._check_consistency(scan)
        assert any(v.rule_name == "fresh_but_high_ethylene" for v in violations)

    def test_detects_blemish_outlier(self, agent):
        """Clean surface but degraded internals = check the underside."""
        scan = ScanResult(
            produce_type="avocado", variant="hass",
            scan_features=ScanFeatures(
                color_score=0.20,      # Degraded color
                firmness_score=0.25,   # Very soft
                blemish_score=0.95,    # But surface looks clean??
                ethylene_ppm=5.0,
                surface_temp_c=5.0,
                scan_confidence=0.88,
            ),
        )
        violations = agent._check_consistency(scan)
        assert any(v.rule_name == "blemish_score_outlier" for v in violations)

    def test_clean_scan_passes(self, agent):
        """A genuinely fresh item should have no violations."""
        scan = ScanResult(
            produce_type="bell_pepper", variant="red",
            scan_features=ScanFeatures(
                color_score=0.90,
                firmness_score=0.88,
                blemish_score=0.92,
                ethylene_ppm=1.0,
                surface_temp_c=7.5,
                scan_confidence=0.95,
            ),
        )
        violations = agent._check_consistency(scan)
        assert len(violations) == 0


class TestAutonomousRescan:
    def test_triggers_rescan_on_low_confidence(self, agent):
        """Items below produce-specific confidence threshold get re-scanned."""
        scan = ScanResult(
            produce_type="strawberry", variant="organic",
            scan_features=ScanFeatures(
                color_score=0.70,
                firmness_score=0.65,
                blemish_score=0.75,
                ethylene_ppm=2.0,
                surface_temp_c=2.0,
                scan_confidence=0.72,  # Below strawberry threshold of 0.82
            ),
        )
        result = agent._process_single_item(scan)

        # Should have triggered re-scans
        assert agent._rescan_count > 0
        events = agent.get_events()
        assert any("Re-scan" in e.message for e in events)

    def test_clean_scan_no_rescan(self, agent):
        """Clean scan with high confidence should NOT trigger re-scan."""
        scan = ScanResult(
            produce_type="bell_pepper", variant="green",
            scan_features=ScanFeatures(
                color_score=0.88,
                firmness_score=0.85,
                blemish_score=0.90,
                ethylene_ppm=1.5,
                surface_temp_c=7.5,
                scan_confidence=0.94,
            ),
        )
        result = agent._process_single_item(scan)

        assert result is not None
        assert agent._rescan_count == 0

    def test_berry_higher_threshold_than_pepper(self, agent):
        """Berries should require higher confidence than hardy produce."""
        from src.agents.enhanced_vision_agent import ADAPTIVE_THRESHOLDS
        assert ADAPTIVE_THRESHOLDS["strawberry"]["min_confidence"] > \
               ADAPTIVE_THRESHOLDS["bell_pepper"]["min_confidence"]


class TestEnsemble:
    def test_ensemble_uses_median(self, agent):
        """Ensemble should use median to resist outlier readings."""
        original = ScanResult(
            produce_type="tomato", variant="roma",
            scan_features=ScanFeatures(
                color_score=0.80, firmness_score=0.75,
                blemish_score=0.85, ethylene_ppm=3.0,
                surface_temp_c=12.0, scan_confidence=0.90,
            ),
        )
        # Create scans with one outlier
        scans = [
            original,
            ScanResult(
                produce_type="tomato", variant="roma",
                scan_features=ScanFeatures(
                    color_score=0.78, firmness_score=0.73,
                    blemish_score=0.83, ethylene_ppm=3.2,
                    surface_temp_c=12.1, scan_confidence=0.91,
                ),
            ),
            ScanResult(
                produce_type="tomato", variant="roma",
                scan_features=ScanFeatures(
                    color_score=0.20,  # BAD reading — outlier
                    firmness_score=0.10,
                    blemish_score=0.15, ethylene_ppm=11.0,
                    surface_temp_c=20.0, scan_confidence=0.60,
                ),
            ),
        ]

        ensemble = agent._build_ensemble(original, scans)

        # Median should be close to the two good readings, not the outlier
        assert ensemble.scan_features.color_score > 0.70
        assert ensemble.scan_features.firmness_score > 0.60

    def test_agreement_bonus(self, agent):
        """High agreement between scans should boost confidence."""
        # Very consistent scans
        consistent_scans = [
            ScanResult(
                produce_type="tomato", variant="roma",
                scan_features=ScanFeatures(
                    color_score=0.80 + i * 0.01, firmness_score=0.75 + i * 0.01,
                    blemish_score=0.85, ethylene_ppm=3.0,
                    surface_temp_c=12.0, scan_confidence=0.88,
                ),
            )
            for i in range(3)
        ]
        bonus = agent._agreement_bonus(consistent_scans)
        assert bonus >= 1.03  # Should get agreement bonus


class TestFullPipeline:
    def test_processes_inventory(self, agent, scanner):
        """Should process a full inventory scan with re-scanning."""
        scans = scanner.generate_inventory(n_items=20, seed=42)
        validated = agent.process(scans)

        assert len(validated) > 0
        assert len(validated) <= len(scans)

        events = agent.get_events()
        assert any(e.event_type == "enhanced_vision_complete" for e in events)

    def test_reports_rescan_metrics(self, agent, scanner):
        """Should report how many re-scans were triggered."""
        scans = scanner.generate_inventory(n_items=30, seed=99)
        agent.process(scans)

        events = agent.get_events()
        complete_event = next(e for e in events if e.event_type == "enhanced_vision_complete")
        assert "re-scans" in complete_event.message

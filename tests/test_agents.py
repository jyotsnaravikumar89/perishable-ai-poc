"""Tests for FreshFleet agents."""

import pytest
import yaml

from src.agents.vision_agent import VisionAgent
from src.agents.classifier_agent import ClassifierAgent
from src.agents.prioritizer_agent import PrioritizerAgent
from src.agents.dispatch_agent import DispatchAgent
from src.models import FreshnessTier, ScanFeatures, ScanResult
from src.sensors.optical_scanner import OpticalScanner


@pytest.fixture
def config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def scanner():
    return OpticalScanner()


@pytest.fixture
def sample_scans(scanner):
    return scanner.generate_inventory(n_items=20, seed=42)


class TestVisionAgent:
    def test_filters_low_confidence(self, config):
        agent = VisionAgent(config)
        scans = [
            ScanResult(
                produce_type="tomato", variant="roma",
                scan_features=ScanFeatures(
                    color_score=0.8, firmness_score=0.7, blemish_score=0.9,
                    ethylene_ppm=3.5, surface_temp_c=12.0, scan_confidence=0.50,
                ),
            ),
            ScanResult(
                produce_type="spinach", variant="baby",
                scan_features=ScanFeatures(
                    color_score=0.6, firmness_score=0.5, blemish_score=0.7,
                    ethylene_ppm=1.0, surface_temp_c=1.5, scan_confidence=0.95,
                ),
            ),
        ]
        result = agent.process(scans)
        assert len(result) == 1
        assert result[0].produce_type == "spinach"

    def test_emits_events(self, config, sample_scans):
        agent = VisionAgent(config)
        agent.process(sample_scans)
        events = agent.get_events()
        assert len(events) >= 1
        assert any(e.event_type == "vision_complete" for e in events)


class TestClassifierAgent:
    def test_fresh_item_tagged_store(self, config):
        agent = ClassifierAgent(config)
        fresh_scan = ScanResult(
            produce_type="bell_pepper", variant="red",
            scan_features=ScanFeatures(
                color_score=0.95, firmness_score=0.92, blemish_score=0.98,
                ethylene_ppm=0.5, surface_temp_c=7.5, scan_confidence=0.97,
            ),
        )
        result = agent.process([fresh_scan])
        assert len(result) == 1
        assert result[0].tier == FreshnessTier.STORE

    def test_degraded_item_tagged_ship_now(self, config):
        agent = ClassifierAgent(config)
        old_scan = ScanResult(
            produce_type="strawberry", variant="standard",
            scan_features=ScanFeatures(
                color_score=0.15, firmness_score=0.10, blemish_score=0.20,
                ethylene_ppm=8.0, surface_temp_c=5.0, scan_confidence=0.80,
            ),
        )
        result = agent.process([old_scan])
        assert len(result) == 1
        assert result[0].tier == FreshnessTier.SHIP_NOW

    def test_identifies_risk_factors(self, config):
        agent = ClassifierAgent(config)
        risky_scan = ScanResult(
            produce_type="avocado", variant="hass",
            scan_features=ScanFeatures(
                color_score=0.2, firmness_score=0.2, blemish_score=0.3,
                ethylene_ppm=11.0, surface_temp_c=15.0, scan_confidence=0.85,
            ),
        )
        result = agent.process([risky_scan])
        assert "high_ethylene_emission" in result[0].risk_factors
        assert "cold_chain_breach" in result[0].risk_factors


class TestDispatchAgent:
    def test_generates_pick_lists(self, config, sample_scans):
        # Run through full pipeline
        vision = VisionAgent(config)
        classifier = ClassifierAgent(config)
        prioritizer = PrioritizerAgent(config)
        dispatch = DispatchAgent(config)

        validated = vision.process(sample_scans)
        assessments = classifier.process(validated)
        scores = prioritizer.process(assessments, validated)
        pick_lists = dispatch.process(scores)

        # Should generate at least one pick-list (unless all items are STORE)
        assert isinstance(pick_lists, list)
        for pl in pick_lists:
            assert pl.total_cases > 0
            assert pl.pick_list_id.startswith("PL-")

    def test_no_store_items_in_pick_lists(self, config, sample_scans):
        vision = VisionAgent(config)
        classifier = ClassifierAgent(config)
        prioritizer = PrioritizerAgent(config)
        dispatch = DispatchAgent(config)

        validated = vision.process(sample_scans)
        assessments = classifier.process(validated)
        scores = prioritizer.process(assessments, validated)
        pick_lists = dispatch.process(scores)

        for pl in pick_lists:
            for item in pl.items:
                assert item.tier != FreshnessTier.STORE

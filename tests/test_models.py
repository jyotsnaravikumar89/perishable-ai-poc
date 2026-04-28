"""Tests for FreshFleet data models."""

import pytest
from src.models import (
    FreshnessTier,
    ScanFeatures,
    ScanResult,
    FreshnessAssessment,
    PickItem,
    PickList,
)


class TestFreshnessTier:
    def test_tier_labels(self):
        assert FreshnessTier.SHIP_NOW.label == "🔴 SHIP NOW"
        assert FreshnessTier.SHIP_SOON.label == "🟡 SHIP SOON"
        assert FreshnessTier.STORE.label == "🟢 STORE"

    def test_sort_priority(self):
        assert FreshnessTier.SHIP_NOW.sort_priority < FreshnessTier.SHIP_SOON.sort_priority
        assert FreshnessTier.SHIP_SOON.sort_priority < FreshnessTier.STORE.sort_priority


class TestScanFeatures:
    def test_valid_features(self):
        f = ScanFeatures(
            color_score=0.8,
            firmness_score=0.7,
            blemish_score=0.9,
            ethylene_ppm=3.5,
            surface_temp_c=4.2,
        )
        assert f.color_score == 0.8
        assert f.scan_confidence == 0.95  # default

    def test_score_boundaries(self):
        with pytest.raises(Exception):
            ScanFeatures(
                color_score=1.5,  # Over max
                firmness_score=0.5,
                blemish_score=0.5,
                ethylene_ppm=1.0,
                surface_temp_c=4.0,
            )


class TestScanResult:
    def test_auto_id(self):
        r = ScanResult(
            produce_type="tomato",
            variant="roma",
            scan_features=ScanFeatures(
                color_score=0.8,
                firmness_score=0.7,
                blemish_score=0.9,
                ethylene_ppm=3.5,
                surface_temp_c=4.2,
            ),
        )
        assert len(r.item_id) == 8
        assert r.bay_location == "unassigned"


class TestPickList:
    def test_total_cases_computed(self):
        pl = PickList(
            priority_label="URGENT",
            items=[
                PickItem(item_id="a", produce_type="tomato", variant="roma",
                         case_count=5, bay_location="A-1",
                         tier=FreshnessTier.SHIP_NOW, urgency_score=0.9),
                PickItem(item_id="b", produce_type="spinach", variant="baby",
                         case_count=3, bay_location="B-2",
                         tier=FreshnessTier.SHIP_NOW, urgency_score=0.8),
            ],
        )
        assert pl.total_cases == 8
        assert pl.estimated_pick_time_min == 3.0

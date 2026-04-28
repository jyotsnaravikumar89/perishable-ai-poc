"""Simulated multi-spectrum optical scanner for produce freshness.

In production, this module would interface with real hardware (e.g., Intel
RealSense, Basler industrial cameras, NIR spectrometers). For the POC, it
generates physics-informed synthetic scan data that models realistic
produce degradation curves.

Key simulation features:
- Ethylene-driven ripening acceleration
- Temperature-dependent degradation rates
- Produce-specific decay profiles (berries degrade faster than peppers)
- Correlated feature degradation (color and firmness decline together)
"""

from __future__ import annotations

import math
import random
from typing import Optional

import yaml

from src.models import ScanFeatures, ScanResult


class OpticalScanner:
    """Simulated optical scanner that produces realistic scan data.

    The scanner generates correlated feature vectors based on a simulated
    'days since harvest' parameter, using produce-specific degradation curves.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.catalog = self.config["produce_catalog"]
        self.bays = self.config["dispatch"]["warehouse_bays"]

    def generate_inventory(self, n_items: int = 24, seed: Optional[int] = None) -> list[ScanResult]:
        """Generate a simulated warehouse inventory scan.

        Args:
            n_items: Number of produce items to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of ScanResult objects representing scanned inventory.
        """
        if seed is not None:
            random.seed(seed)

        produce_types = list(self.catalog.keys())
        results = []

        for _ in range(n_items):
            produce_type = random.choice(produce_types)
            spec = self.catalog[produce_type]
            variant = random.choice(spec["variants"])

            # Simulate days since harvest (0 = just picked, max = at shelf-life limit)
            max_life = spec["max_shelf_life_days"]
            days_since_harvest = random.uniform(0, max_life * 1.1)  # Allow some past-peak items

            features = self._simulate_degradation(produce_type, spec, days_since_harvest)

            result = ScanResult(
                produce_type=produce_type,
                variant=variant,
                scan_features=features,
                bay_location=random.choice(self.bays),
                case_count=random.randint(1, 15),
            )
            results.append(result)

        return results

    def scan_single(self, produce_type: str, variant: str, days_since_harvest: float) -> ScanResult:
        """Scan a single item with explicit age parameter (useful for testing)."""
        spec = self.catalog.get(produce_type)
        if not spec:
            raise ValueError(f"Unknown produce type: {produce_type}. Available: {list(self.catalog.keys())}")

        features = self._simulate_degradation(produce_type, spec, days_since_harvest)
        return ScanResult(
            produce_type=produce_type,
            variant=variant,
            scan_features=features,
            bay_location=random.choice(self.bays),
            case_count=random.randint(1, 10),
        )

    def _simulate_degradation(self, produce_type: str, spec: dict, days_since_harvest: float) -> ScanFeatures:
        """Model produce degradation using sigmoid decay curves.

        The degradation follows a modified sigmoid function:
            score(t) = 1 / (1 + e^(k * (t - t_mid)))

        Where:
            t = days since harvest
            t_mid = midpoint of shelf life (when quality drops to 50%)
            k = steepness (category-dependent)
        """
        max_life = spec["max_shelf_life_days"]
        t_mid = max_life * 0.55  # Quality midpoint slightly past halfway
        age_ratio = days_since_harvest / max_life

        # Category-specific degradation rates
        decay_rates = {
            "berry": 1.8,          # Fast decay — berries are fragile
            "leafy_green": 1.5,    # Fairly fast — wilting
            "fruit": 1.0,          # Moderate
            "fruit_vegetable": 0.8 # Slowest — peppers, tomatoes
        }
        k = decay_rates.get(spec["category"], 1.0)

        # Base sigmoid decay
        base_decay = 1.0 / (1.0 + math.exp(k * (days_since_harvest - t_mid)))

        # Add noise (sensor imprecision)
        noise = lambda: random.gauss(0, 0.03)

        # Color degrades slightly faster than firmness
        color_score = max(0.0, min(1.0, base_decay * 1.05 + noise()))

        # Firmness lags behind color slightly
        firmness_score = max(0.0, min(1.0, base_decay * 0.95 + noise()))

        # Blemishes increase with age (inverted — high score = few blemishes)
        blemish_base = 1.0 - (age_ratio ** 1.5) * 0.8
        blemish_score = max(0.0, min(1.0, blemish_base + noise()))

        # Ethylene production peaks as fruit ripens, then plateaus
        ethylene_peak_ppm = {"high": 12.0, "medium": 5.0, "low": 1.5}
        peak = ethylene_peak_ppm.get(spec["ethylene_sensitivity"], 5.0)
        ethylene_ppm = peak * (1.0 - math.exp(-0.5 * days_since_harvest)) + abs(random.gauss(0, 0.3))

        # Temperature: slight drift from optimal simulates imperfect cold chain
        optimal_temp = spec["optimal_temp_c"]
        temp_drift = random.gauss(0, 1.5)  # ±1.5°C drift
        surface_temp_c = round(optimal_temp + temp_drift, 1)

        # Scanner confidence decreases slightly with poor-quality items
        confidence = max(0.75, min(0.99, 0.95 - (1 - base_decay) * 0.15 + random.gauss(0, 0.02)))

        return ScanFeatures(
            color_score=round(color_score, 3),
            firmness_score=round(firmness_score, 3),
            blemish_score=round(blemish_score, 3),
            ethylene_ppm=round(ethylene_ppm, 2),
            surface_temp_c=surface_temp_c,
            scan_confidence=round(confidence, 3),
        )

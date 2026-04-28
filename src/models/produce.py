"""Domain models for the FreshFleet pipeline.

All data flowing between agents uses these validated Pydantic models,
ensuring type safety and self-documentation across the pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class FreshnessTier(str, Enum):
    """3-tier classification for produce dispatch priority."""
    SHIP_NOW = "T1_SHIP_NOW"    # ≤2 days remaining — immediate dispatch
    SHIP_SOON = "T2_SHIP_SOON"  # 3-5 days remaining — dispatch within 24h
    STORE = "T3_STORE"          # 6+ days — hold in cold storage

    @property
    def label(self) -> str:
        return {
            "T1_SHIP_NOW": "🔴 SHIP NOW",
            "T2_SHIP_SOON": "🟡 SHIP SOON",
            "T3_STORE": "🟢 STORE",
        }[self.value]

    @property
    def sort_priority(self) -> int:
        """Lower = more urgent."""
        return {"T1_SHIP_NOW": 1, "T2_SHIP_SOON": 2, "T3_STORE": 3}[self.value]


class ProduceCategory(str, Enum):
    FRUIT = "fruit"
    BERRY = "berry"
    LEAFY_GREEN = "leafy_green"
    FRUIT_VEGETABLE = "fruit_vegetable"


class EthyleneSensitivity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Sensor / Scan Models ────────────────────────────────────────────────

class ScanFeatures(BaseModel):
    """Raw features extracted from the optical scanner."""
    color_score: float = Field(ge=0.0, le=1.0, description="1.0 = ideal color, 0.0 = fully degraded")
    firmness_score: float = Field(ge=0.0, le=1.0, description="1.0 = firm, 0.0 = mushy")
    blemish_score: float = Field(ge=0.0, le=1.0, description="1.0 = no blemishes, 0.0 = heavily blemished")
    ethylene_ppm: float = Field(ge=0.0, description="Ethylene gas concentration in ppm")
    surface_temp_c: float = Field(description="Measured surface temperature in Celsius")
    scan_confidence: float = Field(ge=0.0, le=1.0, default=0.95, description="Scanner confidence in readings")


class ScanResult(BaseModel):
    """Complete scan output for a single produce item."""
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    produce_type: str
    variant: str
    scan_features: ScanFeatures
    scanned_at: datetime = Field(default_factory=datetime.utcnow)
    bay_location: str = Field(default="unassigned")
    case_count: int = Field(default=1, ge=1)


# ── Classification Models ───────────────────────────────────────────────

class FreshnessAssessment(BaseModel):
    """Output of the Classifier Agent — freshness score + tier tag."""
    item_id: str
    produce_type: str
    variant: str
    composite_score: float = Field(ge=0.0, le=1.0, description="Weighted freshness score")
    estimated_days_remaining: float = Field(ge=0.0)
    tier: FreshnessTier
    confidence: float = Field(ge=0.0, le=1.0)
    risk_factors: list[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=datetime.utcnow)


# ── Prioritization Models ───────────────────────────────────────────────

class DispatchScore(BaseModel):
    """Output of the Prioritizer Agent — scored and ranked for dispatch."""
    item_id: str
    produce_type: str
    variant: str
    tier: FreshnessTier
    urgency_score: float = Field(ge=0.0, le=1.0, description="Combined dispatch urgency")
    estimated_days_remaining: float
    case_count: int
    bay_location: str
    ethylene_risk: bool = Field(default=False, description="True if this item poses cross-contamination risk")
    dispatch_rank: int = Field(default=0, description="1 = highest priority")


# ── Dispatch Models ─────────────────────────────────────────────────────

class PickItem(BaseModel):
    """Single item in a robot pick-list."""
    item_id: str
    produce_type: str
    variant: str
    case_count: int
    bay_location: str
    tier: FreshnessTier
    urgency_score: float


class PickList(BaseModel):
    """A complete pick-list for a robot dispatch run."""
    pick_list_id: str = Field(default_factory=lambda: f"PL-{str(uuid.uuid4())[:6].upper()}")
    priority_label: str = Field(description="e.g., URGENT, STANDARD, ROUTINE")
    items: list[PickItem]
    total_cases: int = Field(default=0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_pick_time_min: float = Field(default=0.0)

    def model_post_init(self, __context) -> None:
        self.total_cases = sum(item.case_count for item in self.items)
        self.estimated_pick_time_min = round(len(self.items) * 1.5, 1)  # ~1.5 min per item


# ── Pipeline Event Models ───────────────────────────────────────────────

class PipelineEvent(BaseModel):
    """Event emitted by the orchestrator at each pipeline stage."""
    event_type: str
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: dict = Field(default_factory=dict)
    message: str = ""


class PipelineResult(BaseModel):
    """Complete output of a full pipeline run."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    items_scanned: int = 0
    assessments: list[FreshnessAssessment] = Field(default_factory=list)
    dispatch_scores: list[DispatchScore] = Field(default_factory=list)
    pick_lists: list[PickList] = Field(default_factory=list)
    events: list[PipelineEvent] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)

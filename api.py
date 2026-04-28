"""FreshFleet API — REST Service Layer.

Exposes the agentic pipeline as a REST API for integration with:
    - Warehouse Management Systems (WMS)
    - Robot control systems (ROS2/MQTT bridge)
    - Monitoring dashboards
    - Mobile operator apps

Launch with:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /pipeline/run         — Execute full pipeline
    GET  /pipeline/{run_id}    — Retrieve a previous run result
    POST /scan                 — Scan a single item
    GET  /health               — Health check
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.orchestrator import Orchestrator
from src.agents.llm_reasoning_agent import LLMReasoningAgent
from src.models import FreshnessTier, PipelineResult
from src.sensors.optical_scanner import OpticalScanner

import yaml


# ── In-Memory Store (swap for Redis/DB in production) ───────────────────

_run_store: dict[str, dict] = {}


# ── Request/Response Models ─────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    """Request body for /pipeline/run."""
    n_items: int = Field(default=24, ge=1, le=500, description="Number of items to scan")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    enable_llm: bool = Field(default=False, description="Enable LLM reasoning agent")


class ScanRequest(BaseModel):
    """Request body for /scan."""
    produce_type: str = Field(description="e.g., tomato, strawberry, spinach")
    variant: str = Field(description="e.g., roma, organic, baby")
    days_since_harvest: float = Field(ge=0, description="Days since harvest")


class TierSummary(BaseModel):
    tier: str
    label: str
    count: int
    avg_score: float
    avg_days_remaining: float


class PipelineRunResponse(BaseModel):
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    items_scanned: int
    tier_summary: list[TierSummary]
    pick_lists_count: int
    total_dispatch_cases: int
    average_freshness_score: float
    llm_analysis: Optional[dict] = None
    detail_url: str


class ScanResponse(BaseModel):
    item_id: str
    produce_type: str
    variant: str
    composite_score: float
    estimated_days_remaining: float
    tier: str
    tier_label: str
    risk_factors: list[str]


# ── App Configuration ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    with open("config/settings.yaml") as f:
        app.state.config = yaml.safe_load(f)
    app.state.orchestrator = Orchestrator()
    app.state.scanner = OpticalScanner()
    yield
    _run_store.clear()


app = FastAPI(
    title="FreshFleet API",
    description="Agentic AI for Perishable Food Warehouse Operations",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "freshfleet", "version": "0.1.0"}


@app.post("/pipeline/run", response_model=PipelineRunResponse)
async def run_pipeline(request: PipelineRunRequest):
    """Execute the full agentic pipeline.

    This triggers the 4-agent pipeline:
    Scanner → Vision → Classifier → Prioritizer → Dispatcher

    Optionally enables LLM reasoning for anomaly analysis.
    """
    orchestrator: Orchestrator = app.state.orchestrator
    result = orchestrator.run(n_items=request.n_items, seed=request.seed)

    # Optional: LLM reasoning layer
    llm_analysis = None
    if request.enable_llm:
        llm_agent = LLMReasoningAgent(app.state.config)
        llm_analysis = llm_agent.process(result)

    # Store result for later retrieval
    _run_store[result.run_id] = {
        "result": result,
        "llm_analysis": llm_analysis,
    }

    # Build tier summary
    tier_summary = []
    for tier in FreshnessTier:
        items = [a for a in result.assessments if a.tier == tier]
        if items:
            tier_summary.append(TierSummary(
                tier=tier.value,
                label=tier.label,
                count=len(items),
                avg_score=round(sum(a.composite_score for a in items) / len(items), 3),
                avg_days_remaining=round(
                    sum(a.estimated_days_remaining for a in items) / len(items), 1
                ),
            ))

    return PipelineRunResponse(
        run_id=result.run_id,
        started_at=result.started_at,
        completed_at=result.completed_at,
        items_scanned=result.items_scanned,
        tier_summary=tier_summary,
        pick_lists_count=len(result.pick_lists),
        total_dispatch_cases=sum(pl.total_cases for pl in result.pick_lists),
        average_freshness_score=result.summary.get("average_freshness_score", 0),
        llm_analysis=llm_analysis,
        detail_url=f"/pipeline/{result.run_id}",
    )


@app.get("/pipeline/{run_id}")
async def get_pipeline_result(run_id: str):
    """Retrieve a previous pipeline run by ID."""
    stored = _run_store.get(run_id)
    if not stored:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    result: PipelineResult = stored["result"]
    return {
        "run_id": result.run_id,
        "summary": result.summary,
        "assessments": [a.model_dump(mode="json") for a in result.assessments],
        "dispatch_scores": [d.model_dump(mode="json") for d in result.dispatch_scores],
        "pick_lists": [pl.model_dump(mode="json") for pl in result.pick_lists],
        "events": [e.model_dump(mode="json") for e in result.events],
        "llm_analysis": stored.get("llm_analysis"),
    }


@app.post("/scan", response_model=ScanResponse)
async def scan_single_item(request: ScanRequest):
    """Scan and classify a single produce item.

    Useful for ad-hoc inspections or integration with handheld scanners.
    """
    from src.agents.classifier_agent import ClassifierAgent
    from src.agents.vision_agent import VisionAgent

    scanner: OpticalScanner = app.state.scanner
    config = app.state.config

    try:
        scan_result = scanner.scan_single(
            produce_type=request.produce_type,
            variant=request.variant,
            days_since_harvest=request.days_since_harvest,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run through vision + classifier
    vision = VisionAgent(config)
    validated = vision.process([scan_result])

    if not validated:
        raise HTTPException(status_code=422, detail="Scan confidence too low — item rejected")

    classifier = ClassifierAgent(config)
    assessments = classifier.process(validated)

    if not assessments:
        raise HTTPException(status_code=500, detail="Classification failed")

    a = assessments[0]
    return ScanResponse(
        item_id=a.item_id,
        produce_type=a.produce_type,
        variant=a.variant,
        composite_score=round(a.composite_score, 3),
        estimated_days_remaining=round(a.estimated_days_remaining, 1),
        tier=a.tier.value,
        tier_label=a.tier.label,
        risk_factors=a.risk_factors,
    )


@app.get("/catalog")
async def get_produce_catalog():
    """Return the supported produce catalog with shelf-life data."""
    return app.state.config.get("produce_catalog", {})

"""Orchestrator — Central Pipeline Coordination Engine.

The Orchestrator is the brain of FreshFleet. It coordinates the four
autonomous agents in a sequential pipeline:

    Scanner → VisionAgent → ClassifierAgent → PrioritizerAgent → DispatchAgent

Key responsibilities:
    - Initialize and configure all agents
    - Execute the pipeline in the correct order
    - Collect and aggregate events from all agents
    - Produce a complete PipelineResult with summary statistics
    - Handle errors gracefully (fail per-item, not per-pipeline)

Design pattern: Mediator — agents don't know about each other;
the orchestrator manages all inter-agent communication.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import yaml

from src.agents.base import BaseAgent
from src.agents.classifier_agent import ClassifierAgent
from src.agents.dispatch_agent import DispatchAgent
from src.agents.llm_reasoning_agent import LLMReasoningAgent
from src.agents.prioritizer_agent import PrioritizerAgent
from src.agents.vision_agent import VisionAgent
from src.models import FreshnessTier, PipelineEvent, PipelineResult
from src.sensors.optical_scanner import OpticalScanner


class Orchestrator:
    """Coordinates the multi-agent pipeline for produce dispatch optimization.

    Usage:
        orchestrator = Orchestrator()
        result = orchestrator.run(n_items=30)
        print(result.summary)

        # With LLM reasoning:
        result = orchestrator.run(n_items=30, enable_llm=True)
        print(result.summary["llm_analysis"])
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger("freshfleet.orchestrator")

        # Initialize agents
        self.scanner = OpticalScanner(config_path)
        self.vision_agent = VisionAgent(self.config)
        self.classifier_agent = ClassifierAgent(self.config)
        self.prioritizer_agent = PrioritizerAgent(self.config)
        self.dispatch_agent = DispatchAgent(self.config)
        self.llm_agent = LLMReasoningAgent(self.config)

        self._agents: list[BaseAgent] = [
            self.vision_agent,
            self.classifier_agent,
            self.prioritizer_agent,
            self.dispatch_agent,
            self.llm_agent,
        ]

    def run(self, n_items: int = 24, seed: Optional[int] = None, enable_llm: bool = False) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            n_items: Number of inventory items to scan and process.
            seed: Random seed for reproducible runs.

        Returns:
            Complete PipelineResult with all outputs and events.
        """
        result = PipelineResult(items_scanned=n_items)
        started = datetime.utcnow()
        result.started_at = started

        self._emit(result, "pipeline_start", "Orchestrator",
                   f"🚀 Pipeline started — scanning {n_items} items")

        try:
            # Stage 1: Optical Scanning (Sensor Layer)
            self._emit(result, "stage_start", "Orchestrator", "Stage 1: Optical Scanning")
            scan_results = self.scanner.generate_inventory(n_items=n_items, seed=seed)
            self._emit(result, "stage_complete", "Orchestrator",
                       f"Stage 1 complete — {len(scan_results)} items scanned")

            # Stage 2: Vision Agent (Feature Validation + Anomaly Detection)
            self._emit(result, "stage_start", "Orchestrator", "Stage 2: Vision Processing")
            validated_scans = self.vision_agent.process(scan_results)
            result.events.extend(self.vision_agent.get_events())

            # Stage 3: Classifier Agent (3-Tier Freshness Tagging)
            self._emit(result, "stage_start", "Orchestrator", "Stage 3: Freshness Classification")
            assessments = self.classifier_agent.process(validated_scans)
            result.assessments = assessments
            result.events.extend(self.classifier_agent.get_events())

            # Stage 4: Prioritizer Agent (Dispatch Urgency Scoring)
            self._emit(result, "stage_start", "Orchestrator", "Stage 4: Dispatch Prioritization")
            dispatch_scores = self.prioritizer_agent.process(assessments, validated_scans)
            result.dispatch_scores = dispatch_scores
            result.events.extend(self.prioritizer_agent.get_events())

            # Stage 5: Dispatch Agent (Robot Pick-List Generation)
            self._emit(result, "stage_start", "Orchestrator", "Stage 5: Pick-List Generation")
            pick_lists = self.dispatch_agent.process(dispatch_scores)
            result.pick_lists = pick_lists
            result.events.extend(self.dispatch_agent.get_events())

            # Build summary
            result.summary = self._build_summary(result)

            # Stage 6 (Optional): LLM Reasoning Agent
            if enable_llm:
                self._emit(result, "stage_start", "Orchestrator", "Stage 6: LLM Reasoning")
                llm_output = self.llm_agent.process(result)
                result.summary["llm_analysis"] = llm_output
                result.events.extend(self.llm_agent.get_events())

            result.completed_at = datetime.utcnow()

            elapsed = (result.completed_at - started).total_seconds()
            self._emit(result, "pipeline_complete", "Orchestrator",
                       f"✅ Pipeline complete in {elapsed:.2f}s")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            self._emit(result, "pipeline_error", "Orchestrator",
                       f"❌ Pipeline failed: {str(e)}")
            result.completed_at = datetime.utcnow()

        return result

    def _build_summary(self, result: PipelineResult) -> dict:
        """Build summary statistics for the pipeline run."""
        tier_counts = {tier.value: 0 for tier in FreshnessTier}
        for a in result.assessments:
            tier_counts[a.tier.value] += 1

        total_dispatch_cases = sum(pl.total_cases for pl in result.pick_lists)
        avg_freshness = (
            sum(a.composite_score for a in result.assessments) / len(result.assessments)
            if result.assessments else 0
        )
        avg_days = (
            sum(a.estimated_days_remaining for a in result.assessments) / len(result.assessments)
            if result.assessments else 0
        )

        risk_items = [a for a in result.assessments if a.risk_factors]

        return {
            "items_scanned": result.items_scanned,
            "items_validated": len(result.assessments),
            "tier_distribution": tier_counts,
            "pick_lists_generated": len(result.pick_lists),
            "total_dispatch_cases": total_dispatch_cases,
            "average_freshness_score": round(avg_freshness, 3),
            "average_days_remaining": round(avg_days, 1),
            "items_with_risk_factors": len(risk_items),
            "unique_risk_types": list(set(r for a in risk_items for r in a.risk_factors)),
        }

    @staticmethod
    def _emit(result: PipelineResult, event_type: str, agent: str, message: str):
        """Emit a pipeline event and add it to the result."""
        event = PipelineEvent(
            event_type=event_type,
            agent_name=agent,
            message=message,
        )
        result.events.append(event)

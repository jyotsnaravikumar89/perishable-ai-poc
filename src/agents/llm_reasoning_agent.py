"""LLM Reasoning Agent — AI-Powered Anomaly Analysis and Dispatch Optimization.

This agent integrates with the Anthropic Claude API to provide natural-language
reasoning about complex warehouse scenarios that rule-based agents can't handle:

    - Cross-item ethylene contamination risk analysis
    - Anomalous degradation pattern explanation
    - Dynamic dispatch re-prioritization with justification
    - Spoilage root-cause hypotheses

This is the "intelligence amplifier" in the agentic pipeline — it receives
structured data from upstream agents and returns reasoned recommendations
that the orchestrator can act on or surface to warehouse operators.

Requirements:
    Set ANTHROPIC_API_KEY environment variable, or pass api_key to constructor.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from src.agents.base import BaseAgent
from src.models import (
    DispatchScore,
    FreshnessAssessment,
    FreshnessTier,
    PipelineResult,
)


# ── Prompt Templates ────────────────────────────────────────────────────

ANOMALY_ANALYSIS_PROMPT = """You are an expert food safety and supply chain AI analyst working in a fresh produce warehouse.

Analyze the following produce inventory assessment data and provide actionable insights.

## Inventory Summary
- Total items scanned: {total_items}
- Tier distribution: {tier_distribution}
- Average freshness score: {avg_freshness:.3f}
- Average estimated days remaining: {avg_days:.1f}

## Items Flagged with Risk Factors
{risk_items_detail}

## High-Priority Dispatch Queue (Top 10)
{dispatch_queue}

## Your Analysis Should Cover:
1. **Spoilage Risk Assessment**: Which items are at highest risk and why?
2. **Cross-Contamination Risks**: Are high-ethylene producers stored near ethylene-sensitive items?
3. **Cold Chain Observations**: Any temperature anomalies suggesting cold chain issues?
4. **Dispatch Optimization**: Should the current priority order be adjusted? Why?
5. **Preventive Actions**: What should warehouse ops do in the next 24 hours?

Be specific — reference item IDs, produce types, and scores. Keep it concise and actionable."""


DISPATCH_REASONING_PROMPT = """You are an AI dispatch optimizer for a fresh produce warehouse.

Given the current dispatch queue, reason about the optimal dispatch order considering:
- Shelf life urgency (items closest to spoilage go first)
- Ethylene cross-contamination (separate high-ethylene items from sensitive ones during transit)
- Store demand signals (higher demand items should be prioritized)
- Truck loading efficiency (group items going to the same region)

## Current Dispatch Queue
{dispatch_queue}

## Constraints
- Maximum {max_per_list} items per robot pick-list
- Ethylene-risk items MUST be in separate pick-lists
- SHIP_NOW items must leave within 4 hours
- SHIP_SOON items must leave within 24 hours

Provide a JSON response with this structure:
{{
    "recommended_changes": [
        {{
            "item_id": "...",
            "current_rank": N,
            "recommended_rank": N,
            "reason": "..."
        }}
    ],
    "risk_alerts": ["..."],
    "efficiency_notes": ["..."]
}}"""


class LLMReasoningAgent(BaseAgent):
    """Uses Claude API for complex reasoning about warehouse operations.

    This agent adds an intelligence layer on top of the rule-based pipeline,
    providing natural-language analysis and recommendations that operators
    can review and act on.
    """

    def __init__(self, config: dict, api_key: Optional[str] = None):
        super().__init__("LLMReasoningAgent", config)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = "claude-sonnet-4-20250514"
        self._client = None

    @property
    def is_available(self) -> bool:
        """Check if the LLM agent can be used (API key present)."""
        return self.api_key is not None

    def _get_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                self.logger.warning("anthropic package not installed. Run: pip install anthropic")
                return None
        return self._client

    def process(self, pipeline_result: PipelineResult) -> dict:
        """Run LLM analysis on the pipeline result.

        Args:
            pipeline_result: Complete output from the orchestrator pipeline.

        Returns:
            Dict with 'anomaly_analysis' and 'dispatch_reasoning' keys,
            each containing the LLM's analysis text.
        """
        self.clear_events()

        if not self.is_available:
            self.emit_event(
                "llm_skipped",
                "⚠️  LLM Reasoning Agent skipped — no ANTHROPIC_API_KEY set. "
                "Set the environment variable or pass api_key to enable.",
            )
            return self._generate_fallback_analysis(pipeline_result)

        results = {}

        # Analysis 1: Anomaly and risk analysis
        try:
            anomaly_prompt = self._build_anomaly_prompt(pipeline_result)
            anomaly_analysis = self._call_llm(anomaly_prompt)
            results["anomaly_analysis"] = anomaly_analysis
            self.emit_event("llm_anomaly_complete", "🧠 LLM anomaly analysis complete")
        except Exception as e:
            self.logger.error(f"LLM anomaly analysis failed: {e}")
            results["anomaly_analysis"] = f"Analysis unavailable: {e}"

        # Analysis 2: Dispatch optimization reasoning
        try:
            dispatch_prompt = self._build_dispatch_prompt(pipeline_result)
            dispatch_reasoning = self._call_llm(dispatch_prompt)
            results["dispatch_reasoning"] = dispatch_reasoning
            self.emit_event("llm_dispatch_complete", "🧠 LLM dispatch reasoning complete")
        except Exception as e:
            self.logger.error(f"LLM dispatch reasoning failed: {e}")
            results["dispatch_reasoning"] = f"Reasoning unavailable: {e}"

        self.emit_event(
            "llm_complete",
            f"🧠 LLM Reasoning Agent complete — {len(results)} analyses generated",
        )

        return results

    def _call_llm(self, prompt: str) -> str:
        """Make a call to the Claude API."""
        client = self._get_client()
        if client is None:
            return "LLM client unavailable (anthropic package not installed)"

        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _build_anomaly_prompt(self, result: PipelineResult) -> str:
        """Build the anomaly analysis prompt from pipeline data."""
        # Risk items detail
        risk_items = [a for a in result.assessments if a.risk_factors]
        risk_detail = "\n".join(
            f"- [{a.item_id}] {a.produce_type} ({a.variant}): "
            f"score={a.composite_score:.3f}, days_left={a.estimated_days_remaining:.1f}, "
            f"tier={a.tier.value}, risks={a.risk_factors}"
            for a in risk_items
        ) or "No risk factors detected."

        # Top dispatch items
        top_dispatch = result.dispatch_scores[:10]
        dispatch_detail = "\n".join(
            f"- Rank {d.dispatch_rank}: [{d.item_id}] {d.produce_type} ({d.variant}) — "
            f"urgency={d.urgency_score:.3f}, days_left={d.estimated_days_remaining:.1f}, "
            f"cases={d.case_count}, bay={d.bay_location}, ethylene_risk={d.ethylene_risk}"
            for d in top_dispatch
        ) or "No items in dispatch queue."

        return ANOMALY_ANALYSIS_PROMPT.format(
            total_items=result.summary.get("items_scanned", 0),
            tier_distribution=result.summary.get("tier_distribution", {}),
            avg_freshness=result.summary.get("average_freshness_score", 0),
            avg_days=result.summary.get("average_days_remaining", 0),
            risk_items_detail=risk_detail,
            dispatch_queue=dispatch_detail,
        )

    def _build_dispatch_prompt(self, result: PipelineResult) -> str:
        """Build the dispatch reasoning prompt."""
        dispatch_detail = "\n".join(
            f"- Rank {d.dispatch_rank}: [{d.item_id}] {d.produce_type} ({d.variant}) — "
            f"tier={d.tier.value}, urgency={d.urgency_score:.3f}, "
            f"days_left={d.estimated_days_remaining:.1f}, cases={d.case_count}, "
            f"bay={d.bay_location}, ethylene_risk={d.ethylene_risk}"
            for d in result.dispatch_scores
            if d.tier != FreshnessTier.STORE
        ) or "No items queued for dispatch."

        max_per_list = self.config.get("dispatch", {}).get("max_items_per_pick_list", 20)

        return DISPATCH_REASONING_PROMPT.format(
            dispatch_queue=dispatch_detail,
            max_per_list=max_per_list,
        )

    def _generate_fallback_analysis(self, result: PipelineResult) -> dict:
        """Generate rule-based analysis when LLM is unavailable.

        This provides basic insights without requiring an API key,
        ensuring the pipeline always produces useful output.
        """
        assessments = result.assessments
        dispatch_scores = result.dispatch_scores

        # Identify critical items
        critical = [a for a in assessments if a.tier == FreshnessTier.SHIP_NOW]
        risk_items = [a for a in assessments if a.risk_factors]
        ethylene_risks = [d for d in dispatch_scores if d.ethylene_risk]

        # Build analysis
        analysis_parts = []
        analysis_parts.append("## Automated Risk Assessment (Rule-Based Fallback)\n")

        if critical:
            analysis_parts.append(f"### ⚠️ Critical Items ({len(critical)})")
            for item in sorted(critical, key=lambda x: x.composite_score):
                analysis_parts.append(
                    f"- **{item.produce_type}** ({item.variant}) [{item.item_id}]: "
                    f"Score {item.composite_score:.2f}, ~{item.estimated_days_remaining:.1f} days left"
                )
            analysis_parts.append("")

        if ethylene_risks:
            analysis_parts.append(f"### 🧪 Ethylene Cross-Contamination Risks ({len(ethylene_risks)})")
            for item in ethylene_risks:
                analysis_parts.append(
                    f"- **{item.produce_type}** in bay {item.bay_location} — "
                    f"high ethylene emission, separated in dispatch"
                )
            analysis_parts.append("")

        cold_chain = [a for a in assessments if "cold_chain_breach" in a.risk_factors]
        if cold_chain:
            analysis_parts.append(f"### 🌡️ Cold Chain Alerts ({len(cold_chain)})")
            for item in cold_chain:
                analysis_parts.append(
                    f"- **{item.produce_type}** [{item.item_id}]: Temperature deviation detected"
                )
            analysis_parts.append("")

        analysis_parts.append("### 📋 Recommended Actions")
        analysis_parts.append(f"1. Dispatch {len(critical)} SHIP_NOW items within 4 hours")
        if ethylene_risks:
            analysis_parts.append(
                f"2. Verify bay separation for {len(ethylene_risks)} ethylene-risk items"
            )
        if cold_chain:
            analysis_parts.append(
                f"3. Investigate cold chain for {len(cold_chain)} items with temperature anomalies"
            )
        analysis_parts.append(
            f"4. Schedule re-scan for "
            f"{sum(1 for a in assessments if a.tier == FreshnessTier.STORE)} STORE items in 48h"
        )

        fallback_text = "\n".join(analysis_parts)

        return {
            "anomaly_analysis": fallback_text,
            "dispatch_reasoning": (
                "Dispatch order follows rule-based urgency scoring. "
                "Enable LLM reasoning (set ANTHROPIC_API_KEY) for optimized dispatch recommendations."
            ),
            "mode": "fallback",
        }

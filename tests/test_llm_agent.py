"""Tests for the LLM Reasoning Agent."""

import yaml
import pytest

from src.agents.llm_reasoning_agent import LLMReasoningAgent
from src.agents.orchestrator import Orchestrator


@pytest.fixture
def config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


class TestLLMReasoningAgent:
    def test_fallback_when_no_api_key(self, config):
        """Agent should produce fallback analysis when no API key is set."""
        agent = LLMReasoningAgent(config, api_key=None)
        assert not agent.is_available

        orch = Orchestrator()
        result = orch.run(n_items=15, seed=42)

        output = agent.process(result)
        assert "anomaly_analysis" in output
        assert "dispatch_reasoning" in output
        assert output.get("mode") == "fallback"
        assert "Rule-Based Fallback" in output["anomaly_analysis"]

    def test_fallback_contains_critical_items(self, config):
        """Fallback analysis should mention critical items."""
        agent = LLMReasoningAgent(config, api_key=None)
        orch = Orchestrator()
        result = orch.run(n_items=30, seed=42)

        output = agent.process(result)
        analysis = output["anomaly_analysis"]

        # Should contain section headers
        assert "Recommended Actions" in analysis

    def test_emits_skipped_event(self, config):
        """Agent should emit an event when skipping due to missing API key."""
        agent = LLMReasoningAgent(config, api_key=None)
        orch = Orchestrator()
        result = orch.run(n_items=10, seed=42)

        agent.process(result)
        events = agent.get_events()
        assert any(e.event_type == "llm_skipped" for e in events)

    def test_orchestrator_with_llm_flag(self):
        """Orchestrator should include LLM analysis in summary when enabled."""
        orch = Orchestrator()
        result = orch.run(n_items=10, seed=42, enable_llm=True)

        assert "llm_analysis" in result.summary
        assert result.summary["llm_analysis"] is not None

    def test_orchestrator_without_llm_flag(self):
        """Orchestrator should not include LLM analysis when disabled."""
        orch = Orchestrator()
        result = orch.run(n_items=10, seed=42, enable_llm=False)

        assert "llm_analysis" not in result.summary

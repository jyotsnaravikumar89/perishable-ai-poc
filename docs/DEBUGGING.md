# Debugging & Observability Guide

This guide covers how to analyze, debug, and fix issues across the FreshFleet agentic pipeline.

---

## Quick Diagnostics

### See all agent events

```bash
python main.py --items 10 --seed 42 --verbose
```

This prints every event from every agent — scan rejections, consistency violations, re-scans, classification decisions, and dispatch groupings.

### Trace a single item through the pipeline

```bash
python main.py --items 20 --seed 42 --json | python3 -c "
import json, sys
data = json.load(sys.stdin)
item_id = data['assessments'][0]['item_id']
print(f'Tracing item: {item_id}')
for a in data['assessments']:
    if a['item_id'] == item_id:
        print(json.dumps(a, indent=2, default=str))
for pl in data['pick_lists']:
    for item in pl['items']:
        if item['item_id'] == item_id:
            print(f'In pick-list {pl[\"pick_list_id\"]} ({pl[\"priority_label\"]})')
"
```

### Run a single test in isolation

```bash
python -m pytest tests/test_agents.py::TestClassifierAgent::test_fresh_item_tagged_store -v -s
```

The `-s` flag shows print output. The `-v` flag shows verbose test names.

---

## Common Issues and How to Fix Them

### Classifier is tagging fresh items as Ship Now

The composite score is too low. Add this diagnostic print inside `classifier_agent.py` in the `_compute_composite_score` method:

```python
print(f"""
Item: {scan.produce_type} ({scan.item_id})
  color:    {f.color_score:.3f} x {w.get('color_score', 0.30)} = {f.color_score * w.get('color_score', 0.30):.3f}
  firmness: {f.firmness_score:.3f} x {w.get('firmness_score', 0.25)} = {f.firmness_score * w.get('firmness_score', 0.25):.3f}
  blemish:  {f.blemish_score:.3f} x {w.get('blemish_score', 0.20)} = {f.blemish_score * w.get('blemish_score', 0.20):.3f}
  ethylene: {ethylene_normalized:.3f} x {w.get('ethylene_level', 0.15)} = {ethylene_normalized * w.get('ethylene_level', 0.15):.3f}
  temp:     {temp_score:.3f} x {w.get('temperature_delta', 0.10)} = {temp_score * w.get('temperature_delta', 0.10):.3f}
  TOTAL:    {composite:.3f}
""")
```

This shows which feature is dragging the score down. Fix by adjusting thresholds in `config/settings.yaml`.

### Enhanced vision agent is rejecting too many items

Check which consistency rules are firing:

```bash
python main.py --items 30 --seed 42 --verbose 2>&1 | grep -E "Re-scan|rejected|violation"
```

If a rule is too aggressive, adjust its threshold in `enhanced_vision_agent.py`. For example, change the `color_firmness_divergence` gap from 0.45 to 0.55 to make it less sensitive.

### Pick-lists are empty

Everything got tagged T3 Store. Check the tier distribution:

```bash
python main.py --items 20 --seed 42 --json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data['summary']['tier_distribution'])
"
```

Try a different seed or increase inventory size. Some seeds generate mostly fresh inventory.

### Tests pass locally but fail in CI

Go to your repo's Actions tab on GitHub, click the failed run, expand the failing test. Most common cause: Python version differences (CI runs 3.10, 3.11, 3.12). Check for features that only exist in newer Python versions.

---

## Adding Custom Diagnostics

### Emit a custom event from any agent

```python
self.emit_event(
    "debug_custom",
    f"Score breakdown for {scan.produce_type}: total={composite:.3f}",
    {"item_id": scan.item_id, "composite": composite},
)
```

Events appear in verbose output and in the pipeline result's events list.

### Write a regression test for a bug you fixed

```python
def test_soft_tomato_not_tagged_store(self, config):
    """A tomato with low firmness should never be tagged STORE."""
    agent = ClassifierAgent(config)
    scan = ScanResult(
        produce_type="tomato", variant="roma",
        scan_features=ScanFeatures(
            color_score=0.80,
            firmness_score=0.20,
            blemish_score=0.70,
            ethylene_ppm=6.0,
            surface_temp_c=12.0,
            scan_confidence=0.90,
        ),
    )
    result = agent.process([scan])
    assert result[0].tier != FreshnessTier.STORE
```

### Monitor aggregate accuracy with the feedback agent

```python
from src.agents.feedback_agent import FeedbackLoopAgent
from src.agents.orchestrator import Orchestrator

orch = Orchestrator()
result = orch.run(n_items=50, seed=42)

feedback = FeedbackLoopAgent(orch.config)
outcomes = feedback.simulate_outcomes(result.assessments)
report = feedback.process(outcomes)

print(f"Accuracy: {report['accuracy_metrics']['accuracy']:.1%}")
print(f"Bias: {report['accuracy_metrics']['bias']}")
print(f"MAE: {report['accuracy_metrics']['mean_absolute_error_days']:.1f} days")
```

---

## Architecture-Level Observability

Every agent in the pipeline follows the same observability contract:

1. All agents extend `BaseAgent` with a common `emit_event()` method
2. All data flows through Pydantic models — malformed data is caught at agent boundaries, not downstream
3. The orchestrator collects events from all agents into a single `PipelineResult.events` list
4. Every pipeline run has a unique `run_id` for traceability
5. The feedback loop agent monitors aggregate system health, not just individual items

In production, these events would flow to a logging service (ELK, Datadog) for dashboarding and alerting. The pattern is the same — the event schema doesn't change, only the transport.

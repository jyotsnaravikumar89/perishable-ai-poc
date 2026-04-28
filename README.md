# 🥬 FreshFleet — Agentic AI for Perishable Food Warehouse Operations

An **Agentic AI Proof-of-Concept** for intelligent perishable food management in warehouse environments. The system uses a multi-agent orchestration pipeline to optically scan fresh produce, classify freshness into 3-tier priority tags, and generate optimized robot pick-lists for store dispatch.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![CI](https://github.com/YOUR-USERNAME/perishable-ai-poc/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-POC-orange)

---

## Problem Statement

Warehouses handling fresh farm produce (tomatoes, leafy greens, fruits) face a core challenge: **dispatching the right produce at the right time** to minimize spoilage and maximize shelf life at retail. Manual inspection is slow, inconsistent, and doesn't scale.

## Solution: Multi-Agent Orchestration

FreshFleet decomposes the problem into **four autonomous agents** coordinated by a central orchestrator:

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                         │
│            (Pipeline Coordinator + State Manager)       │
├────────────┬────────────┬──────────────┬────────────────┤
│  📷 Vision  │ 🏷️ Classifier│ 📊 Prioritizer│ 🤖 Dispatcher  │
│   Agent    │   Agent    │    Agent     │    Agent       │
│            │            │              │                │
│ Optical    │ 3-Tier     │ Dispatch     │ Robot Pick-    │
│ Scan →     │ Freshness  │ Order +      │ List +         │
│ Features   │ Tagging    │ Scoring      │ Bay Assignment │
└────────────┴────────────┴──────────────┴────────────────┘
```

### The 3-Tier Tagging System

| Tier | Tag | Color | Meaning | Action |
|------|-----|-------|---------|--------|
| **T1** | 🔴 `SHIP_NOW` | Red | ≤2 days remaining shelf life | Immediate dispatch — next outbound truck |
| **T2** | 🟡 `SHIP_SOON` | Yellow | 3–5 days remaining shelf life | Queue for dispatch within 24 hours |
| **T3** | 🟢 `STORE` | Green | 6+ days remaining shelf life | Hold in cold storage, re-scan in 48 hours |

## Architecture

```
                    ┌──────────────┐
                    │   Warehouse  │
                    │   Inventory  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Optical      │   Simulated sensor data:
                    │ Scanner      │   color, firmness, blemishes,
                    │ (Sensor Sim) │   ethylene, temperature
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │     ORCHESTRATOR        │
              │                         │
              │  1. Scan all produce    │
              │  2. Classify freshness  │
              │  3. Prioritize dispatch │
              │  4. Generate pick-lists │
              │  5. Emit events         │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Robot Pick-Lists      │
              │   + Dashboard Output    │
              └─────────────────────────┘
```

## Project Structure

```
perishable-ai-poc/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── config/
│   └── settings.yaml              # Thresholds, weights, agent configs
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── produce.py             # Pydantic data models
│   ├── sensors/
│   │   ├── __init__.py
│   │   └── optical_scanner.py     # Simulated multi-spectrum scanner
│   └── agents/
│       ├── __init__.py
│       ├── base.py                # Abstract agent interface
│       ├── vision_agent.py        # Feature extraction from scans
│       ├── classifier_agent.py    # 3-tier freshness classification
│       ├── prioritizer_agent.py   # Dispatch priority scoring
│       ├── dispatch_agent.py      # Robot pick-list generation
│       ├── llm_reasoning_agent.py # Claude-powered anomaly analysis
│       └── orchestrator.py        # Pipeline coordination engine
├── data/
│   └── sample_inventory.json      # Seed data for demo
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_agents.py
│   └── test_orchestrator.py
├── main.py                        # CLI entry point
├── api.py                         # FastAPI REST service
└── dashboard.py                   # Streamlit dashboard
```

## Quick Start

### Prerequisites

- Python 3.10+

### Install

```bash
git clone https://github.com/<your-username>/perishable-ai-poc.git
cd perishable-ai-poc
pip install -r requirements.txt
```

### Run the POC

```bash
# Run full pipeline with sample inventory
python main.py

# Run with custom inventory size
python main.py --items 50

# Run with verbose agent logging
python main.py --verbose

# Run with LLM reasoning agent (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --llm

# Output raw JSON (for programmatic consumption)
python main.py --json

# Run the Streamlit dashboard
pip install streamlit
streamlit run dashboard.py

# Run the FastAPI REST service
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000
# Then visit http://localhost:8000/docs for interactive API docs
```

### Run Tests

```bash
pytest tests/ -v
```

## Sample Output

```
═══════════════════════════════════════════════════════════
  🥬 FreshFleet — Agentic AI Pipeline Run
  Timestamp: 2026-04-22 14:30:00
  Items Scanned: 24
═══════════════════════════════════════════════════════════

📷 Vision Agent — Scanned 24 items, extracted 5 features each
🏷️  Classifier Agent — Tagged: 6x SHIP_NOW | 10x SHIP_SOON | 8x STORE
📊 Prioritizer Agent — Dispatch queue sorted by urgency score
🤖 Dispatcher Agent — Generated 2 pick-lists across 3 bays

┌─── Pick-List #1 (URGENT) ──────────────────────────────┐
│ Bay A-3: Tomatoes (Roma)    x12 cases │ T1 SHIP_NOW   │
│ Bay B-1: Strawberries       x8 cases  │ T1 SHIP_NOW   │
│ Bay A-7: Spinach            x6 cases  │ T2 SHIP_SOON  │
└─────────────────────────────────────────────────────────┘
```

## REST API

The FastAPI service exposes the pipeline for integration with WMS, robot controllers, and dashboards:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/pipeline/run` | POST | Execute full pipeline (params: `n_items`, `seed`, `enable_llm`) |
| `/pipeline/{run_id}` | GET | Retrieve previous run with full detail |
| `/scan` | POST | Scan and classify a single produce item |
| `/catalog` | GET | List supported produce types and shelf-life data |

```bash
# Run a pipeline
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"n_items": 30, "enable_llm": false}'

# Scan a single item
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"produce_type": "tomato", "variant": "roma", "days_since_harvest": 6.5}'
```

## LLM Reasoning Agent

The optional 5th agent uses the Claude API for complex reasoning that rule-based agents can't handle:

- **Anomaly analysis**: Explains unusual degradation patterns and cross-contamination risks
- **Dispatch optimization**: Reasons about optimal dispatch order considering multiple constraints
- **Root-cause hypotheses**: Generates explanations for cold chain breaches or unexpected spoilage

When no API key is set, the agent automatically falls back to rule-based analysis — the pipeline always produces useful output.

```bash
# Enable with API key
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --items 30 --llm
```

## Extending the POC

### Plug in Real Vision Models

Replace `OpticalScanner` with actual CV inference:

```python
# src/sensors/optical_scanner.py
class RealOpticalScanner(OpticalScanner):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)  # YOLOv8, EfficientNet, etc.

    def scan(self, image: np.ndarray) -> ScanResult:
        features = self.model.predict(image)
        return ScanResult.from_cv_output(features)
```

### Add LLM-Powered Decision Agent

Wire in an LLM for complex dispatch reasoning:

```python
# Example: Use Claude API for nuanced prioritization
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{
        "role": "user",
        "content": f"Given these produce items with freshness scores: {items}, "
                   f"and these store demand signals: {demand}, "
                   f"generate an optimal dispatch plan."
    }]
)
```

### IoT Sensor Integration

The `OpticalScanner` interface supports real sensor backends:

```python
scanner = OpticalScanner(backend="realsense_d455")  # Intel RealSense
scanner = OpticalScanner(backend="basler_ace2")      # Basler industrial camera
```

## Key Design Decisions

1. **Agent-based decomposition** — Each agent has a single responsibility, can be tested independently, and can be swapped without pipeline changes.
2. **Event-driven orchestration** — The orchestrator emits typed events at each stage, enabling logging, monitoring, and future async scaling.
3. **Simulated sensors** — The POC uses physics-informed randomization (ethylene curves, color degradation models) rather than naive random numbers.
4. **Pydantic models** — All data flows through validated schemas, making the system type-safe and self-documenting.

## Roadmap

- [ ] Real computer vision model integration (YOLOv8 + custom freshness classifier)
- [ ] MQTT/ROS2 bridge for physical robot dispatch
- [ ] Time-series tracking (re-scan history per item)
- [ ] Store demand signal integration (pull-based dispatch)
- [ ] Multi-warehouse federation
- [ ] LLM-powered anomaly reasoning agent

## License

MIT

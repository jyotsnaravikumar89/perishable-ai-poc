"""FreshFleet Dashboard — Streamlit-based Pipeline Visualization.

Launch with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime

from src.agents.orchestrator import Orchestrator
from src.models import FreshnessTier


st.set_page_config(page_title="FreshFleet Dashboard", page_icon="🥬", layout="wide")

st.title("🥬 FreshFleet — Agentic AI Dashboard")
st.caption("Perishable Food Warehouse Intelligence & Dispatch Optimization")

# ── Sidebar Controls ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Controls")
    n_items = st.slider("Inventory Size", min_value=5, max_value=100, value=24, step=5)
    seed = st.number_input("Random Seed (0 = random)", min_value=0, max_value=9999, value=42)
    run_pipeline = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# ── Pipeline Execution ──────────────────────────────────────────────────
if run_pipeline or "result" not in st.session_state:
    with st.spinner("Running agentic pipeline..."):
        orchestrator = Orchestrator()
        result = orchestrator.run(n_items=n_items, seed=seed if seed > 0 else None)
        st.session_state["result"] = result

result = st.session_state["result"]

# ── KPI Metrics Row ─────────────────────────────────────────────────────
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)

tier_dist = result.summary.get("tier_distribution", {})
col1.metric("Items Scanned", result.summary.get("items_scanned", 0))
col2.metric("🔴 Ship Now", tier_dist.get("T1_SHIP_NOW", 0))
col3.metric("🟡 Ship Soon", tier_dist.get("T2_SHIP_SOON", 0))
col4.metric("🟢 Store", tier_dist.get("T3_STORE", 0))
col5.metric("Avg Freshness", f"{result.summary.get('average_freshness_score', 0):.2f}")

# ── Tier Distribution Chart ─────────────────────────────────────────────
st.divider()
col_chart, col_detail = st.columns([1, 2])

with col_chart:
    st.subheader("Tier Distribution")
    chart_data = pd.DataFrame({
        "Tier": ["SHIP NOW", "SHIP SOON", "STORE"],
        "Count": [
            tier_dist.get("T1_SHIP_NOW", 0),
            tier_dist.get("T2_SHIP_SOON", 0),
            tier_dist.get("T3_STORE", 0),
        ],
    })
    st.bar_chart(chart_data.set_index("Tier"))

with col_detail:
    st.subheader("All Assessments")
    if result.assessments:
        rows = []
        for a in result.assessments:
            rows.append({
                "ID": a.item_id,
                "Produce": a.produce_type.replace("_", " ").title(),
                "Variant": a.variant,
                "Score": round(a.composite_score, 3),
                "Days Left": round(a.estimated_days_remaining, 1),
                "Tier": a.tier.label,
                "Risks": ", ".join(a.risk_factors) if a.risk_factors else "—",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ── Pick Lists ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🤖 Robot Pick-Lists")

if result.pick_lists:
    tabs = st.tabs([f"{pl.pick_list_id} ({pl.priority_label})" for pl in result.pick_lists])
    for tab, pl in zip(tabs, result.pick_lists):
        with tab:
            st.caption(f"Total cases: {pl.total_cases} | Est. pick time: {pl.estimated_pick_time_min} min")
            rows = []
            for item in pl.items:
                rows.append({
                    "Bay": item.bay_location,
                    "Produce": item.produce_type.replace("_", " ").title(),
                    "Variant": item.variant,
                    "Cases": item.case_count,
                    "Urgency": round(item.urgency_score, 3),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No pick-lists generated — all items are in STORE tier.")

# ── Pipeline Event Log ──────────────────────────────────────────────────
st.divider()
with st.expander("📋 Pipeline Event Log", expanded=False):
    for event in result.events:
        icon = {
            "pipeline_start": "🚀",
            "pipeline_complete": "✅",
            "pipeline_error": "❌",
            "stage_start": "▶️",
            "scan_rejected": "⚠️",
            "ethylene_anomaly": "🧪",
        }.get(event.event_type, "📌")
        st.text(f"{icon} [{event.agent_name}] {event.message}")

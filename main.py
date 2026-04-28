"""FreshFleet — CLI Entry Point.

Run the full agentic AI pipeline with formatted console output.

Usage:
    python main.py                    # Default: 24 items
    python main.py --items 50         # Custom inventory size
    python main.py --seed 42          # Reproducible run
    python main.py --verbose          # Show all agent events
    python main.py --json             # Output raw JSON result
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.agents.orchestrator import Orchestrator
from src.models import FreshnessTier, PipelineResult

console = Console()


def print_header(n_items: int):
    console.print()
    console.print(Panel.fit(
        "[bold green]🥬 FreshFleet — Agentic AI Pipeline[/bold green]\n"
        f"[dim]Perishable Food Warehouse Optimization[/dim]\n"
        f"[dim]Scanning {n_items} inventory items...[/dim]",
        border_style="green",
    ))
    console.print()


def print_events(result: PipelineResult):
    console.print("[bold]Pipeline Events:[/bold]")
    for event in result.events:
        style = {
            "pipeline_start": "bold cyan",
            "pipeline_complete": "bold green",
            "pipeline_error": "bold red",
            "stage_start": "dim",
            "stage_complete": "dim",
        }.get(event.event_type, "white")
        console.print(f"  [{style}]{event.message}[/{style}]")
    console.print()


def print_tier_summary(result: PipelineResult):
    table = Table(title="Freshness Classification Summary", show_header=True)
    table.add_column("Tier", style="bold", width=20)
    table.add_column("Count", justify="center", width=10)
    table.add_column("Avg Score", justify="center", width=12)
    table.add_column("Avg Days Left", justify="center", width=14)

    for tier in FreshnessTier:
        items = [a for a in result.assessments if a.tier == tier]
        if items:
            avg_score = sum(a.composite_score for a in items) / len(items)
            avg_days = sum(a.estimated_days_remaining for a in items) / len(items)
            color = {"T1_SHIP_NOW": "red", "T2_SHIP_SOON": "yellow", "T3_STORE": "green"}[tier.value]
            table.add_row(
                f"[{color}]{tier.label}[/{color}]",
                str(len(items)),
                f"{avg_score:.2f}",
                f"{avg_days:.1f} days",
            )

    console.print(table)
    console.print()


def print_pick_lists(result: PipelineResult):
    if not result.pick_lists:
        console.print("[dim]No pick-lists generated.[/dim]\n")
        return

    for pl in result.pick_lists:
        table = Table(
            title=f"Pick-List {pl.pick_list_id} — {pl.priority_label}",
            show_header=True,
            border_style="blue" if "URGENT" in pl.priority_label else "yellow",
        )
        table.add_column("Bay", width=8)
        table.add_column("Produce", width=18)
        table.add_column("Variant", width=14)
        table.add_column("Cases", justify="center", width=8)
        table.add_column("Tier", width=16)
        table.add_column("Urgency", justify="center", width=10)

        for item in pl.items:
            color = {"T1_SHIP_NOW": "red", "T2_SHIP_SOON": "yellow", "T3_STORE": "green"}[item.tier.value]
            table.add_row(
                item.bay_location,
                item.produce_type.replace("_", " ").title(),
                item.variant,
                str(item.case_count),
                f"[{color}]{item.tier.label}[/{color}]",
                f"{item.urgency_score:.2f}",
            )

        console.print(table)
        console.print(f"  [dim]Total cases: {pl.total_cases} | Est. pick time: {pl.estimated_pick_time_min} min[/dim]")
        console.print()


def print_summary(result: PipelineResult):
    s = result.summary
    elapsed = (result.completed_at - result.started_at).total_seconds() if result.completed_at else 0

    console.print(Panel.fit(
        f"[bold]Pipeline Summary[/bold]\n\n"
        f"  Items scanned:        {s.get('items_scanned', 0)}\n"
        f"  Items validated:      {s.get('items_validated', 0)}\n"
        f"  Avg freshness score:  {s.get('average_freshness_score', 0):.3f}\n"
        f"  Avg days remaining:   {s.get('average_days_remaining', 0):.1f}\n"
        f"  Pick-lists generated: {s.get('pick_lists_generated', 0)}\n"
        f"  Total dispatch cases: {s.get('total_dispatch_cases', 0)}\n"
        f"  Risk-flagged items:   {s.get('items_with_risk_factors', 0)}\n"
        f"  Elapsed time:         {elapsed:.2f}s",
        title="✅ Run Complete",
        border_style="green",
    ))

    if s.get("unique_risk_types"):
        console.print(f"\n  [yellow]Risk types detected:[/yellow] {', '.join(s['unique_risk_types'])}")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="FreshFleet — Agentic AI for Perishable Food")
    parser.add_argument("--items", type=int, default=24, help="Number of inventory items to scan (default: 24)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Show all pipeline events")
    parser.add_argument("--json", action="store_true", help="Output raw JSON result")
    parser.add_argument("--llm", action="store_true", help="Enable LLM reasoning agent (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s | %(message)s")

    if not args.json:
        print_header(args.items)

    # Run the pipeline
    orchestrator = Orchestrator()
    result = orchestrator.run(n_items=args.items, seed=args.seed, enable_llm=args.llm)

    if args.json:
        # Raw JSON output for programmatic consumption
        output = {
            "run_id": result.run_id,
            "summary": result.summary,
            "assessments": [a.model_dump(mode="json") for a in result.assessments],
            "pick_lists": [pl.model_dump(mode="json") for pl in result.pick_lists],
            "events": [e.model_dump(mode="json") for e in result.events],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        if args.verbose:
            print_events(result)

        print_tier_summary(result)
        print_pick_lists(result)
        print_summary(result)

        # Display LLM analysis if available
        llm = result.summary.get("llm_analysis")
        if llm:
            console.print(Panel.fit(
                llm.get("anomaly_analysis", "No analysis available."),
                title="🧠 LLM Reasoning Agent — Analysis",
                border_style="magenta",
            ))
            console.print()


if __name__ == "__main__":
    main()

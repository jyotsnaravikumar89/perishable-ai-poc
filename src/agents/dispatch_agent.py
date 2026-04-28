"""Dispatch Agent — Robot Pick-List Generation.

This agent takes the prioritized dispatch queue and generates optimized
pick-lists for warehouse robots. Each pick-list represents a single
robot dispatch run.

Key behaviors:
    - Groups items by priority tier into separate pick-lists
    - Respects max items per pick-list constraint
    - Separates ethylene-risk items to prevent cross-contamination during transit
    - Estimates pick time based on item count and bay spread
"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.models import DispatchScore, FreshnessTier, PickItem, PickList


class DispatchAgent(BaseAgent):
    """Generates optimized robot pick-lists from prioritized dispatch scores."""

    def __init__(self, config: dict):
        super().__init__("DispatchAgent", config)
        dispatch_config = config.get("dispatch", {})
        self.max_items_per_list = dispatch_config.get("max_items_per_pick_list", 20)

    def process(self, dispatch_scores: list[DispatchScore]) -> list[PickList]:
        """Generate pick-lists grouped by priority tier.

        Args:
            dispatch_scores: Ranked dispatch scores from PrioritizerAgent.

        Returns:
            List of PickList objects ready for robot execution.
        """
        self.clear_events()

        # Partition items by tier
        tier_groups: dict[FreshnessTier, list[DispatchScore]] = {
            FreshnessTier.SHIP_NOW: [],
            FreshnessTier.SHIP_SOON: [],
            FreshnessTier.STORE: [],
        }

        for score in dispatch_scores:
            tier_groups[score.tier].append(score)

        pick_lists = []

        # Generate pick-lists for SHIP_NOW (URGENT)
        if tier_groups[FreshnessTier.SHIP_NOW]:
            urgent_lists = self._create_pick_lists(
                tier_groups[FreshnessTier.SHIP_NOW],
                priority_label="URGENT",
            )
            pick_lists.extend(urgent_lists)

        # Generate pick-lists for SHIP_SOON (STANDARD)
        if tier_groups[FreshnessTier.SHIP_SOON]:
            standard_lists = self._create_pick_lists(
                tier_groups[FreshnessTier.SHIP_SOON],
                priority_label="STANDARD",
            )
            pick_lists.extend(standard_lists)

        # STORE items don't get pick-lists (they stay in cold storage)
        store_count = len(tier_groups[FreshnessTier.STORE])

        total_cases = sum(pl.total_cases for pl in pick_lists)
        self.emit_event(
            "dispatch_complete",
            f"🤖 Generated {len(pick_lists)} pick-list(s) — "
            f"{total_cases} total cases for dispatch, "
            f"{store_count} items held in storage",
            {
                "pick_lists": len(pick_lists),
                "total_cases": total_cases,
                "stored_items": store_count,
            },
        )

        return pick_lists

    def _create_pick_lists(
        self,
        items: list[DispatchScore],
        priority_label: str,
    ) -> list[PickList]:
        """Create pick-lists from a group of items, splitting if needed.

        Separates ethylene-risk items into their own pick-lists to
        prevent cross-contamination during transit.
        """
        # Separate ethylene-risk items
        safe_items = [i for i in items if not i.ethylene_risk]
        risky_items = [i for i in items if i.ethylene_risk]

        pick_lists = []

        # Create pick-lists for safe items
        for chunk in self._chunk(safe_items, self.max_items_per_list):
            pick_lists.append(self._build_pick_list(chunk, priority_label))

        # Create separate pick-lists for ethylene-risk items
        for chunk in self._chunk(risky_items, self.max_items_per_list):
            pick_lists.append(self._build_pick_list(chunk, f"{priority_label} (ETH-RISK)"))

        return pick_lists

    def _build_pick_list(self, items: list[DispatchScore], priority_label: str) -> PickList:
        """Build a single PickList from dispatch scores."""
        pick_items = [
            PickItem(
                item_id=item.item_id,
                produce_type=item.produce_type,
                variant=item.variant,
                case_count=item.case_count,
                bay_location=item.bay_location,
                tier=item.tier,
                urgency_score=item.urgency_score,
            )
            for item in items
        ]

        return PickList(
            priority_label=priority_label,
            items=pick_items,
        )

    @staticmethod
    def _chunk(items: list, size: int) -> list[list]:
        """Split a list into chunks of a given size."""
        return [items[i:i + size] for i in range(0, len(items), size)] if items else []

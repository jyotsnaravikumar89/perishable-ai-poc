"""Abstract base class for all FreshFleet agents.

Every agent in the pipeline follows the same contract:
    1. Receives typed input
    2. Processes autonomously
    3. Emits typed output + pipeline events

This enables agents to be tested independently, swapped without
pipeline changes, and monitored via a unified event system.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

from src.models import PipelineEvent

T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")


class BaseAgent(ABC):
    """Base contract for all pipeline agents."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"freshfleet.{name}")
        self._events: list[PipelineEvent] = []

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Execute the agent's core logic. Subclasses must implement."""
        ...

    def emit_event(self, event_type: str, message: str, payload: dict | None = None) -> PipelineEvent:
        """Emit a typed pipeline event for observability."""
        event = PipelineEvent(
            event_type=event_type,
            agent_name=self.name,
            message=message,
            payload=payload or {},
        )
        self._events.append(event)
        self.logger.info(f"[{self.name}] {message}")
        return event

    def get_events(self) -> list[PipelineEvent]:
        """Return all events emitted during this agent's processing."""
        return list(self._events)

    def clear_events(self) -> None:
        self._events.clear()

"""Planner interfaces for the SMART-LLM rebuild.

The planner layer stays model-agnostic so the benchmark runner does not care
which provider generated the plan. Tool calling means the model emits one
structured function invocation instead of free text, which gives us a stable
JSON contract across Anthropic, OpenAI, and Kimi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from smart_llm_v2.agents.plan import TaskPlan
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import RobotSpec


@dataclass(frozen=True, slots=True)
class PlanningImage:
    data: bytes
    media_type: str = "image/png"
    agent_id: int | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class PlanBuildResult:
    plan: TaskPlan
    provider: str | None = None
    model: str | None = None
    usage: Mapping[str, object] | None = None
    profile_variant: str | None = None


class Planner(Protocol):
    @property
    def uses_planning_images(self) -> bool: ...

    def build_plan(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        scene_objects: Sequence[Mapping[str, object]],
        planning_images: Sequence[PlanningImage] = (),
    ) -> PlanBuildResult: ...

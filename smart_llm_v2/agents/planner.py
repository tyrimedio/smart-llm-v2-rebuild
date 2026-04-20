"""Planner interfaces for the SMART-LLM rebuild.

The planner layer is model-agnostic so we can compare the paper's staged GPT prompts
against newer tool-calling planners without changing the benchmark harness. Tool
calling means the model returns structured action data instead of free-form Python,
which makes the plan easier to validate, cache, and replay in the simulator.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from smart_llm_v2.agents.plan import TaskPlan
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import RobotSpec


class Planner(Protocol):
    def build_plan(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        scene_objects: Sequence[Mapping[str, object]],
    ) -> TaskPlan: ...

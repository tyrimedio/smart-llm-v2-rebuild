from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GoalState:
    name: str
    contains: tuple[str, ...] = ()
    state: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    floor_plan: int
    task_index: int
    instruction: str
    robot_ids: tuple[int, ...]
    goal_states: tuple[GoalState, ...] = ()
    transition_count: int = 0
    max_transition_count: int = 0
    source_path: Path | None = None
    source_line: int | None = None

    @property
    def task_id(self) -> str:
        return f"FloorPlan{self.floor_plan}:{self.task_index}"

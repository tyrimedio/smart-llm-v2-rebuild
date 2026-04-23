from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionRequest:
    robots: tuple[str, ...]
    skill: str
    object_name: str | None = None
    receptacle_name: str | None = None

    def __post_init__(self) -> None:
        if not self.robots:
            raise ValueError("ActionRequest requires at least one robot")


@dataclass(frozen=True, slots=True)
class PlanSubTask:
    actions: tuple[ActionRequest, ...]
    assigned_robots: tuple[str, ...] = ()
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("PlanSubTask requires at least one action")
        if self.assigned_robots:
            return
        robots = tuple(dict.fromkeys(robot for action in self.actions for robot in action.robots))
        object.__setattr__(self, "assigned_robots", robots)


@dataclass(frozen=True, slots=True, init=False)
class PlanPhase:
    subtasks: tuple[PlanSubTask, ...]
    label: str | None = None

    def __init__(
        self,
        *,
        subtasks: tuple[PlanSubTask, ...] | None = None,
        actions: tuple[ActionRequest, ...] | None = None,
        label: str | None = None,
    ) -> None:
        if subtasks is not None and actions is not None:
            raise ValueError("PlanPhase accepts either subtasks or actions, not both")
        if subtasks is None:
            subtasks = (PlanSubTask(actions=actions),) if actions else ()
        object.__setattr__(self, "subtasks", subtasks)
        object.__setattr__(self, "label", label)

    @property
    def actions(self) -> tuple[ActionRequest, ...]:
        return tuple(action for subtask in self.subtasks for action in subtask.actions)


@dataclass(frozen=True, slots=True)
class TaskPlan:
    phases: tuple[PlanPhase, ...]
    planner_name: str | None = None
    notes: str | None = None

    @property
    def transition_count(self) -> int:
        active_phases = sum(1 for phase in self.phases if phase.actions)
        return max(active_phases - 1, 0)

    @property
    def total_actions(self) -> int:
        return sum(len(phase.actions) for phase in self.phases)

    def flatten(self) -> tuple[ActionRequest, ...]:
        return tuple(action for phase in self.phases for action in phase.actions)

    @classmethod
    def sequential(
        cls,
        *actions: ActionRequest,
        planner_name: str | None = None,
        notes: str | None = None,
    ) -> "TaskPlan":
        return cls(
            phases=tuple(PlanPhase(actions=(action,)) for action in actions),
            planner_name=planner_name,
            notes=notes,
        )

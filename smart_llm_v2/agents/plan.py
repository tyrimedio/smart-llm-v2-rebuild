from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionRequest:
    robot: str
    skill: str
    object_name: str | None = None
    receptacle_name: str | None = None


@dataclass(frozen=True, slots=True)
class PlanPhase:
    actions: tuple[ActionRequest, ...]
    label: str | None = None


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

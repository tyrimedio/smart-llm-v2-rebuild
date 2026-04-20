from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from smart_llm_v2.agents.plan import ActionRequest, TaskPlan
from smart_llm_v2.robots import RobotSpec
from smart_llm_v2.skills import get_skill


class EnvironmentAdapter(Protocol):
    def navigate_to_object(self, *, agent_id: int, object_name: str): ...

    def perform_action(
        self,
        *,
        agent_id: int,
        action_name: str,
        target_name: str | None = None,
    ): ...

    def scene_objects(self) -> Sequence[Mapping[str, object]]: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    request: ActionRequest
    succeeded: bool
    error_message: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    plan: TaskPlan
    records: tuple[ExecutionRecord, ...]
    observed_objects: tuple[Mapping[str, object], ...]
    transition_count: int
    successful_actions: int
    total_actions: int


class ExecutionError(RuntimeError):
    """Raised when the executor cannot dispatch a requested action."""


class BaselineExecutor:
    def __init__(self, *, environment: EnvironmentAdapter, robots: Sequence[RobotSpec]) -> None:
        self.environment = environment
        self._robots_by_name = {robot.name: robot for robot in robots}
        self._agent_ids = {robot.name: index for index, robot in enumerate(robots)}

    def execute_plan(self, steps: Sequence[ActionRequest]) -> list[ExecutionRecord]:
        return [self.execute_step(step) for step in steps]

    def run_plan(self, plan: TaskPlan) -> ExecutionReport:
        records: list[ExecutionRecord] = []
        executed_phases = 0

        # Phases preserve the paper's transition-count metric even before
        # we introduce true concurrent execution.
        for phase in plan.phases:
            if not phase.actions:
                continue
            executed_phases += 1
            for action in phase.actions:
                records.append(self.execute_step(action))

        successful_actions = sum(record.succeeded for record in records)
        return ExecutionReport(
            plan=plan,
            records=tuple(records),
            observed_objects=tuple(self.scene_objects()),
            transition_count=max(executed_phases - 1, 0),
            successful_actions=successful_actions,
            total_actions=len(records),
        )

    def scene_objects(self) -> Sequence[Mapping[str, object]]:
        return self.environment.scene_objects()

    def close(self) -> None:
        self.environment.stop()

    def execute_step(self, step: ActionRequest) -> ExecutionRecord:
        robot = self._get_robot(step.robot)
        if not robot.can(step.skill):
            raise ExecutionError(f"{robot.name} does not have skill {step.skill}")

        agent_id = self._agent_ids[robot.name]
        skill = get_skill(step.skill)

        if step.skill == "GoToObject":
            if step.object_name is None:
                raise ExecutionError("GoToObject requires object_name")
            outcome = self.environment.navigate_to_object(
                agent_id=agent_id,
                object_name=step.object_name,
            )
            return ExecutionRecord(
                request=step,
                succeeded=outcome.succeeded,
                error_message=outcome.error_message,
            )

        target_name = self._target_name_for_step(step)
        outcome = self.environment.perform_action(
            agent_id=agent_id,
            action_name=skill.simulator_action or skill.name,
            target_name=target_name,
        )
        return ExecutionRecord(
            request=step,
            succeeded=outcome.succeeded,
            error_message=outcome.error_message,
        )

    def _get_robot(self, name: str) -> RobotSpec:
        try:
            return self._robots_by_name[name]
        except KeyError as exc:
            raise ExecutionError(f"Unknown robot {name}") from exc

    def _target_name_for_step(self, step: ActionRequest) -> str | None:
        if step.skill == "PutObject":
            if step.receptacle_name is None:
                raise ExecutionError("PutObject requires receptacle_name")
            return step.receptacle_name

        if step.skill in {"DropHandObject", "ThrowObject"}:
            return step.object_name

        if step.object_name is None:
            raise ExecutionError(f"{step.skill} requires object_name")
        return step.object_name

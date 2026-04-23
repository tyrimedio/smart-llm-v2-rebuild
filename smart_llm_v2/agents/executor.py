from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from smart_llm_v2.agents.plan import ActionRequest, TaskPlan
from smart_llm_v2.agents.planner import PlanningImage
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

    def planning_images(self, *, agent_ids: Sequence[int]) -> Sequence[PlanningImage]: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    request: ActionRequest
    succeeded: bool
    error_message: str = ""
    successful_simulator_calls: int | None = None
    total_simulator_calls: int | None = None

    def __post_init__(self) -> None:
        total_calls = self.total_simulator_calls
        if total_calls is None:
            total_calls = len(self.request.robots)
            object.__setattr__(self, "total_simulator_calls", total_calls)

        if self.successful_simulator_calls is None:
            successful_calls = total_calls if self.succeeded else 0
            object.__setattr__(self, "successful_simulator_calls", successful_calls)


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
        total_actions = len(records)
        return ExecutionReport(
            plan=plan,
            records=tuple(records),
            observed_objects=tuple(self.scene_objects()),
            transition_count=max(executed_phases - 1, 0),
            successful_actions=successful_actions,
            total_actions=total_actions,
        )

    def scene_objects(self) -> Sequence[Mapping[str, object]]:
        return self.environment.scene_objects()

    def planning_images(self) -> Sequence[PlanningImage]:
        return self.environment.planning_images(agent_ids=tuple(self._agent_ids.values()))

    def close(self) -> None:
        self.environment.stop()

    def execute_step(self, step: ActionRequest) -> ExecutionRecord:
        robots = self._get_robots(step.robots)
        for robot in robots:
            if not robot.can(step.skill):
                raise ExecutionError(f"{robot.name} does not have skill {step.skill}")

        skill = get_skill(step.skill)

        if step.skill == "GoToObject":
            if step.object_name is None:
                raise ExecutionError("GoToObject requires object_name")
            outcomes = self._navigate_team(
                robots=robots,
                object_name=step.object_name,
            )
            return self._execution_record(step=step, outcomes=outcomes)

        target_name = self._target_name_for_step(step)
        outcomes = self._perform_team_action(
            robots=robots,
            action_name=skill.simulator_action or skill.name,
            target_name=target_name,
        )
        return self._execution_record(step=step, outcomes=outcomes)

    def _get_robot(self, name: str) -> RobotSpec:
        try:
            return self._robots_by_name[name]
        except KeyError as exc:
            raise ExecutionError(f"Unknown robot {name}") from exc

    def _get_robots(self, names: Sequence[str]) -> tuple[RobotSpec, ...]:
        return tuple(self._get_robot(name) for name in names)

    def _navigate_team(
        self,
        *,
        robots: Sequence[RobotSpec],
        object_name: str,
    ):
        outcomes = [
            self.environment.navigate_to_object(
                agent_id=self._agent_ids[robot.name],
                object_name=object_name,
            )
            for robot in robots
        ]
        return outcomes

    def _perform_action(
        self,
        *,
        robot: RobotSpec,
        action_name: str,
        target_name: str | None,
    ):
        return self.environment.perform_action(
            agent_id=self._agent_ids[robot.name],
            action_name=action_name,
            target_name=target_name,
        )

    def _perform_team_action(
        self,
        *,
        robots: Sequence[RobotSpec],
        action_name: str,
        target_name: str | None,
    ):
        outcomes = [
            self._perform_action(
                robot=robot,
                action_name=action_name,
                target_name=target_name,
            )
            for robot in robots
        ]
        return outcomes

    def _execution_record(self, *, step: ActionRequest, outcomes):
        errors = [outcome.error_message for outcome in outcomes if outcome.error_message]
        return ExecutionRecord(
            request=step,
            succeeded=all(outcome.succeeded for outcome in outcomes),
            error_message="; ".join(dict.fromkeys(errors)),
            successful_simulator_calls=sum(outcome.succeeded for outcome in outcomes),
            total_simulator_calls=len(outcomes),
        )

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

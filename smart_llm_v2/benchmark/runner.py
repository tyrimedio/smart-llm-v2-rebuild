from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from smart_llm_v2.agents.executor import ExecutionReport
from smart_llm_v2.agents.plan import TaskPlan
from smart_llm_v2.agents.planner import Planner
from smart_llm_v2.benchmark.metrics import TaskMetrics, compute_metrics
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import RobotSpec, build_task_robot_team


class TaskExecutor(Protocol):
    def scene_objects(self) -> Sequence[Mapping[str, object]]: ...

    def run_plan(self, plan: TaskPlan) -> ExecutionReport: ...

    def close(self) -> None: ...


class ExecutorFactory(Protocol):
    def __call__(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
    ) -> TaskExecutor: ...


@dataclass(frozen=True, slots=True)
class TaskRunResult:
    task: BenchmarkTask
    robots: tuple[RobotSpec, ...]
    plan: TaskPlan
    execution: ExecutionReport
    metrics: TaskMetrics


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    task_runs: tuple[TaskRunResult, ...]

    @property
    def task_count(self) -> int:
        return len(self.task_runs)

    def mean_metrics(self) -> dict[str, float]:
        if not self.task_runs:
            return {
                "success_rate": 0.0,
                "task_completion_rate": 0.0,
                "goal_condition_recall": 0.0,
                "robot_utilization": 0.0,
                "executability": 0.0,
            }

        return {
            "success_rate": sum(run.metrics.success_rate for run in self.task_runs)
            / self.task_count,
            "task_completion_rate": sum(
                run.metrics.task_completion_rate for run in self.task_runs
            )
            / self.task_count,
            "goal_condition_recall": sum(
                run.metrics.goal_condition_recall for run in self.task_runs
            )
            / self.task_count,
            "robot_utilization": sum(
                run.metrics.robot_utilization for run in self.task_runs
            )
            / self.task_count,
            "executability": sum(run.metrics.executability for run in self.task_runs)
            / self.task_count,
        }


class BenchmarkRunner:
    def __init__(
        self,
        *,
        planner: Planner,
        executor_factory: ExecutorFactory,
    ) -> None:
        self.planner = planner
        self.executor_factory = executor_factory

    def run_task(self, task: BenchmarkTask) -> TaskRunResult:
        robots = build_task_robot_team(task.robot_ids)
        executor = self.executor_factory(task=task, robots=robots)
        try:
            plan = self.planner.build_plan(
                task=task,
                robots=robots,
                scene_objects=tuple(executor.scene_objects()),
            )
            execution = executor.run_plan(plan)
            metrics = compute_metrics(
                goal_states=task.goal_states,
                observed_objects=execution.observed_objects,
                transition_count=execution.transition_count,
                transition_count_ground_truth=task.transition_count,
                max_transition_count=task.max_transition_count,
                successful_actions=execution.successful_actions,
                total_actions=execution.total_actions,
            )
        finally:
            executor.close()

        return TaskRunResult(
            task=task,
            robots=tuple(robots),
            plan=plan,
            execution=execution,
            metrics=metrics,
        )

    def run_benchmark(self, tasks: Sequence[BenchmarkTask]) -> BenchmarkSummary:
        return BenchmarkSummary(task_runs=tuple(self.run_task(task) for task in tasks))

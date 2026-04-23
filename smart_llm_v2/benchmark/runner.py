from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from smart_llm_v2.agents.executor import ExecutionReport
from smart_llm_v2.agents.plan import TaskPlan
from smart_llm_v2.agents.planner import Planner, PlanningImage
from smart_llm_v2.agents.verifier import PlanVerifier, VerificationIssue
from smart_llm_v2.benchmark.metrics import TaskMetrics, compute_metrics
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import RobotSpec, build_task_robot_team


class TaskExecutor(Protocol):
    def scene_objects(self) -> Sequence[Mapping[str, object]]: ...

    def planning_images(self) -> Sequence[PlanningImage]: ...

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
    planner_provider: str | None = None
    planner_model: str | None = None
    planner_usage: Mapping[str, object] | None = None
    planner_profile_variant: str | None = None
    verifier_provider: str | None = None
    verifier_model: str | None = None
    verifier_usage: Mapping[str, object] | None = None
    verification_issues: tuple[VerificationIssue, ...] = ()
    error_message: str | None = None

    @classmethod
    def failed(
        cls,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        error_message: str,
        plan: TaskPlan | None = None,
        planner_provider: str | None = None,
        planner_model: str | None = None,
        planner_usage: Mapping[str, object] | None = None,
        planner_profile_variant: str | None = None,
        verifier_provider: str | None = None,
        verifier_model: str | None = None,
        verifier_usage: Mapping[str, object] | None = None,
        verification_issues: Sequence[VerificationIssue] = (),
    ) -> "TaskRunResult":
        resolved_plan = plan or TaskPlan(phases=())
        execution = ExecutionReport(
            plan=resolved_plan,
            records=(),
            observed_objects=(),
            transition_count=0,
            successful_actions=0,
            total_actions=0,
        )
        return cls(
            task=task,
            robots=tuple(robots),
            plan=resolved_plan,
            execution=execution,
            metrics=TaskMetrics(
                success_rate=0.0,
                task_completion_rate=0.0,
                goal_condition_recall=0.0,
                robot_utilization=0.0,
                executability=0.0,
            ),
            planner_provider=planner_provider,
            planner_model=planner_model,
            planner_usage=planner_usage,
            planner_profile_variant=planner_profile_variant,
            verifier_provider=verifier_provider,
            verifier_model=verifier_model,
            verifier_usage=verifier_usage,
            verification_issues=tuple(verification_issues),
            error_message=error_message,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    task_runs: tuple[TaskRunResult, ...]

    @property
    def task_count(self) -> int:
        return len(self.task_runs)

    @property
    def failed_task_count(self) -> int:
        return sum(run.error_message is not None for run in self.task_runs)

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
        verifier: PlanVerifier | None = None,
    ) -> None:
        self.planner = planner
        self.executor_factory = executor_factory
        self.verifier = verifier

    def run_task(self, task: BenchmarkTask) -> TaskRunResult:
        robots = build_task_robot_team(task.robot_ids)
        executor = self.executor_factory(task=task, robots=robots)
        try:
            planning_images: tuple[PlanningImage, ...] = ()
            scene_objects = tuple(executor.scene_objects())
            if self.planner.uses_planning_images:
                planning_images = tuple(executor.planning_images())
            planning = self.planner.build_plan(
                task=task,
                robots=robots,
                scene_objects=scene_objects,
                planning_images=planning_images,
            )
            verification = None
            if self.verifier is not None:
                verification = self.verifier.verify(
                    task=task,
                    robots=robots,
                    scene_objects=scene_objects,
                    plan=planning.plan,
                    planning_images=planning_images,
                )
                if not verification.passed:
                    return TaskRunResult.failed(
                        task=task,
                        robots=robots,
                        plan=planning.plan,
                        planner_provider=planning.provider,
                        planner_model=planning.model,
                        planner_usage=planning.usage,
                        planner_profile_variant=planning.profile_variant,
                        verifier_provider=verification.provider,
                        verifier_model=verification.model,
                        verifier_usage=verification.usage,
                        verification_issues=verification.issues,
                        error_message=verification.error_message(),
                    )
            execution = executor.run_plan(planning.plan)
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
            plan=planning.plan,
            execution=execution,
            metrics=metrics,
            planner_provider=planning.provider,
            planner_model=planning.model,
            planner_usage=planning.usage,
            planner_profile_variant=planning.profile_variant,
            verifier_provider=verification.provider if verification is not None else None,
            verifier_model=verification.model if verification is not None else None,
            verifier_usage=verification.usage if verification is not None else None,
            verification_issues=verification.issues if verification is not None else (),
        )

    def run_benchmark(self, tasks: Sequence[BenchmarkTask]) -> BenchmarkSummary:
        task_runs: list[TaskRunResult] = []
        for task in tasks:
            try:
                task_runs.append(self.run_task(task))
            except Exception as exc:
                try:
                    robots = build_task_robot_team(task.robot_ids)
                except Exception:
                    robots = ()
                task_runs.append(
                    TaskRunResult.failed(
                        task=task,
                        robots=robots,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                )
        return BenchmarkSummary(task_runs=tuple(task_runs))

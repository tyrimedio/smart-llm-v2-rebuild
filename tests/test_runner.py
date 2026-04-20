from __future__ import annotations

from smart_llm_v2.agents.plan import ActionRequest, PlanPhase, TaskPlan
from smart_llm_v2.benchmark.models import BenchmarkTask, GoalState
from smart_llm_v2.benchmark.runner import BenchmarkRunner
from smart_llm_v2.robots import RobotSpec


class FakePlanner:
    def __init__(self, plan: TaskPlan) -> None:
        self.plan = plan
        self.calls: list[dict[str, object]] = []

    def build_plan(self, *, task, robots, scene_objects):
        self.calls.append(
            {
                "task": task.instruction,
                "robots": tuple(robot.name for robot in robots),
                "scene_objects": tuple(scene_objects),
            }
        )
        return self.plan


class FakeExecutor:
    def __init__(self, observed_objects) -> None:
        self._observed_objects = tuple(observed_objects)
        self.closed = False
        self.received_plan: TaskPlan | None = None

    def scene_objects(self):
        return self._observed_objects

    def run_plan(self, plan: TaskPlan):
        from smart_llm_v2.agents.executor import ExecutionRecord, ExecutionReport

        self.received_plan = plan
        records = tuple(
            ExecutionRecord(request=action, succeeded=True)
            for action in plan.flatten()
        )
        return ExecutionReport(
            plan=plan,
            records=records,
            observed_objects=self._observed_objects,
            transition_count=plan.transition_count,
            successful_actions=len(records),
            total_actions=len(records),
        )

    def close(self) -> None:
        self.closed = True


class FakeExecutorFactory:
    def __init__(self, executor: FakeExecutor) -> None:
        self.executor = executor
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, *, task: BenchmarkTask, robots: tuple[RobotSpec, ...]) -> FakeExecutor:
        self.calls.append((task.instruction, tuple(robot.name for robot in robots)))
        return self.executor


def test_task_plan_tracks_parallel_phase_transitions() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robot="robot1", skill="GoToObject", object_name="Laptop"),
                    ActionRequest(robot="robot2", skill="GoToObject", object_name="LightSwitch"),
                )
            ),
            PlanPhase(
                actions=(
                    ActionRequest(robot="robot1", skill="SwitchOn", object_name="Laptop"),
                )
            ),
        )
    )

    assert plan.transition_count == 1
    assert plan.total_actions == 3


def test_benchmark_runner_connects_planner_executor_and_metrics() -> None:
    task = BenchmarkTask(
        floor_plan=303,
        task_index=1,
        instruction="Turn on the laptop",
        robot_ids=(24, 1),
        goal_states=(GoalState(name="Laptop", state="ON"),),
    )
    plan = TaskPlan.sequential(
        ActionRequest(robot="robot2", skill="GoToObject", object_name="Laptop"),
        ActionRequest(robot="robot2", skill="SwitchOn", object_name="Laptop"),
        planner_name="fake-planner",
    )
    scene_objects = ({"name": "Laptop|0", "isToggled": True},)
    planner = FakePlanner(plan)
    executor = FakeExecutor(scene_objects)
    runner = BenchmarkRunner(
        planner=planner,
        executor_factory=FakeExecutorFactory(executor),
    )

    result = runner.run_task(task)

    assert planner.calls[0]["task"] == "Turn on the laptop"
    assert planner.calls[0]["robots"] == ("robot1", "robot2")
    assert result.execution.transition_count == 1
    assert result.metrics.goal_condition_recall == 1.0
    assert result.metrics.success_rate == 1.0
    assert executor.closed is True


def test_benchmark_summary_averages_task_metrics() -> None:
    task = BenchmarkTask(
        floor_plan=15,
        task_index=1,
        instruction="Make the kitchen dark",
        robot_ids=(24,),
        goal_states=(GoalState(name="LightSwitch", state="OFF"),),
    )
    off_plan = TaskPlan.sequential(
        ActionRequest(robot="robot1", skill="GoToObject", object_name="LightSwitch"),
        ActionRequest(robot="robot1", skill="SwitchOff", object_name="LightSwitch"),
    )
    off_runner = BenchmarkRunner(
        planner=FakePlanner(off_plan),
        executor_factory=FakeExecutorFactory(
            FakeExecutor(({"name": "LightSwitch|0", "isToggled": False},))
        ),
    )

    on_plan = TaskPlan.sequential(
        ActionRequest(robot="robot1", skill="GoToObject", object_name="LightSwitch"),
    )
    on_runner = BenchmarkRunner(
        planner=FakePlanner(on_plan),
        executor_factory=FakeExecutorFactory(
            FakeExecutor(({"name": "LightSwitch|0", "isToggled": True},))
        ),
    )

    summary = off_runner.run_benchmark((task,))
    summary = type(summary)(task_runs=summary.task_runs + on_runner.run_benchmark((task,)).task_runs)

    assert summary.task_count == 2
    means = summary.mean_metrics()
    assert means["task_completion_rate"] == 0.5
    assert means["executability"] == 1.0

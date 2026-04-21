from __future__ import annotations

from smart_llm_v2.agents.plan import ActionRequest, PlanPhase, TaskPlan
from smart_llm_v2.agents.planner import PlanBuildResult, PlanningImage
from smart_llm_v2.benchmark.models import BenchmarkTask, GoalState
from smart_llm_v2.benchmark.runner import BenchmarkRunner
from smart_llm_v2.robots import RobotSpec


class FakePlanner:
    def __init__(
        self,
        plan: TaskPlan,
        *,
        uses_planning_images: bool = True,
        failure_calls: set[int] | None = None,
    ) -> None:
        self.plan = plan
        self.uses_planning_images = uses_planning_images
        self.calls: list[dict[str, object]] = []
        self.failure_calls = failure_calls or set()

    def build_plan(self, *, task, robots, scene_objects, planning_images=()):
        call_index = len(self.calls) + 1
        self.calls.append(
            {
                "task": task.instruction,
                "robots": tuple(robot.name for robot in robots),
                "scene_objects": tuple(scene_objects),
                "planning_images": tuple(planning_images),
            }
        )
        if call_index in self.failure_calls:
            raise RuntimeError("boom")
        return PlanBuildResult(
            plan=self.plan,
            provider="anthropic",
            model="claude-opus-4-7",
            usage={"prompt_tokens": 42},
            profile_variant="symbolic",
        )


class FakeExecutor:
    def __init__(self, observed_objects, planning_images=()) -> None:
        self._observed_objects = tuple(observed_objects)
        self._planning_images = tuple(planning_images)
        self.closed = False
        self.received_plan: TaskPlan | None = None

    def scene_objects(self):
        return self._observed_objects

    def planning_images(self):
        return self._planning_images

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
                    ActionRequest(robots=("robot1",), skill="GoToObject", object_name="Laptop"),
                    ActionRequest(robots=("robot2",), skill="GoToObject", object_name="LightSwitch"),
                )
            ),
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="Laptop"),
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
        transition_count=1,
        max_transition_count=1,
    )
    plan = TaskPlan.sequential(
        ActionRequest(robots=("robot2",), skill="GoToObject", object_name="Laptop"),
        ActionRequest(robots=("robot2",), skill="SwitchOn", object_name="Laptop"),
        planner_name="fake-planner",
    )
    scene_objects = ({"name": "Laptop|0", "isToggled": True},)
    planner = FakePlanner(plan)
    executor = FakeExecutor(
        scene_objects,
        planning_images=(PlanningImage(data=b"png", agent_id=0, label="agent_0_egocentric"),),
    )
    runner = BenchmarkRunner(
        planner=planner,
        executor_factory=FakeExecutorFactory(executor),
    )

    result = runner.run_task(task)

    assert planner.calls[0]["task"] == "Turn on the laptop"
    assert planner.calls[0]["robots"] == ("robot1", "robot2")
    assert planner.calls[0]["planning_images"] == (
        PlanningImage(data=b"png", agent_id=0, label="agent_0_egocentric"),
    )
    assert result.execution.transition_count == 1
    assert result.metrics.goal_condition_recall == 1.0
    assert result.metrics.success_rate == 1.0
    assert result.planner_provider == "anthropic"
    assert result.planner_model == "claude-opus-4-7"
    assert result.planner_profile_variant == "symbolic"
    assert result.planner_usage == {"prompt_tokens": 42}
    assert executor.closed is True


def test_benchmark_runner_persists_clamped_robot_utilization() -> None:
    task = BenchmarkTask(
        floor_plan=15,
        task_index=4,
        instruction="Cook the potato and put it in the fridge",
        robot_ids=(24,),
        goal_states=(GoalState(name="Fridge", contains=("Potato",)),),
        transition_count=0,
        max_transition_count=1,
    )
    plan = TaskPlan.sequential(
        ActionRequest(robots=("robot1",), skill="GoToObject", object_name="Microwave"),
        ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="Microwave"),
        ActionRequest(robots=("robot1",), skill="GoToObject", object_name="Fridge"),
    )
    planner = FakePlanner(plan, uses_planning_images=False)
    executor = FakeExecutor(({"name": "Fridge|0", "receptacleObjectIds": ["Potato|0"]},))
    runner = BenchmarkRunner(
        planner=planner,
        executor_factory=FakeExecutorFactory(executor),
    )

    result = runner.run_task(task)

    assert result.execution.transition_count == 2
    assert result.metrics.robot_utilization == 0.0


def test_benchmark_runner_skips_image_capture_for_symbolic_planners() -> None:
    task = BenchmarkTask(
        floor_plan=303,
        task_index=1,
        instruction="Turn on the laptop",
        robot_ids=(24,),
        goal_states=(GoalState(name="Laptop", state="ON"),),
    )
    plan = TaskPlan.sequential(
        ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="Laptop"),
    )
    planner = FakePlanner(plan, uses_planning_images=False)
    executor = FakeExecutor(
        ({"name": "Laptop|0", "isToggled": True},),
        planning_images=(PlanningImage(data=b"png", agent_id=0, label="agent_0_egocentric"),),
    )
    runner = BenchmarkRunner(
        planner=planner,
        executor_factory=FakeExecutorFactory(executor),
    )

    runner.run_task(task)

    assert planner.calls[0]["planning_images"] == ()


def test_benchmark_summary_averages_task_metrics() -> None:
    task = BenchmarkTask(
        floor_plan=15,
        task_index=1,
        instruction="Make the kitchen dark",
        robot_ids=(24,),
        goal_states=(GoalState(name="LightSwitch", state="OFF"),),
    )
    off_plan = TaskPlan.sequential(
        ActionRequest(robots=("robot1",), skill="GoToObject", object_name="LightSwitch"),
        ActionRequest(robots=("robot1",), skill="SwitchOff", object_name="LightSwitch"),
    )
    off_runner = BenchmarkRunner(
        planner=FakePlanner(off_plan),
        executor_factory=FakeExecutorFactory(
            FakeExecutor(({"name": "LightSwitch|0", "isToggled": False},))
        ),
    )

    on_plan = TaskPlan.sequential(
        ActionRequest(robots=("robot1",), skill="GoToObject", object_name="LightSwitch"),
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


def test_benchmark_runner_records_failures_without_aborting_batch() -> None:
    task = BenchmarkTask(
        floor_plan=15,
        task_index=1,
        instruction="Turn on the lamp",
        robot_ids=(24,),
        goal_states=(GoalState(name="DeskLamp", state="ON"),),
    )
    other_task = BenchmarkTask(
        floor_plan=15,
        task_index=2,
        instruction="Turn on the laptop",
        robot_ids=(24,),
        goal_states=(GoalState(name="Laptop", state="ON"),),
    )
    plan = TaskPlan.sequential(
        ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="DeskLamp"),
    )
    planner = FakePlanner(plan, failure_calls={2})
    runner = BenchmarkRunner(
        planner=planner,
        executor_factory=FakeExecutorFactory(
            FakeExecutor(({"name": "DeskLamp|0", "isToggled": True},))
        ),
    )

    summary = runner.run_benchmark((task, other_task))

    assert summary.task_count == 2
    assert summary.failed_task_count == 1
    assert summary.task_runs[0].error_message is None
    assert summary.task_runs[1].error_message == "RuntimeError: boom"
    assert summary.task_runs[1].metrics.success_rate == 0.0
    assert summary.task_runs[1].execution.total_actions == 0
    assert summary.mean_metrics()["success_rate"] == 0.5

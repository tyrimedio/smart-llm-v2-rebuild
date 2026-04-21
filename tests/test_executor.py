import pytest

from smart_llm_v2.agents.executor import BaselineExecutor, ExecutionError
from smart_llm_v2.agents.plan import ActionRequest, PlanPhase, TaskPlan
from smart_llm_v2.agents.planner import PlanningImage
from smart_llm_v2.env.ai2thor_wrapper import ActionOutcome
from smart_llm_v2.robots import build_task_robot_team


class FakeEnvironment:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None, str | None]] = []
        self.objects = ({"name": "Laptop|0", "isToggled": True},)

    def navigate_to_object(self, *, agent_id: int, object_name: str) -> ActionOutcome:
        self.calls.append(("navigate", agent_id, object_name, None))
        return ActionOutcome(action="GoToObject", succeeded=True)

    def perform_action(
        self,
        *,
        agent_id: int,
        action_name: str,
        target_name: str | None = None,
    ) -> ActionOutcome:
        self.calls.append(("perform", agent_id, action_name, target_name))
        return ActionOutcome(action=action_name, succeeded=True)

    def scene_objects(self):
        return self.objects

    def planning_images(self, *, agent_ids):
        return tuple(
            PlanningImage(data=b"png", agent_id=agent_id, label=f"agent_{agent_id}_egocentric")
            for agent_id in agent_ids
        )

    def stop(self) -> None:
        return None


def test_executor_dispatches_navigation() -> None:
    executor = BaselineExecutor(
        environment=FakeEnvironment(),
        robots=build_task_robot_team((24,)),
    )

    result = executor.execute_step(
        ActionRequest(robots=("robot1",), skill="GoToObject", object_name="LightSwitch"),
    )

    assert result.succeeded is True
    assert executor.environment.calls == [("navigate", 0, "LightSwitch", None)]


def test_executor_maps_switch_on_to_ai2thor_toggle() -> None:
    environment = FakeEnvironment()
    executor = BaselineExecutor(
        environment=environment,
        robots=build_task_robot_team((24,)),
    )

    result = executor.execute_step(
        ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="Laptop"),
    )

    assert result.succeeded is True
    assert environment.calls == [("perform", 0, "ToggleObjectOn", "Laptop")]


def test_executor_uses_receptacle_for_put_object() -> None:
    environment = FakeEnvironment()
    executor = BaselineExecutor(
        environment=environment,
        robots=build_task_robot_team((25,)),
    )

    executor.execute_step(
        ActionRequest(
            robots=("robot1",),
            skill="PutObject",
            object_name="Mug",
            receptacle_name="CoffeeMachine",
        ),
    )

    assert environment.calls == [("perform", 0, "PutObject", "CoffeeMachine")]


def test_executor_rejects_missing_robot_skill() -> None:
    executor = BaselineExecutor(
        environment=FakeEnvironment(),
        robots=build_task_robot_team((25,)),
    )

    with pytest.raises(ExecutionError, match="does not have skill SwitchOn"):
        executor.execute_step(
            ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="Laptop"),
        )


def test_executor_run_plan_returns_execution_report() -> None:
    environment = FakeEnvironment()
    executor = BaselineExecutor(
        environment=environment,
        robots=build_task_robot_team((24,)),
    )
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="GoToObject", object_name="Laptop"),
                )
            ),
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="Laptop"),
                )
            ),
        )
    )

    report = executor.run_plan(plan)

    assert report.transition_count == 1
    assert report.total_actions == 2
    assert report.successful_actions == 2
    assert report.observed_objects == environment.objects


def test_executor_fans_out_team_actions() -> None:
    environment = FakeEnvironment()
    executor = BaselineExecutor(
        environment=environment,
        robots=build_task_robot_team((1, 2)),
    )

    result = executor.execute_step(
        ActionRequest(robots=("robot1", "robot2"), skill="GoToObject", object_name="Laptop"),
    )

    assert result.succeeded is True
    assert environment.calls == [
        ("navigate", 0, "Laptop", None),
        ("navigate", 1, "Laptop", None),
    ]


def test_executor_executes_team_pickup_for_each_robot() -> None:
    environment = FakeEnvironment()
    executor = BaselineExecutor(
        environment=environment,
        robots=build_task_robot_team((1, 2)),
    )

    result = executor.execute_step(
        ActionRequest(robots=("robot1", "robot2"), skill="PickupObject", object_name="Mug"),
    )

    assert result.succeeded is True
    assert environment.calls == [
        ("perform", 0, "PickupObject", "Mug"),
        ("perform", 1, "PickupObject", "Mug"),
    ]


def test_executor_executes_team_throw_for_each_robot() -> None:
    environment = FakeEnvironment()
    executor = BaselineExecutor(
        environment=environment,
        robots=build_task_robot_team((1, 2)),
    )

    result = executor.execute_step(
        ActionRequest(robots=("robot1", "robot2"), skill="ThrowObject", object_name="Fork"),
    )

    assert result.succeeded is True
    assert environment.calls == [
        ("perform", 0, "ThrowObject", "Fork"),
        ("perform", 1, "ThrowObject", "Fork"),
    ]

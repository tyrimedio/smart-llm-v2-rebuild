import pytest

from smart_llm_v2.agents.executor import ActionRequest, BaselineExecutor, ExecutionError
from smart_llm_v2.env.ai2thor_wrapper import ActionOutcome
from smart_llm_v2.robots import build_task_robot_team


class FakeEnvironment:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None, str | None]] = []

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


def test_executor_dispatches_navigation() -> None:
    executor = BaselineExecutor(
        environment=FakeEnvironment(),
        robots=build_task_robot_team((24,)),
    )

    result = executor.execute_step(
        ActionRequest(robot="robot1", skill="GoToObject", object_name="LightSwitch"),
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
        ActionRequest(robot="robot1", skill="SwitchOn", object_name="Laptop"),
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
            robot="robot1",
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
            ActionRequest(robot="robot1", skill="SwitchOn", object_name="Laptop"),
        )

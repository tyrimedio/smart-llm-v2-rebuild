import pytest

from smart_llm_v2.benchmark.metrics import (
    compute_metrics,
    goal_condition_recall,
    legacy_goal_condition_recall,
    robot_utilization,
)
from smart_llm_v2.benchmark.models import GoalState


def test_goal_condition_recall_counts_each_goal_once() -> None:
    goal_states = (
        GoalState(name="Drawer", contains=("Watch", "KeyChain")),
        GoalState(name="Laptop", state="ON"),
    )
    observed_objects = (
        {
            "name": "Drawer|0",
            "receptacleObjectIds": ["Watch|0", "KeyChain|0"],
        },
        {
            "name": "Laptop|0",
            "isToggled": True,
        },
    )

    assert goal_condition_recall(goal_states, observed_objects) == 1.0


def test_legacy_goal_condition_recall_matches_reference_behavior() -> None:
    goal_states = (GoalState(name="Drawer", contains=("Watch", "KeyChain")),)
    observed_objects = (
        {
            "name": "Drawer|0",
            "receptacleObjectIds": ["Watch|0", "KeyChain|0"],
        },
    )

    assert legacy_goal_condition_recall(goal_states, observed_objects) == 3.0


def test_robot_utilization_matches_reference_formula() -> None:
    assert robot_utilization(
        transition_count=2,
        transition_count_ground_truth=1,
        max_transition_count=4,
    ) == pytest.approx(2 / 3)


def test_robot_utilization_treats_zero_transition_exact_match_as_full_use() -> None:
    assert robot_utilization(
        transition_count=0,
        transition_count_ground_truth=0,
        max_transition_count=0,
    ) == 1.0


def test_robot_utilization_floors_overrun_at_zero() -> None:
    assert robot_utilization(
        transition_count=2,
        transition_count_ground_truth=0,
        max_transition_count=1,
    ) == 0.0


def test_compute_metrics_combines_components() -> None:
    metrics = compute_metrics(
        goal_states=(GoalState(name="LightSwitch", state="OFF"),),
        observed_objects=({"name": "LightSwitch|0", "isToggled": False},),
        transition_count=0,
        transition_count_ground_truth=0,
        max_transition_count=0,
        successful_actions=3,
        total_actions=3,
    )

    assert metrics.goal_condition_recall == 1.0
    assert metrics.task_completion_rate == 1.0
    assert metrics.robot_utilization == 1.0
    assert metrics.success_rate == 1.0
    assert metrics.executability == 1.0

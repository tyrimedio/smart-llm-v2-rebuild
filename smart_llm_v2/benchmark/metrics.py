from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from smart_llm_v2.benchmark.models import GoalState


@dataclass(frozen=True, slots=True)
class TaskMetrics:
    success_rate: float
    task_completion_rate: float
    goal_condition_recall: float
    robot_utilization: float
    executability: float


def compute_metrics(
    *,
    goal_states: Sequence[GoalState],
    observed_objects: Sequence[Mapping[str, object]],
    transition_count: int,
    transition_count_ground_truth: int,
    max_transition_count: int,
    successful_actions: int,
    total_actions: int,
    use_legacy_gcr: bool = False,
) -> TaskMetrics:
    gcr = (
        legacy_goal_condition_recall(goal_states, observed_objects)
        if use_legacy_gcr
        else goal_condition_recall(goal_states, observed_objects)
    )
    tcr = 1.0 if gcr == 1.0 else 0.0
    ru = robot_utilization(
        transition_count=transition_count,
        transition_count_ground_truth=transition_count_ground_truth,
        max_transition_count=max_transition_count,
    )
    sr = 1.0 if tcr == 1.0 and ru == 1.0 else 0.0

    return TaskMetrics(
        success_rate=sr,
        task_completion_rate=tcr,
        goal_condition_recall=gcr,
        robot_utilization=ru,
        executability=executability(
            successful_actions=successful_actions,
            total_actions=total_actions,
        ),
    )


def goal_condition_recall(
    goal_states: Sequence[GoalState],
    observed_objects: Sequence[Mapping[str, object]],
) -> float:
    if not goal_states:
        return 1.0

    satisfied = 0
    for goal_state in goal_states:
        if any(_goal_state_matches_object(goal_state, obj) for obj in observed_objects):
            satisfied += 1
    return satisfied / len(goal_states)


def legacy_goal_condition_recall(
    goal_states: Sequence[GoalState],
    observed_objects: Sequence[Mapping[str, object]],
) -> float:
    if not goal_states:
        return 1.0

    completed = 0.0
    for goal_state in goal_states:
        for observed_object in observed_objects:
            if _state_matches(goal_state.state, observed_object) and _name_matches(
                goal_state.name,
                observed_object,
            ):
                completed += 1

            if not goal_state.contains:
                continue

            if not _name_matches(goal_state.name, observed_object):
                continue

            receptacle_object_ids = observed_object.get("receptacleObjectIds") or ()
            for required_name in goal_state.contains:
                if any(
                    _canonicalize_name(required_name) in _canonicalize_name(str(object_id))
                    for object_id in receptacle_object_ids
                ):
                    completed += 1

    return completed / len(goal_states)


def robot_utilization(
    *,
    transition_count: int,
    transition_count_ground_truth: int,
    max_transition_count: int,
) -> float:
    reference_max = max_transition_count + 1
    reference_ground_truth = transition_count_ground_truth + 1

    if (
        reference_max == reference_ground_truth
        and reference_ground_truth == transition_count
    ):
        return 1.0

    if reference_max == reference_ground_truth:
        return 0.0

    return (reference_max - transition_count) / (
        reference_max - reference_ground_truth
    )


def executability(*, successful_actions: int, total_actions: int) -> float:
    if total_actions == 0:
        return 1.0
    return successful_actions / total_actions


def _goal_state_matches_object(
    goal_state: GoalState,
    observed_object: Mapping[str, object],
) -> bool:
    if not _name_matches(goal_state.name, observed_object):
        return False

    if not _state_matches(goal_state.state, observed_object):
        return False

    receptacle_object_ids = observed_object.get("receptacleObjectIds") or ()
    for required_name in goal_state.contains:
        if not any(
            _canonicalize_name(required_name) in _canonicalize_name(str(object_id))
            for object_id in receptacle_object_ids
        ):
            return False

    return True


def _name_matches(target_name: str, observed_object: Mapping[str, object]) -> bool:
    candidate = (
        observed_object.get("name")
        or observed_object.get("objectId")
        or observed_object.get("objectType")
        or ""
    )
    return _canonicalize_name(target_name) in _canonicalize_name(str(candidate))


def _state_matches(state: str | None, observed_object: Mapping[str, object]) -> bool:
    if state is None:
        return True

    if state == "SLICED":
        return bool(observed_object.get("isSliced"))
    if state == "OFF":
        return not bool(observed_object.get("isToggled"))
    if state == "ON":
        return bool(observed_object.get("isToggled"))
    if state == "HOT":
        return _canonicalize_name(str(observed_object.get("temperature", ""))) == "hot"
    if state == "COOKED":
        return bool(observed_object.get("isCooked"))
    if state == "OPENED":
        return bool(observed_object.get("isOpen"))
    if state == "CLOSED":
        return not bool(observed_object.get("isOpen"))
    if state == "PICKED":
        return bool(observed_object.get("isPickedUp"))
    if state == "BROKEN":
        return bool(observed_object.get("isBroken"))
    return False


def _canonicalize_name(value: str) -> str:
    return value.strip().rstrip(",.").casefold()

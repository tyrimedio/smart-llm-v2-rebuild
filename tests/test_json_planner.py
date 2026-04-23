from __future__ import annotations

import pytest

from smart_llm_v2.agents.json_planner import (
    DEFAULT_SYSTEM_MESSAGE,
    JsonPlanner,
    JsonPlanningResult,
    JsonPlanValidationError,
    JsonPlannerRequest,
    JsonTaskPlan,
    build_planning_context,
    task_plan_json_schema,
)
from smart_llm_v2.agents.model_profiles import resolve_model_profile
from smart_llm_v2.agents.plan import ActionRequest
from smart_llm_v2.agents.planner import PlanningImage
from smart_llm_v2.benchmark.models import BenchmarkTask, GoalState
from smart_llm_v2.robots import build_task_robot_team


class FakeJsonClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[JsonPlannerRequest] = []

    def complete(self, request: JsonPlannerRequest):
        self.requests.append(request)
        return JsonPlanningResult(
            payload=self.payload,
            provider=request.profile.provider.value,
            model=request.profile.model,
            usage={"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        )


def test_default_system_message_prioritizes_success_before_robot_count() -> None:
    assert "minimum number of robots" not in DEFAULT_SYSTEM_MESSAGE
    assert "Maximize successful task completion" in DEFAULT_SYSTEM_MESSAGE
    assert "Prefer fewer robots only as a tie-break" in DEFAULT_SYSTEM_MESSAGE


def test_json_task_plan_converts_to_task_plan() -> None:
    payload = {
        "notes": "Laptop needs the switch turned on first.",
        "phases": [
            {
                "label": "prepare",
                "actions": [
                    {
                        "robots": ["robot1", "robot2"],
                        "skill": "GoToObject",
                        "object_name": "LightSwitch",
                    }
                ],
            },
            {
                "actions": [
                    {
                        "robots": ["robot1"],
                        "skill": "SwitchOn",
                        "object_name": "Laptop",
                    }
                ]
            },
        ],
    }

    json_plan = JsonTaskPlan.from_mapping(
        payload,
        valid_robot_names=frozenset({"robot1", "robot2"}),
        valid_skills=frozenset({"GoToObject", "SwitchOn"}),
        robot_skills_by_name={
            "robot1": frozenset({"GoToObject", "SwitchOn"}),
            "robot2": frozenset({"GoToObject", "SwitchOn"}),
        },
    )
    plan = json_plan.to_task_plan(planner_name="json-test")

    assert plan.planner_name == "json-test"
    assert plan.notes == "Laptop needs the switch turned on first."
    assert plan.transition_count == 1
    assert plan.phases[0].label == "prepare"
    assert plan.phases[0].actions == (
        ActionRequest(robots=("robot1", "robot2"), skill="GoToObject", object_name="LightSwitch"),
    )


def test_json_task_plan_rejects_unknown_robot_names() -> None:
    payload = {
        "phases": [
            {
                "actions": [
                    {
                        "robots": ["robot9"],
                        "skill": "GoToObject",
                        "object_name": "Laptop",
                    }
                ]
            }
        ]
    }

    with pytest.raises(JsonPlanValidationError, match="unknown robots"):
        JsonTaskPlan.from_mapping(
            payload,
            valid_robot_names=frozenset({"robot1", "robot2"}),
            valid_skills=frozenset({"GoToObject"}),
            robot_skills_by_name={
                "robot1": frozenset({"GoToObject"}),
                "robot2": frozenset({"GoToObject"}),
            },
        )


def test_json_task_plan_rejects_missing_put_object_receptacle() -> None:
    payload = {
        "phases": [
            {
                "actions": [
                    {
                        "robots": ["robot1"],
                        "skill": "PutObject",
                        "object_name": "Mug",
                    }
                ]
            }
        ]
    }

    with pytest.raises(JsonPlanValidationError, match="requires receptacle_name"):
        JsonTaskPlan.from_mapping(
            payload,
            valid_robot_names=frozenset({"robot1"}),
            valid_skills=frozenset({"PutObject"}),
            robot_skills_by_name={"robot1": frozenset({"PutObject"})},
        )


def test_json_planner_rejects_action_for_robot_without_required_skill() -> None:
    task = BenchmarkTask(
        floor_plan=1,
        task_index=2,
        instruction="Put the mug in the sink",
        robot_ids=(24,),
    )
    planner = JsonPlanner(
        client=FakeJsonClient(
            {
                "phases": [
                    {
                        "actions": [
                            {
                                "robots": ["robot1"],
                                "skill": "PutObject",
                                "object_name": "Mug",
                                "receptacle_name": "Sink",
                            }
                        ]
                    }
                ]
            }
        ),
        profile=resolve_model_profile(provider="anthropic", variant="symbolic"),
    )

    with pytest.raises(JsonPlanValidationError, match="cannot execute"):
        planner.build_plan(
            task=task,
            robots=build_task_robot_team(task.robot_ids),
            scene_objects=(),
        )


def test_json_planner_rejects_pickup_when_team_capacity_is_too_low() -> None:
    task = BenchmarkTask(
        floor_plan=201,
        task_index=1,
        instruction="Pick up the book",
        robot_ids=(8,),
    )
    planner = JsonPlanner(
        client=FakeJsonClient(
            {
                "phases": [
                    {
                        "actions": [
                            {
                                "robots": ["robot1"],
                                "skill": "PickupObject",
                                "object_name": "Book",
                            }
                        ]
                    }
                ]
            }
        ),
        profile=resolve_model_profile(provider="anthropic", variant="symbolic"),
    )

    with pytest.raises(JsonPlanValidationError, match="combined mass capacity"):
        planner.build_plan(
            task=task,
            robots=build_task_robot_team(task.robot_ids),
            scene_objects=({"objectType": "Book", "mass": 6.0},),
        )


def test_json_planner_allows_team_pickup_when_combined_capacity_is_enough() -> None:
    task = BenchmarkTask(
        floor_plan=201,
        task_index=2,
        instruction="Pick up the book together",
        robot_ids=(8, 9),
    )
    planner = JsonPlanner(
        client=FakeJsonClient(
            {
                "phases": [
                    {
                        "actions": [
                            {
                                "robots": ["robot1", "robot2"],
                                "skill": "PickupObject",
                                "object_name": "Book",
                            }
                        ]
                    }
                ]
            }
        ),
        profile=resolve_model_profile(provider="anthropic", variant="symbolic"),
    )

    plan = planner.build_plan(
        task=task,
        robots=build_task_robot_team(task.robot_ids),
        scene_objects=({"objectType": "Book", "mass": 5.0},),
    ).plan

    assert plan.phases[0].actions == (
        ActionRequest(robots=("robot1", "robot2"), skill="PickupObject", object_name="Book"),
    )


def test_json_planner_builds_request_context_and_validates_response() -> None:
    task = BenchmarkTask(
        floor_plan=303,
        task_index=1,
        instruction="Turn on the laptop",
        robot_ids=(24, 1),
        goal_states=(GoalState(name="Laptop", state="ON"),),
    )
    robots = build_task_robot_team(task.robot_ids)
    scene_objects = (
        {"objectType": "Laptop", "mass": 2.0, "isToggled": False},
        {"objectType": "LightSwitch", "mass": 1.0, "isToggled": False},
    )
    planner = JsonPlanner(
        client=FakeJsonClient(
            {
                "notes": "robot1 reaches the switch while robot2 handles the laptop",
                "phases": [
                    {
                        "actions": [
                            {
                                "robots": ["robot1"],
                                "skill": "GoToObject",
                                "object_name": "LightSwitch",
                            },
                            {
                                "robots": ["robot2"],
                                "skill": "GoToObject",
                                "object_name": "Laptop",
                            },
                        ]
                    },
                    {
                        "actions": [
                            {
                                "robots": ["robot2"],
                                "skill": "SwitchOn",
                                "object_name": "Laptop",
                            }
                        ]
                    },
                ],
            }
        ),
        profile=resolve_model_profile(provider="anthropic", variant="multimodal"),
        planner_name="json-foundation",
    )

    planning = planner.build_plan(
        task=task,
        robots=robots,
        scene_objects=scene_objects,
        planning_images=(
            PlanningImage(data=b"png-bytes", agent_id=0, label="agent_0_egocentric"),
        ),
    )
    plan = planning.plan
    request = planner.client.requests[0]

    assert plan.planner_name == "json-foundation"
    assert plan.transition_count == 1
    assert request.context["task"]["instruction"] == "Turn on the laptop"
    assert request.context["robots"][0]["name"] == "robot1"
    assert request.context["scene_objects"][0]["name"] == "Laptop"
    assert request.context["observation_images"] == [
        {
            "agent_id": 0,
            "label": "agent_0_egocentric",
            "media_type": "image/png",
        }
    ]
    assert request.response_schema["properties"]["phases"]["type"] == "array"
    assert planning.provider == "anthropic"
    assert planning.model == "claude-opus-4-7"
    assert planning.usage == {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}


def test_json_planner_drops_images_for_symbolic_profiles() -> None:
    task = BenchmarkTask(
        floor_plan=303,
        task_index=1,
        instruction="Turn on the laptop",
        robot_ids=(24,),
    )
    planner = JsonPlanner(
        client=FakeJsonClient(
            {
                "phases": [
                    {
                        "actions": [
                            {
                                "robots": ["robot1"],
                                "skill": "GoToObject",
                                "object_name": "Laptop",
                            }
                        ]
                    }
                ]
            }
        ),
        profile=resolve_model_profile(provider="anthropic", variant="symbolic"),
    )

    planner.build_plan(
        task=task,
        robots=build_task_robot_team(task.robot_ids),
        scene_objects=(),
        planning_images=(PlanningImage(data=b"png-bytes", agent_id=0, label="agent_0_egocentric"),),
    )

    request = planner.client.requests[0]
    assert request.images == ()
    assert "observation_images" not in request.context


def test_build_planning_context_includes_goal_states_and_action_schema() -> None:
    task = BenchmarkTask(
        floor_plan=1,
        task_index=2,
        instruction="Put the mug in the sink",
        robot_ids=(25,),
        goal_states=(GoalState(name="Sink", contains=("Mug",)),),
    )
    robots = build_task_robot_team(task.robot_ids)
    context = build_planning_context(
        task=task,
        robots=robots,
        scene_objects=({"name": "Mug|0", "mass": 1.0},),
        images=(PlanningImage(data=b"x", agent_id=0, label="robot1_view"),),
    )

    assert context["task"]["goal_states"] == [
        {"name": "Sink", "state": None, "contains": ["Mug"]}
    ]
    assert context["observation_images"] == [
        {"agent_id": 0, "label": "robot1_view", "media_type": "image/png"}
    ]
    assert context["action_schema"] == task_plan_json_schema()


@pytest.mark.parametrize(
    ("provider", "variant"),
    [
        ("anthropic", "symbolic"),
        ("openai", "symbolic"),
        ("kimi", "symbolic"),
    ],
)
def test_same_json_payload_maps_to_same_task_plan_across_providers(
    provider: str,
    variant: str,
) -> None:
    task = BenchmarkTask(
        floor_plan=303,
        task_index=1,
        instruction="Turn on the laptop",
        robot_ids=(24,),
        goal_states=(GoalState(name="Laptop", state="ON"),),
    )
    profile = resolve_model_profile(provider=provider, variant=variant)
    planner = JsonPlanner(
        client=FakeJsonClient(
            {
                "phases": [
                    {
                        "actions": [
                            {
                                "robots": ["robot1"],
                                "skill": "GoToObject",
                                "object_name": "Laptop",
                            }
                        ]
                    },
                    {
                        "actions": [
                            {
                                "robots": ["robot1"],
                                "skill": "SwitchOn",
                                "object_name": "Laptop",
                            }
                        ]
                    },
                ]
            }
        ),
        profile=profile,
    )

    planning = planner.build_plan(
        task=task,
        robots=build_task_robot_team(task.robot_ids),
        scene_objects=(),
    )

    assert planning.plan.phases[0].actions == (
        ActionRequest(robots=("robot1",), skill="GoToObject", object_name="Laptop"),
    )
    assert planning.plan.phases[1].actions == (
        ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="Laptop"),
    )

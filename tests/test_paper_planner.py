from __future__ import annotations

from smart_llm_v2.agents.paper_planner import (
    PaperPlannerArtifacts,
    PaperPromptAssets,
    PaperStagedPlanner,
    PromptRequest,
    serialize_prompt_objects,
    serialize_prompt_robots,
)
from smart_llm_v2.agents.plan import ActionRequest, TaskPlan
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import build_task_robot_team


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[PromptRequest] = []

    def complete(self, request: PromptRequest) -> str:
        self.requests.append(request)
        return self.responses[len(self.requests) - 1]


class FakeParser:
    def __init__(self) -> None:
        self.calls: list[PaperPlannerArtifacts] = []

    def parse(self, *, task, robots, artifacts):
        self.calls.append(artifacts)
        return TaskPlan.sequential(
            ActionRequest(robot="robot1", skill="GoToObject", object_name="Laptop"),
            planner_name="fake-paper-parser",
        )


def test_serialize_prompt_robots_preserves_skill_and_mass_metadata() -> None:
    robots = build_task_robot_team((24, 25))

    serialized = serialize_prompt_robots(robots)

    assert serialized[0]["name"] == "robot1"
    assert serialized[0]["no_skills"] == 3
    assert serialized[0]["mass"] == 100.0
    assert "SwitchOn" in serialized[0]["skills"]


def test_serialize_prompt_objects_prefers_object_type_and_mass() -> None:
    prompt_objects = serialize_prompt_objects(
        (
            {"objectType": "Laptop", "name": "Laptop|1", "mass": 2.5},
            {"name": "LightSwitch|2", "mass": 1.0},
        )
    )

    assert "{'name': 'Laptop', 'mass': 2.5}" in prompt_objects
    assert "{'name': 'LightSwitch', 'mass': 1.0}" in prompt_objects


def test_paper_staged_planner_builds_three_stage_requests_and_parses_result() -> None:
    planner = PaperStagedPlanner(
        client=FakeClient(
            [
                "# decomposition output",
                "# allocation solution output",
                "# code solution output",
            ]
        ),
        parser=FakeParser(),
        prompt_assets=PaperPromptAssets(
            decomposition_examples="# decompose examples",
            allocation_solution_examples="# allocation solution examples",
            allocation_code_examples="# allocation code examples",
        ),
    )
    task = BenchmarkTask(
        floor_plan=303,
        task_index=1,
        instruction="Turn on the laptop",
        robot_ids=(24, 1),
    )
    robots = build_task_robot_team(task.robot_ids)
    scene_objects = (
        {"objectType": "Laptop", "mass": 2.0},
        {"objectType": "LightSwitch", "mass": 1.0},
    )

    plan = planner.build_plan(task=task, robots=robots, scene_objects=scene_objects)

    assert plan.planner_name == "fake-paper-parser"
    assert [request.stage for request in planner.client.requests] == [
        "decomposition",
        "allocation_solution",
        "code_solution",
    ]
    assert "# Task Description: Turn on the laptop" in planner.client.requests[0].prompt
    assert "robots = " in planner.client.requests[1].prompt
    assert "# allocation solution output" in planner.client.requests[2].prompt

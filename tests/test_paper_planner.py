from __future__ import annotations

import pytest

from smart_llm_v2.agents.paper_planner import (
    AstPaperPlanParser,
    PaperPlanParseError,
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
            ActionRequest(robots=("robot1",), skill="GoToObject", object_name="Laptop"),
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

    planning = planner.build_plan(task=task, robots=robots, scene_objects=scene_objects)
    plan = planning.plan

    assert plan.planner_name == "fake-paper-parser"
    assert planning.provider == "legacy"
    assert planning.profile_variant == "legacy"
    assert [request.stage for request in planner.client.requests] == [
        "decomposition",
        "allocation_solution",
        "code_solution",
    ]
    assert "# Task Description: Turn on the laptop" in planner.client.requests[0].prompt
    assert "robots = " in planner.client.requests[1].prompt
    assert "# allocation solution output" in planner.client.requests[2].prompt


def test_paper_prompt_assets_fix_known_bad_example_signature() -> None:
    prompt_assets = PaperPromptAssets.load()

    assert "def throw_fork_in_trash(robot_list):" in prompt_assets.allocation_code_examples
    assert "def throw_fork_in_trash():" not in prompt_assets.allocation_code_examples


def test_agents_package_root_exports_paper_planner_symbols() -> None:
    from smart_llm_v2.agents import AstPaperPlanParser, PaperPromptAssets, PaperStagedPlanner

    assert AstPaperPlanParser is not None
    assert PaperPromptAssets is not None
    assert PaperStagedPlanner is not None


def test_ast_paper_parser_builds_single_phase_for_one_subtask() -> None:
    parser = AstPaperPlanParser()
    robots = build_task_robot_team((24, 25))
    task = BenchmarkTask(
        floor_plan=1,
        task_index=1,
        instruction="Wash the fork",
        robot_ids=(24, 25),
    )
    artifacts = PaperPlannerArtifacts(
        decomposition="",
        allocation_solution="# Robot 2 is assigned",
        code_solution="""
def wash_fork(robot_list):
    GoToObject(robot_list[0], 'Fork')
    PickupObject(robot_list[0], 'Fork')
    GoToObject(robot_list[0], 'Sink')
    PutObject(robot_list[0], 'Fork', 'Sink')

wash_fork([robots[1]])
""",
    )

    plan = parser.parse(task=task, robots=robots, artifacts=artifacts)

    assert plan.planner_name == "paper-ast-parser"
    assert plan.transition_count == 0
    assert plan.phases[0].actions == (
        ActionRequest(robots=("robot2",), skill="GoToObject", object_name="Fork"),
        ActionRequest(robots=("robot2",), skill="PickupObject", object_name="Fork"),
        ActionRequest(robots=("robot2",), skill="GoToObject", object_name="Sink"),
        ActionRequest(
            robots=("robot2",),
            skill="PutObject",
            object_name="Fork",
            receptacle_name="Sink",
        ),
    )


def test_ast_paper_parser_resolves_indexed_team_members_inside_subtask() -> None:
    parser = AstPaperPlanParser()
    robots = build_task_robot_team((26, 24, 25))
    task = BenchmarkTask(
        floor_plan=1,
        task_index=2,
        instruction="Slice the potato",
        robot_ids=(26, 24, 25),
    )
    artifacts = PaperPlannerArtifacts(
        decomposition="",
        allocation_solution="# Team of Robots 1 and 3 is assigned",
        code_solution="""
def slice_potato(robot_list):
    GoToObject(robot_list[1], 'Knife')
    PickupObject(robot_list[1], 'Knife')
    GoToObject(robot_list[1], 'Potato')
    SliceObject(robot_list[0], 'Potato')

slice_potato([robots[0], robots[2]])
""",
    )

    plan = parser.parse(task=task, robots=robots, artifacts=artifacts)

    assert plan.transition_count == 0
    assert plan.phases[0].actions == (
        ActionRequest(robots=("robot3",), skill="GoToObject", object_name="Knife"),
        ActionRequest(robots=("robot3",), skill="PickupObject", object_name="Knife"),
        ActionRequest(robots=("robot3",), skill="GoToObject", object_name="Potato"),
        ActionRequest(robots=("robot1",), skill="SliceObject", object_name="Potato"),
    )


def test_ast_paper_parser_keeps_sequential_team_subtasks_in_separate_phases() -> None:
    parser = AstPaperPlanParser()
    robots = build_task_robot_team((1, 2, 3))
    task = BenchmarkTask(
        floor_plan=1,
        task_index=3,
        instruction="Throw the fork in the trash",
        robot_ids=(1, 2, 3),
    )
    artifacts = PaperPlannerArtifacts(
        decomposition="",
        allocation_solution="# Team of Robots 1 and 2, then Robots 1 and 3",
        code_solution="""
def pick_up_fork(robot_list):
    GoToObject(robot_list, 'Fork')
    PickupObject(robot_list, 'Fork')

def throw_fork_in_trash(robot_list):
    GoToObject(robot_list, 'GarbageCan')
    ThrowObject(robot_list, 'Fork', 'GarbageCan')

pick_up_fork([robots[0], robots[1]])
throw_fork_in_trash([robots[0], robots[2]])
""",
    )

    plan = parser.parse(task=task, robots=robots, artifacts=artifacts)

    assert len(plan.phases) == 2
    assert plan.transition_count == 1
    assert plan.phases[0].actions == (
        ActionRequest(robots=("robot1", "robot2"), skill="GoToObject", object_name="Fork"),
        ActionRequest(robots=("robot1", "robot2"), skill="PickupObject", object_name="Fork"),
    )
    assert plan.phases[1].actions == (
        ActionRequest(robots=("robot1", "robot3"), skill="GoToObject", object_name="GarbageCan"),
        ActionRequest(
            robots=("robot1", "robot3"),
            skill="ThrowObject",
            object_name="Fork",
            receptacle_name="GarbageCan",
        ),
    )


def test_ast_paper_parser_keeps_main_thread_actions_concurrent_until_join() -> None:
    parser = AstPaperPlanParser()
    robots = build_task_robot_team((24, 25))
    task = BenchmarkTask(
        floor_plan=1,
        task_index=4,
        instruction="Pick up the mug while another robot moves to the switch",
        robot_ids=(24, 25),
    )
    artifacts = PaperPlannerArtifacts(
        decomposition="",
        allocation_solution="# concurrent main-thread action",
        code_solution="""
def pick_up_mug(robot_list):
    PickupObject(robot_list[0], 'Mug')

t1 = threading.Thread(target=pick_up_mug, args=([robots[0]],))
t1.start()
GoToObject(robots[1], 'LightSwitch')
t1.join()
""",
    )

    plan = parser.parse(task=task, robots=robots, artifacts=artifacts)

    assert len(plan.phases) == 1
    assert plan.transition_count == 0
    assert plan.phases[0].actions == (
        ActionRequest(robots=("robot1",), skill="PickupObject", object_name="Mug"),
        ActionRequest(robots=("robot2",), skill="GoToObject", object_name="LightSwitch"),
    )


def test_ast_paper_parser_rejects_malformed_function_calls() -> None:
    parser = AstPaperPlanParser()
    robots = build_task_robot_team((1, 2, 3))
    task = BenchmarkTask(
        floor_plan=1,
        task_index=5,
        instruction="Throw the fork in the trash",
        robot_ids=(1, 2, 3),
    )
    artifacts = PaperPlannerArtifacts(
        decomposition="",
        allocation_solution="# malformed output",
        code_solution="""
def throw_fork_in_trash():
    GoToObject(robot_list, 'GarbageCan')
    ThrowObject(robot_list, 'Fork', 'GarbageCan')

throw_fork_in_trash([robots[0], robots[2]])
""",
    )

    with pytest.raises(PaperPlanParseError, match="expected 0 arguments, got 1"):
        parser.parse(task=task, robots=robots, artifacts=artifacts)

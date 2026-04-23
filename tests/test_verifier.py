from __future__ import annotations

from smart_llm_v2.agents.paper_planner import AstPaperPlanParser, PaperPlannerArtifacts
from smart_llm_v2.agents.plan import ActionRequest, PlanPhase, PlanSubTask, TaskPlan
from smart_llm_v2.agents.verifier import (
    PlanVerifier,
    SemanticVerificationResult,
    VerificationIssue,
    semantic_verification_json_schema,
    semantic_verification_result_from_mapping,
)
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import build_task_robot_team


class FakeSemanticClient:
    def __init__(self, result: SemanticVerificationResult) -> None:
        self.result = result
        self.requests = []

    def review(self, request):
        self.requests.append(request)
        return self.result


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        floor_plan=1,
        task_index=1,
        instruction="Put the mug in the fridge",
        robot_ids=(24,),
    )


def test_verifier_allows_same_robot_reuse_within_a_phase() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="GoToObject", object_name="Fridge"),
                    ActionRequest(robots=("robot1",), skill="OpenObject", object_name="Fridge"),
                )
            ),
        )
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1,)),
        scene_objects=(),
        plan=plan,
    )

    assert result.passed is True


def test_verifier_allows_ordered_reuse_of_one_resource_within_a_phase() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="OpenObject", object_name="Fridge"),
                    ActionRequest(robots=("robot1",), skill="CloseObject", object_name="Fridge"),
                )
            ),
        )
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1,)),
        scene_objects=({"name": "Fridge|0", "isOpen": False},),
        plan=plan,
    )

    assert result.passed is True


def test_verifier_rejects_same_robot_assigned_to_parallel_subtasks() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                subtasks=(
                    PlanSubTask(
                        actions=(
                            ActionRequest(
                                robots=("robot1",),
                                skill="GoToObject",
                                object_name="Fridge",
                            ),
                        )
                    ),
                    PlanSubTask(
                        actions=(
                            ActionRequest(
                                robots=("robot1",),
                                skill="GoToObject",
                                object_name="Mug",
                            ),
                        )
                    ),
                )
            ),
        )
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1,)),
        scene_objects=(),
        plan=plan,
    )

    assert result.passed is False
    assert result.issues[0].code == "same_phase_robot_conflict"


def test_verifier_rejects_state_dependency_between_parallel_subtasks() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot2",), skill="PickupObject", object_name="Mug"),
                )
            ),
            PlanPhase(
                subtasks=(
                    PlanSubTask(
                        actions=(
                            ActionRequest(
                                robots=("robot1",),
                                skill="OpenObject",
                                object_name="Fridge",
                            ),
                        )
                    ),
                    PlanSubTask(
                        actions=(
                            ActionRequest(
                                robots=("robot2",),
                                skill="PutObject",
                                object_name="Mug",
                                receptacle_name="Fridge",
                            ),
                        )
                    ),
                )
            ),
        )
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1, 2)),
        scene_objects=({"name": "Fridge|0", "isOpen": False},),
        plan=plan,
    )

    assert result.passed is False
    assert "closed_receptacle" in {issue.code for issue in result.issues}


def test_verifier_allows_independent_parallel_subtasks() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                subtasks=(
                    PlanSubTask(
                        actions=(
                            ActionRequest(
                                robots=("robot1",),
                                skill="SwitchOn",
                                object_name="LightSwitch",
                            ),
                        )
                    ),
                    PlanSubTask(
                        actions=(
                            ActionRequest(
                                robots=("robot2",),
                                skill="PickupObject",
                                object_name="Mug",
                            ),
                        )
                    ),
                )
            ),
        )
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1, 2)),
        scene_objects=(
            {"name": "LightSwitch|0", "objectType": "LightSwitch"},
            {"name": "Mug|0", "objectType": "Mug"},
        ),
        plan=plan,
    )

    assert result.passed is True


def test_verifier_accepts_paper_plan_that_hands_off_team_members_between_phases() -> None:
    parser = AstPaperPlanParser()
    task = BenchmarkTask(
        floor_plan=1,
        task_index=3,
        instruction="Throw the fork in the trash",
        robot_ids=(1, 2, 3),
    )
    robots = build_task_robot_team(task.robot_ids)
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

    result = PlanVerifier().verify(
        task=task,
        robots=robots,
        scene_objects=(),
        plan=plan,
    )

    assert result.passed is True


def test_verifier_tracks_pickup_state_within_a_phase() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="PickupObject", object_name="Mug"),
                    ActionRequest(robots=("robot1",), skill="ThrowObject", object_name="Mug"),
                )
            ),
        )
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1,)),
        scene_objects=(),
        plan=plan,
    )

    assert result.passed is True


def test_verifier_allows_matching_pickups_when_scene_has_multiple_instances() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="PickupObject", object_name="Mug"),
                )
            ),
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot2",), skill="PickupObject", object_name="Mug"),
                )
            ),
        )
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1, 2)),
        scene_objects=(
            {"name": "Mug|1", "objectId": "Mug|1", "objectType": "Mug"},
            {"name": "Mug|2", "objectId": "Mug|2", "objectType": "Mug"},
        ),
        plan=plan,
    )

    assert result.passed is True


def test_verifier_tracks_receptacle_state_within_a_phase() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="PickupObject", object_name="Mug"),
                )
            ),
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="OpenObject", object_name="Fridge"),
                    ActionRequest(
                        robots=("robot1",),
                        skill="PutObject",
                        object_name="Mug",
                        receptacle_name="Fridge",
                    ),
                )
            ),
        )
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1,)),
        scene_objects=({"name": "Fridge|0", "isOpen": False},),
        plan=plan,
    )

    assert result.passed is True


def test_verifier_ignores_scene_order_when_matching_receptacles_have_mixed_open_state() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="PickupObject", object_name="Mug"),
                )
            ),
            PlanPhase(
                actions=(
                    ActionRequest(
                        robots=("robot1",),
                        skill="PutObject",
                        object_name="Mug",
                        receptacle_name="Drawer",
                    ),
                )
            ),
        )
    )

    for scene_objects in (
        (
            {
                "name": "Drawer|closed",
                "objectId": "Drawer|closed",
                "objectType": "Drawer",
                "isOpen": False,
            },
            {
                "name": "Drawer|open",
                "objectId": "Drawer|open",
                "objectType": "Drawer",
                "isOpen": True,
            },
        ),
        (
            {
                "name": "Drawer|open",
                "objectId": "Drawer|open",
                "objectType": "Drawer",
                "isOpen": True,
            },
            {
                "name": "Drawer|closed",
                "objectId": "Drawer|closed",
                "objectType": "Drawer",
                "isOpen": False,
            },
        ),
    ):
        result = PlanVerifier().verify(
            task=_task(),
            robots=build_task_robot_team((1,)),
            scene_objects=scene_objects,
            plan=plan,
        )

        assert result.passed is True


def test_verifier_clears_dropped_teammates_after_overlapping_handoff() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1", "robot2"), skill="PickupObject", object_name="Fork"),
                )
            ),
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1", "robot3"), skill="ThrowObject", object_name="Fork"),
                )
            ),
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot2",), skill="PickupObject", object_name="Mug"),
                )
            ),
        )
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1, 2, 3)),
        scene_objects=(),
        plan=plan,
    )

    assert result.passed is True


def test_verifier_rejects_unknown_robot_names() -> None:
    plan = TaskPlan.sequential(
        ActionRequest(robots=("robot2",), skill="GoToObject", object_name="Fridge")
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((24,)),
        scene_objects=(),
        plan=plan,
    )

    assert result.passed is False
    assert result.issues[0].code == "unknown_robot"


def test_verifier_rejects_missing_required_action_arguments() -> None:
    plan = TaskPlan.sequential(
        ActionRequest(robots=("robot1",), skill="GoToObject")
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1,)),
        scene_objects=(),
        plan=plan,
    )

    assert result.passed is False
    assert result.issues[0].code == "missing_action_argument"
    assert "object_name" in result.issues[0].message


def test_verifier_rejects_unsupported_robot_skills() -> None:
    plan = TaskPlan.sequential(
        ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="LightSwitch")
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((11,)),
        scene_objects=(),
        plan=plan,
    )

    assert result.passed is False
    assert result.issues[0].code == "unsupported_skill"


def test_verifier_rejects_pickup_when_team_capacity_is_too_low() -> None:
    plan = TaskPlan.sequential(
        ActionRequest(robots=("robot1",), skill="PickupObject", object_name="Book")
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((10,)),
        scene_objects=({"objectType": "Book", "mass": 6.0},),
        plan=plan,
    )

    assert result.passed is False
    assert result.issues[0].code == "insufficient_mass_capacity"
    assert "combined mass capacity" in result.issues[0].message


def test_verifier_rejects_put_before_pickup_and_closed_receptacle() -> None:
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(
                        robots=("robot1",),
                        skill="PutObject",
                        object_name="Mug",
                        receptacle_name="Fridge",
                    ),
                )
            ),
        )
    )

    result = PlanVerifier().verify(
        task=_task(),
        robots=build_task_robot_team((1,)),
        scene_objects=({"name": "Fridge|0", "isOpen": False},),
        plan=plan,
    )

    assert result.passed is False
    assert {issue.code for issue in result.issues} == {
        "closed_receptacle",
        "missing_pickup_before_handoff",
    }


def test_verifier_skips_semantic_review_when_deterministic_checks_fail() -> None:
    semantic_client = FakeSemanticClient(SemanticVerificationResult())
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(
                        robots=("robot1",),
                        skill="PutObject",
                        object_name="Mug",
                        receptacle_name="Fridge",
                    ),
                )
            ),
        )
    )

    result = PlanVerifier(semantic_client=semantic_client).verify(
        task=_task(),
        robots=build_task_robot_team((1,)),
        scene_objects=(),
        plan=plan,
    )

    assert result.passed is False
    assert result.semantic_checked is False
    assert semantic_client.requests == []


def test_verifier_runs_semantic_review_after_deterministic_checks_pass() -> None:
    semantic_client = FakeSemanticClient(
        SemanticVerificationResult(
            issues=(
                VerificationIssue(
                    code="semantic_gap",
                    message="The plan never reaches the fridge before PutObject.",
                    source="semantic",
                ),
            ),
            provider="openai",
            model="gpt-5.4",
            usage={"reasoning_tokens": 12},
        )
    )
    plan = TaskPlan(
        phases=(
            PlanPhase(
                actions=(
                    ActionRequest(robots=("robot1",), skill="PickupObject", object_name="Mug"),
                )
            ),
            PlanPhase(
                actions=(
                    ActionRequest(
                        robots=("robot1",),
                        skill="PutObject",
                        object_name="Mug",
                        receptacle_name="CounterTop",
                    ),
                )
            ),
        )
    )

    result = PlanVerifier(semantic_client=semantic_client).verify(
        task=_task(),
        robots=build_task_robot_team((1,)),
        scene_objects=(),
        plan=plan,
    )

    assert result.passed is False
    assert result.semantic_checked is True
    assert result.provider == "openai"
    assert result.model == "gpt-5.4"
    assert result.usage == {"reasoning_tokens": 12}
    assert result.issues[0].source == "semantic"
    assert len(semantic_client.requests) == 1


def test_semantic_verification_schema_requires_nullable_issue_indexes() -> None:
    schema = semantic_verification_json_schema()
    issues_property = schema["properties"]["issues"]
    issue_schema = issues_property["items"]

    assert set(issue_schema["required"]) == set(issue_schema["properties"])
    assert issue_schema["properties"]["phase_index"]["type"] == ["integer", "null"]
    assert issue_schema["properties"]["action_index"]["type"] == ["integer", "null"]


def test_semantic_verification_result_accepts_null_issue_indexes() -> None:
    result = semantic_verification_result_from_mapping(
        {
            "issues": [
                {
                    "code": "semantic_gap",
                    "message": "The plan never reaches the fridge before PutObject.",
                    "phase_index": None,
                    "action_index": None,
                }
            ]
        }
    )

    assert result.issues[0].phase_index is None
    assert result.issues[0].action_index is None

"""Primary JSON-first planning path for SMART-LLM v2.

Modern planning models can return structured data directly through tool calling
or strict JSON outputs. The contract here is a typed task-plan payload plus a
small amount of provider metadata so the benchmark loop stays stable while we
swap between Anthropic, OpenAI, and Kimi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from smart_llm_v2.agents.model_profiles import ModelProfile
from smart_llm_v2.agents.plan import ActionRequest, PlanPhase, TaskPlan
from smart_llm_v2.agents.planner import PlanBuildResult, PlanningImage
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import RobotSpec
from smart_llm_v2.skills import ALL_SKILLS, get_skill


@dataclass(frozen=True, slots=True)
class JsonPlannerRequest:
    task: BenchmarkTask
    robots: tuple[RobotSpec, ...]
    scene_objects: tuple[Mapping[str, object], ...]
    profile: ModelProfile
    response_schema: dict[str, object]
    system_message: str
    images: tuple[PlanningImage, ...] = ()

    @property
    def context(self) -> dict[str, object]:
        return build_planning_context(
            task=self.task,
            robots=self.robots,
            scene_objects=self.scene_objects,
            images=self.images if self.profile.vision_enabled else (),
        )


@dataclass(frozen=True, slots=True)
class JsonPlanningResult:
    payload: Mapping[str, object]
    provider: str
    model: str
    usage: Mapping[str, object] | None = None


class JsonPlanningClient(Protocol):
    def complete(self, request: JsonPlannerRequest) -> JsonPlanningResult: ...


class JsonPlanValidationError(ValueError):
    """Raised when a structured plan payload does not match the expected schema."""


@dataclass(frozen=True, slots=True)
class JsonAction:
    robots: tuple[str, ...]
    skill: str
    object_name: str | None = None
    receptacle_name: str | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        valid_robot_names: frozenset[str],
        valid_skills: frozenset[str],
        robot_skills_by_name: Mapping[str, frozenset[str]],
        robot_mass_by_name: Mapping[str, float] | None = None,
        scene_object_mass_by_name: Mapping[str, float] | None = None,
    ) -> "JsonAction":
        robots = _tuple_of_strings(payload, "robots")
        if not robots:
            raise JsonPlanValidationError("Action payload requires at least one robot")
        if len(set(robots)) != len(robots):
            raise JsonPlanValidationError("Action payload cannot repeat the same robot")
        unknown_robots = [robot for robot in robots if robot not in valid_robot_names]
        if unknown_robots:
            raise JsonPlanValidationError(
                f"Action payload references unknown robots: {', '.join(unknown_robots)}"
            )

        skill = _required_string(payload, "skill")
        if skill not in valid_skills:
            raise JsonPlanValidationError(f"Action payload references unknown skill {skill!r}")
        _validate_action_robot_skills(
            robots=robots,
            skill=skill,
            robot_skills_by_name=robot_skills_by_name,
        )

        object_name = _optional_string(payload, "object_name")
        receptacle_name = _optional_string(payload, "receptacle_name")
        _validate_action_arguments(
            skill=skill,
            object_name=object_name,
            receptacle_name=receptacle_name,
        )
        _validate_action_mass_capacity(
            robots=robots,
            skill=skill,
            object_name=object_name,
            robot_mass_by_name=robot_mass_by_name or {},
            scene_object_mass_by_name=scene_object_mass_by_name or {},
        )
        return cls(
            robots=robots,
            skill=skill,
            object_name=object_name,
            receptacle_name=receptacle_name,
        )

    def to_action_request(self) -> ActionRequest:
        return ActionRequest(
            robots=self.robots,
            skill=self.skill,
            object_name=self.object_name,
            receptacle_name=self.receptacle_name,
        )


@dataclass(frozen=True, slots=True)
class JsonPhase:
    actions: tuple[JsonAction, ...]
    label: str | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        valid_robot_names: frozenset[str],
        valid_skills: frozenset[str],
        robot_skills_by_name: Mapping[str, frozenset[str]],
        robot_mass_by_name: Mapping[str, float] | None = None,
        scene_object_mass_by_name: Mapping[str, float] | None = None,
    ) -> "JsonPhase":
        actions_payload = payload.get("actions")
        if not isinstance(actions_payload, Sequence) or isinstance(actions_payload, (str, bytes)):
            raise JsonPlanValidationError("Phase payload must contain an actions list")
        actions = tuple(
            JsonAction.from_mapping(
                _mapping_item(action_payload, "actions"),
                valid_robot_names=valid_robot_names,
                valid_skills=valid_skills,
                robot_skills_by_name=robot_skills_by_name,
                robot_mass_by_name=robot_mass_by_name,
                scene_object_mass_by_name=scene_object_mass_by_name,
            )
            for action_payload in actions_payload
        )
        if not actions:
            raise JsonPlanValidationError("Phase payload must contain at least one action")
        return cls(
            actions=actions,
            label=_optional_string(payload, "label"),
        )

    def to_plan_phase(self) -> PlanPhase:
        return PlanPhase(
            actions=tuple(action.to_action_request() for action in self.actions),
            label=self.label,
        )


@dataclass(frozen=True, slots=True)
class JsonTaskPlan:
    phases: tuple[JsonPhase, ...]
    notes: str | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        valid_robot_names: frozenset[str],
        valid_skills: frozenset[str],
        robot_skills_by_name: Mapping[str, frozenset[str]],
        robot_mass_by_name: Mapping[str, float] | None = None,
        scene_object_mass_by_name: Mapping[str, float] | None = None,
    ) -> "JsonTaskPlan":
        phases_payload = payload.get("phases")
        if not isinstance(phases_payload, Sequence) or isinstance(phases_payload, (str, bytes)):
            raise JsonPlanValidationError("Plan payload must contain a phases list")
        phases = tuple(
            JsonPhase.from_mapping(
                _mapping_item(phase_payload, "phases"),
                valid_robot_names=valid_robot_names,
                valid_skills=valid_skills,
                robot_skills_by_name=robot_skills_by_name,
                robot_mass_by_name=robot_mass_by_name,
                scene_object_mass_by_name=scene_object_mass_by_name,
            )
            for phase_payload in phases_payload
        )
        if not phases:
            raise JsonPlanValidationError("Plan payload must contain at least one phase")
        return cls(
            phases=phases,
            notes=_optional_string(payload, "notes"),
        )

    def to_task_plan(self, *, planner_name: str) -> TaskPlan:
        return TaskPlan(
            phases=tuple(phase.to_plan_phase() for phase in self.phases),
            planner_name=planner_name,
            notes=self.notes,
        )


class JsonPlanner:
    def __init__(
        self,
        *,
        client: JsonPlanningClient,
        profile: ModelProfile,
        planner_name: str | None = None,
        system_message: str | None = None,
    ) -> None:
        self.client = client
        self.profile = profile
        self.planner_name = planner_name or profile.planner_name
        self.system_message = system_message or DEFAULT_SYSTEM_MESSAGE

    @property
    def uses_planning_images(self) -> bool:
        return self.profile.vision_enabled

    def build_plan(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        scene_objects: Sequence[Mapping[str, object]],
        planning_images: Sequence[PlanningImage] = (),
    ) -> PlanBuildResult:
        images = tuple(planning_images) if self.uses_planning_images else ()
        request = JsonPlannerRequest(
            task=task,
            robots=tuple(robots),
            scene_objects=tuple(scene_objects),
            profile=self.profile,
            response_schema=task_plan_json_schema(),
            system_message=self.system_message,
            images=images,
        )
        result = self.client.complete(request)
        json_plan = JsonTaskPlan.from_mapping(
            result.payload,
            valid_robot_names=frozenset(robot.name for robot in robots),
            valid_skills=frozenset(skill.name for skill in ALL_SKILLS),
            robot_skills_by_name={robot.name: robot.skills for robot in robots},
            robot_mass_by_name={robot.name: robot.mass_capacity for robot in robots},
            scene_object_mass_by_name=_build_scene_object_mass_index(scene_objects),
        )
        plan = json_plan.to_task_plan(planner_name=self.planner_name)
        return PlanBuildResult(
            plan=plan,
            provider=result.provider,
            model=result.model,
            usage=result.usage,
            profile_variant=self.profile.variant.value,
        )


DEFAULT_SYSTEM_MESSAGE = (
    "Plan the task using structured JSON only. Maximize successful task completion, "
    "respect robot skills, robot-team assignments, object constraints, and temporal "
    "dependencies, and use safe parallelism where it helps. Prefer fewer robots only "
    "as a tie-break when plans are otherwise equally good."
)


def build_planning_context(
    *,
    task: BenchmarkTask,
    robots: Sequence[RobotSpec],
    scene_objects: Sequence[Mapping[str, object]],
    images: Sequence[PlanningImage] = (),
) -> dict[str, object]:
    context = {
        "task": {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "goal_states": [
                {
                    "name": goal_state.name,
                    "state": goal_state.state,
                    "contains": list(goal_state.contains),
                }
                for goal_state in task.goal_states
            ],
        },
        "robots": [
            {
                "name": robot.name,
                "skills": sorted(robot.skills),
                "mass_capacity": robot.mass_capacity,
                "reference_id": robot.reference_id,
            }
            for robot in robots
        ],
        "scene_objects": [
            {
                "name": _scene_object_name(scene_object),
                "object_type": scene_object.get("objectType"),
                "mass": scene_object.get("mass"),
                "states": {
                    field: scene_object[field]
                    for field in (
                        "isOpen",
                        "isToggled",
                        "isBroken",
                        "isCooked",
                        "isSliced",
                        "isPickedUp",
                        "temperature",
                    )
                    if field in scene_object
                },
            }
            for scene_object in scene_objects
        ],
        "action_schema": task_plan_json_schema(),
    }
    if images:
        context["observation_images"] = [
            {
                "agent_id": image.agent_id,
                "label": image.label,
                "media_type": image.media_type,
            }
            for image in images
        ]
    return context


def task_plan_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["phases"],
        "properties": {
            "notes": {"type": "string"},
            "phases": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["actions"],
                    "properties": {
                        "label": {"type": "string"},
                        "actions": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["robots", "skill"],
                                "properties": {
                                    "robots": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {"type": "string"},
                                    },
                                    "skill": {
                                        "type": "string",
                                        "enum": sorted(skill.name for skill in ALL_SKILLS),
                                    },
                                    "object_name": {"type": "string"},
                                    "receptacle_name": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise JsonPlanValidationError(f"Expected non-empty string for {key!r}")
    return value


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise JsonPlanValidationError(f"Expected string for {key!r}")
    return value


def _tuple_of_strings(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise JsonPlanValidationError(f"Expected list of strings for {key!r}")
    strings = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise JsonPlanValidationError(f"Expected list of strings for {key!r}")
        strings.append(item)
    return tuple(strings)


def _mapping_item(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise JsonPlanValidationError(f"Expected mapping entries inside {key!r}")
    return value


def _validate_action_arguments(
    *,
    skill: str,
    object_name: str | None,
    receptacle_name: str | None,
) -> None:
    parameters = get_skill(skill).parameters
    if len(parameters) >= 2 and object_name is None:
        raise JsonPlanValidationError(f"Skill {skill!r} requires object_name")
    if skill == "PutObject" and receptacle_name is None:
        raise JsonPlanValidationError("Skill 'PutObject' requires receptacle_name")


def _validate_action_robot_skills(
    *,
    robots: Sequence[str],
    skill: str,
    robot_skills_by_name: Mapping[str, frozenset[str]],
) -> None:
    invalid_robots = [
        robot_name
        for robot_name in robots
        if skill not in robot_skills_by_name.get(robot_name, frozenset())
    ]
    if invalid_robots:
        joined_names = ", ".join(invalid_robots)
        raise JsonPlanValidationError(
            f"Action payload assigns skill {skill!r} to robots that cannot execute it: "
            f"{joined_names}"
        )


def _validate_action_mass_capacity(
    *,
    robots: Sequence[str],
    skill: str,
    object_name: str | None,
    robot_mass_by_name: Mapping[str, float],
    scene_object_mass_by_name: Mapping[str, float],
) -> None:
    if skill != "PickupObject" or object_name is None:
        return

    object_mass = scene_object_mass_by_name.get(_canonical_object_name(object_name))
    if object_mass is None:
        return

    team_capacity = sum(robot_mass_by_name[robot_name] for robot_name in robots)
    if team_capacity < object_mass:
        raise JsonPlanValidationError(
            f"Action payload assigns PickupObject to robots with combined mass capacity "
            f"{team_capacity}, but {object_name!r} has mass {object_mass}"
        )


def _build_scene_object_mass_index(
    scene_objects: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    masses: dict[str, float] = {}
    for scene_object in scene_objects:
        mass = scene_object.get("mass")
        if not isinstance(mass, (int, float)):
            continue
        for candidate in (
            scene_object.get("objectType"),
            scene_object.get("name"),
            scene_object.get("objectId"),
        ):
            if not isinstance(candidate, str) or not candidate:
                continue
            masses.setdefault(_canonical_object_name(candidate), float(mass))
    return masses


def _canonical_object_name(name: str) -> str:
    return name.split("|", 1)[0].strip().casefold()


def _scene_object_name(scene_object: Mapping[str, object]) -> str:
    candidate = (
        scene_object.get("objectType")
        or scene_object.get("name")
        or scene_object.get("objectId")
        or "UnknownObject"
    )
    return str(candidate)

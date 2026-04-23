"""Pre-execution plan checks for SMART-LLM v2.

A verifier pass is a second planning check before the executor touches AI2-THOR.
This module uses deterministic rules for the high-confidence failures we can
encode directly, and it leaves a protocol hook for a structured model review
when the project is ready to spend tokens on semantic checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from smart_llm_v2.agents.json_planner import build_planning_context
from smart_llm_v2.agents.plan import ActionRequest, TaskPlan
from smart_llm_v2.agents.planner import PlanningImage
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import RobotSpec
from smart_llm_v2.skills import SKILL_REGISTRY, get_skill

_HANDOFF_SKILLS = frozenset({"PutObject", "DropHandObject", "ThrowObject"})

PLAN_VERIFICATION_TOOL_NAME = "submit_plan_verification"
DEFAULT_SEMANTIC_VERIFIER_SYSTEM_MESSAGE = (
    "Review candidate task plans using structured JSON only. Deterministic checks "
    "already validated schema shape, robot skill coverage, basic argument validity, "
    "and PickupObject mass capacity. "
    "Report only "
    "semantic execution risks that would likely make the plan fail or miss the goal, "
    "and return an empty issues list when the plan is acceptable."
)


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    code: str
    message: str
    source: str = "deterministic"
    phase_index: int | None = None
    action_index: int | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "source": self.source,
        }
        if self.phase_index is not None:
            payload["phase_index"] = self.phase_index
        if self.action_index is not None:
            payload["action_index"] = self.action_index
        return payload


@dataclass(frozen=True, slots=True)
class SemanticVerificationRequest:
    task: BenchmarkTask
    robots: tuple[RobotSpec, ...]
    scene_objects: tuple[Mapping[str, object], ...]
    plan: TaskPlan
    images: tuple[PlanningImage, ...] = ()

    @property
    def context(self) -> dict[str, object]:
        return build_semantic_verification_context(
            task=self.task,
            robots=self.robots,
            scene_objects=self.scene_objects,
            plan=self.plan,
            images=self.images,
        )


@dataclass(frozen=True, slots=True)
class SemanticVerificationResult:
    issues: tuple[VerificationIssue, ...] = ()
    provider: str | None = None
    model: str | None = None
    usage: Mapping[str, object] | None = None


class SemanticVerifierClient(Protocol):
    def review(
        self,
        request: SemanticVerificationRequest,
    ) -> SemanticVerificationResult: ...


class SemanticVerificationPayloadError(ValueError):
    """Raised when a semantic verifier payload does not match the expected schema."""


@dataclass(frozen=True, slots=True)
class PlanVerificationResult:
    issues: tuple[VerificationIssue, ...]
    semantic_checked: bool = False
    provider: str | None = None
    model: str | None = None
    usage: Mapping[str, object] | None = None

    @property
    def passed(self) -> bool:
        return not self.issues

    def error_message(self) -> str:
        if self.passed:
            return ""
        summary = "; ".join(issue.message for issue in self.issues[:3])
        if len(self.issues) > 3:
            summary += f"; {len(self.issues) - 3} more issue(s)"
        return f"PlanVerificationError: {summary}"


class PlanVerifier:
    def __init__(self, *, semantic_client: SemanticVerifierClient | None = None) -> None:
        self.semantic_client = semantic_client

    def verify(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        scene_objects: Sequence[Mapping[str, object]],
        plan: TaskPlan,
        planning_images: Sequence[PlanningImage] = (),
    ) -> PlanVerificationResult:
        issues = list(
            _deterministic_issues(
                robots=robots,
                scene_objects=scene_objects,
                plan=plan,
            )
        )
        if issues or self.semantic_client is None:
            return PlanVerificationResult(issues=tuple(issues))

        semantic_result = self.semantic_client.review(
            SemanticVerificationRequest(
                task=task,
                robots=tuple(robots),
                scene_objects=tuple(scene_objects),
                plan=plan,
                images=tuple(planning_images),
            )
        )
        return PlanVerificationResult(
            issues=semantic_result.issues,
            semantic_checked=True,
            provider=semantic_result.provider,
            model=semantic_result.model,
            usage=semantic_result.usage,
        )


def build_semantic_verification_context(
    *,
    task: BenchmarkTask,
    robots: Sequence[RobotSpec],
    scene_objects: Sequence[Mapping[str, object]],
    plan: TaskPlan,
    images: Sequence[PlanningImage] = (),
) -> dict[str, object]:
    context = build_planning_context(
        task=task,
        robots=robots,
        scene_objects=scene_objects,
        images=images,
    )
    context.pop("action_schema", None)
    context["candidate_plan"] = _plan_payload(plan)
    context["deterministic_checks_passed"] = [
        "Schema shape and required plan fields",
        "Robot names and robot-skill coverage for each action",
        "Required action arguments such as object_name and receptacle_name",
        "PickupObject robot-team mass capacity",
    ]
    return context


def semantic_verification_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["issues"],
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "message", "phase_index", "action_index"],
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "phase_index": {"type": ["integer", "null"], "minimum": 0},
                        "action_index": {"type": ["integer", "null"], "minimum": 0},
                    },
                },
            }
        },
    }


def semantic_verification_result_from_mapping(
    payload: Mapping[str, object],
    *,
    provider: str | None = None,
    model: str | None = None,
    usage: Mapping[str, object] | None = None,
) -> SemanticVerificationResult:
    issues_payload = payload.get("issues")
    if not isinstance(issues_payload, Sequence) or isinstance(issues_payload, (str, bytes)):
        raise SemanticVerificationPayloadError("Verification payload must contain an issues list")

    issues: list[VerificationIssue] = []
    for issue_payload in issues_payload:
        if not isinstance(issue_payload, Mapping):
            raise SemanticVerificationPayloadError("Verification issues must be mappings")
        issues.append(
            VerificationIssue(
                code=_required_string(issue_payload, "code"),
                message=_required_string(issue_payload, "message"),
                source="semantic",
                phase_index=_optional_non_negative_int(issue_payload, "phase_index"),
                action_index=_optional_non_negative_int(issue_payload, "action_index"),
            )
        )
    return SemanticVerificationResult(
        issues=tuple(issues),
        provider=provider,
        model=model,
        usage=usage,
    )


def _deterministic_issues(
    *,
    robots: Sequence[RobotSpec],
    scene_objects: Sequence[Mapping[str, object]],
    plan: TaskPlan,
) -> tuple[VerificationIssue, ...]:
    issues: list[VerificationIssue] = []
    issues.extend(_contract_issues(robots=robots, plan=plan))
    issues.extend(
        _mass_capacity_issues(robots=robots, scene_objects=scene_objects, plan=plan)
    )
    issues.extend(_temporal_issues(scene_objects=scene_objects, plan=plan))
    return tuple(issues)


def _contract_issues(
    *,
    robots: Sequence[RobotSpec],
    plan: TaskPlan,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    robots_by_name = {robot.name: robot for robot in robots}

    for phase_index, phase in enumerate(plan.phases):
        for action_index, action in enumerate(phase.actions):
            if len(set(action.robots)) != len(action.robots):
                issues.append(
                    VerificationIssue(
                        code="duplicate_robot_assignment",
                        message=(
                            f"Action {action_index + 1} in phase {phase_index + 1} "
                            "assigns the same robot more than once"
                        ),
                        phase_index=phase_index,
                        action_index=action_index,
                    )
                )

            unknown_robots = tuple(
                dict.fromkeys(
                    robot_name
                    for robot_name in action.robots
                    if robot_name not in robots_by_name
                )
            )
            if unknown_robots:
                issues.append(
                    VerificationIssue(
                        code="unknown_robot",
                        message=(
                            f"Action {action_index + 1} in phase {phase_index + 1} "
                            f"references unknown robot(s): {', '.join(unknown_robots)}"
                        ),
                        phase_index=phase_index,
                        action_index=action_index,
                    )
                )

            if action.skill not in SKILL_REGISTRY:
                issues.append(
                    VerificationIssue(
                        code="unknown_skill",
                        message=(
                            f"Action {action_index + 1} in phase {phase_index + 1} "
                            f"references unknown skill {action.skill!r}"
                        ),
                        phase_index=phase_index,
                        action_index=action_index,
                    )
                )
                continue

            issues.extend(
                _action_argument_issues(
                    action=action,
                    phase_index=phase_index,
                    action_index=action_index,
                )
            )

            unsupported_robots = tuple(
                robot_name
                for robot_name in action.robots
                if robot_name in robots_by_name
                and not robots_by_name[robot_name].can(action.skill)
            )
            if unsupported_robots:
                issues.append(
                    VerificationIssue(
                        code="unsupported_skill",
                        message=(
                            f"Action {action_index + 1} in phase {phase_index + 1} "
                            f"assigns {action.skill!r} to robot(s) that cannot execute it: "
                            f"{', '.join(unsupported_robots)}"
                        ),
                        phase_index=phase_index,
                        action_index=action_index,
                    )
                )
    return issues


def _mass_capacity_issues(
    *,
    robots: Sequence[RobotSpec],
    scene_objects: Sequence[Mapping[str, object]],
    plan: TaskPlan,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    robots_by_name = {robot.name: robot for robot in robots}
    mass_by_object = _scene_mass_index(scene_objects)

    for phase_index, phase in enumerate(plan.phases):
        for action_index, action in enumerate(phase.actions):
            if action.skill != "PickupObject" or action.object_name is None:
                continue
            object_mass = mass_by_object.get(_canonical_text(action.object_name))
            if object_mass is None:
                continue
            if any(robot_name not in robots_by_name for robot_name in action.robots):
                continue

            team_capacity = sum(
                robots_by_name[robot_name].mass_capacity for robot_name in action.robots
            )
            if team_capacity >= object_mass:
                continue
            issues.append(
                VerificationIssue(
                    code="insufficient_mass_capacity",
                    message=(
                        f"Action {action_index + 1} in phase {phase_index + 1} assigns "
                        f"PickupObject to robot(s) with combined mass capacity {team_capacity}, "
                        f"but {action.object_name!r} has mass {object_mass}"
                    ),
                    phase_index=phase_index,
                    action_index=action_index,
                )
            )
    return issues


def _action_argument_issues(
    *,
    action: ActionRequest,
    phase_index: int,
    action_index: int,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    if len(get_skill(action.skill).parameters) >= 2 and action.object_name is None:
        issues.append(
            VerificationIssue(
                code="missing_action_argument",
                message=(
                    f"Action {action_index + 1} in phase {phase_index + 1} "
                    f"uses {action.skill!r} without object_name"
                ),
                phase_index=phase_index,
                action_index=action_index,
            )
        )
    if action.skill == "PutObject" and action.receptacle_name is None:
        issues.append(
            VerificationIssue(
                code="missing_action_argument",
                message=(
                    f"Action {action_index + 1} in phase {phase_index + 1} "
                    "uses 'PutObject' without receptacle_name"
                ),
                phase_index=phase_index,
                action_index=action_index,
            )
        )
    return issues


def _temporal_issues(
    *,
    scene_objects: Sequence[Mapping[str, object]],
    plan: TaskPlan,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    held_objects_by_robot: dict[str, str] = {}
    open_state_by_object = _scene_open_state_index(scene_objects)

    for phase_index, phase in enumerate(plan.phases):
        phase_holds = dict(held_objects_by_robot)
        phase_open_state = dict(open_state_by_object)
        for action_index, action in enumerate(phase.actions):
            issues.extend(
                _holding_state_issues(
                    action=action,
                    phase_index=phase_index,
                    action_index=action_index,
                    scene_objects=scene_objects,
                    held_objects_by_robot=phase_holds,
                )
            )
            issues.extend(
                _receptacle_state_issues(
                    action=action,
                    phase_index=phase_index,
                    action_index=action_index,
                    scene_objects=scene_objects,
                    open_state_by_object=phase_open_state,
                )
            )
            _apply_action_effect(
                action=action,
                scene_objects=scene_objects,
                next_holds=phase_holds,
                next_open_state=phase_open_state,
                current_holds=phase_holds,
            )
        held_objects_by_robot = phase_holds
        open_state_by_object = phase_open_state
    return issues


def _holding_state_issues(
    *,
    action: ActionRequest,
    phase_index: int,
    action_index: int,
    scene_objects: Sequence[Mapping[str, object]],
    held_objects_by_robot: Mapping[str, str],
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    if action.object_name is None:
        return issues
    matching_keys = _matching_object_keys(action.object_name, scene_objects)

    current_holders = {
        robot_name
        for robot_name, held_object in held_objects_by_robot.items()
        if held_object in matching_keys
    }

    if action.skill == "PickupObject":
        blocked_by = sorted(current_holders.difference(action.robots))
        blocked_keys = {held_objects_by_robot[robot_name] for robot_name in blocked_by}
        if blocked_by and set(matching_keys).issubset(blocked_keys):
            issues.append(
                VerificationIssue(
                    code="object_already_held",
                    message=(
                        f"{action.object_name!r} is already assigned to {', '.join(blocked_by)} "
                        "in an earlier phase"
                    ),
                    phase_index=phase_index,
                    action_index=action_index,
                )
            )
        for robot_name in action.robots:
            held_object = held_objects_by_robot.get(robot_name)
            if held_object is None:
                continue
            if held_object in matching_keys:
                message = (
                    f"{robot_name} picks up {action.object_name!r} again in phase "
                    f"{phase_index + 1}"
                )
            else:
                message = (
                    f"{robot_name} picks up {action.object_name!r} in phase {phase_index + 1} "
                    f"while still holding {_held_object_label(held_object, scene_objects)!r}"
                )
            issues.append(
                VerificationIssue(
                    code="pickup_without_release",
                    message=message,
                    phase_index=phase_index,
                    action_index=action_index,
                )
            )
        return issues

    if action.skill not in _HANDOFF_SKILLS:
        return issues

    for robot_name in action.robots:
        held_object = held_objects_by_robot.get(robot_name)
        if held_object in matching_keys:
            return issues

    issues.append(
        VerificationIssue(
            code="missing_pickup_before_handoff",
            message=(
                f"{action.skill} on {action.object_name!r} in phase {phase_index + 1} "
                "does not include a robot with an earlier PickupObject"
            ),
            phase_index=phase_index,
            action_index=action_index,
        )
    )
    return issues


def _receptacle_state_issues(
    *,
    action: ActionRequest,
    phase_index: int,
    action_index: int,
    scene_objects: Sequence[Mapping[str, object]],
    open_state_by_object: Mapping[str, bool],
) -> list[VerificationIssue]:
    if action.skill != "PutObject":
        return []

    if action.receptacle_name is None:
        return []
    matching_keys = _matching_scene_object_keys(action.receptacle_name, scene_objects)
    if not matching_keys:
        if open_state_by_object.get(_fallback_object_key(action.receptacle_name), True):
            return []
    elif any(key not in open_state_by_object for key in matching_keys):
        return []
    elif any(open_state_by_object[key] for key in matching_keys):
        return []
    return [
        VerificationIssue(
            code="closed_receptacle",
            message=(
                f"{action.skill} targets closed receptacle {action.receptacle_name!r} in "
                f"phase {phase_index + 1}"
            ),
            phase_index=phase_index,
            action_index=action_index,
        )
    ]


def _apply_action_effect(
    *,
    action: ActionRequest,
    scene_objects: Sequence[Mapping[str, object]],
    next_holds: dict[str, str],
    next_open_state: dict[str, bool],
    current_holds: Mapping[str, str],
) -> None:
    if action.skill == "PickupObject" and action.object_name is not None:
        held_object = _pickup_claim_key(
            object_name=action.object_name,
            robots=action.robots,
            scene_objects=scene_objects,
            current_holds=current_holds,
        )
        if held_object is None:
            return
        for robot_name in action.robots:
            if robot_name not in current_holds:
                next_holds[robot_name] = held_object
        return

    if action.skill in _HANDOFF_SKILLS and action.object_name is not None:
        held_keys = _held_keys_for_action(
            object_name=action.object_name,
            robots=action.robots,
            scene_objects=scene_objects,
            current_holds=current_holds,
        )
        for robot_name, held_object in tuple(current_holds.items()):
            if held_object in held_keys:
                next_holds.pop(robot_name, None)

    if action.skill == "OpenObject" and action.object_name is not None:
        object_key = _state_effect_target_key(
            object_name=action.object_name,
            desired_state=True,
            scene_objects=scene_objects,
            current_open_state=next_open_state,
        )
        if object_key is not None:
            next_open_state[object_key] = True
    if action.skill == "CloseObject" and action.object_name is not None:
        object_key = _state_effect_target_key(
            object_name=action.object_name,
            desired_state=False,
            scene_objects=scene_objects,
            current_open_state=next_open_state,
        )
        if object_key is not None:
            next_open_state[object_key] = False


def _scene_open_state_index(
    scene_objects: Sequence[Mapping[str, object]],
) -> dict[str, bool]:
    states: dict[str, bool] = {}
    for index, scene_object in enumerate(scene_objects):
        is_open = scene_object.get("isOpen")
        if not isinstance(is_open, bool):
            continue
        states[_scene_object_key(index)] = is_open
    return states


def _scene_mass_index(
    scene_objects: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    masses: dict[str, float] = {}
    for scene_object in scene_objects:
        mass = scene_object.get("mass")
        if isinstance(mass, bool) or not isinstance(mass, (int, float)):
            continue
        for candidate in (
            scene_object.get("objectType"),
            scene_object.get("name"),
            scene_object.get("objectId"),
        ):
            if isinstance(candidate, str) and candidate:
                masses.setdefault(_canonical_text(candidate), float(mass))
    return masses


def _pickup_claim_key(
    *,
    object_name: str,
    robots: Sequence[str],
    scene_objects: Sequence[Mapping[str, object]],
    current_holds: Mapping[str, str],
) -> str | None:
    matching_keys = _matching_scene_object_keys(object_name, scene_objects)
    if not matching_keys:
        return _fallback_object_key(object_name)

    for robot_name in robots:
        held_object = current_holds.get(robot_name)
        if held_object in matching_keys:
            return held_object

    held_keys = set(current_holds.values())
    for matching_key in matching_keys:
        if matching_key not in held_keys:
            return matching_key
    return None


def _held_keys_for_action(
    *,
    object_name: str,
    robots: Sequence[str],
    scene_objects: Sequence[Mapping[str, object]],
    current_holds: Mapping[str, str],
) -> set[str]:
    matching_keys = set(_matching_object_keys(object_name, scene_objects))
    return {
        held_object
        for robot_name in robots
        if (held_object := current_holds.get(robot_name)) in matching_keys
    }


def _state_effect_target_key(
    *,
    object_name: str,
    desired_state: bool,
    scene_objects: Sequence[Mapping[str, object]],
    current_open_state: Mapping[str, bool],
) -> str | None:
    matching_keys = _matching_scene_object_keys(object_name, scene_objects)
    if not matching_keys:
        return _fallback_object_key(object_name)
    if "|" in object_name:
        return matching_keys[0]

    for matching_key in matching_keys:
        if current_open_state.get(matching_key) is not desired_state:
            return matching_key
    return matching_keys[0]


def _matching_object_keys(
    object_name: str,
    scene_objects: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    matching_keys = _matching_scene_object_keys(object_name, scene_objects)
    if matching_keys:
        return matching_keys
    return (_fallback_object_key(object_name),)


def _matching_scene_object_keys(
    object_name: str,
    scene_objects: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    exact_query = _canonical_exact_text(object_name)
    base_query = _canonical_text(object_name)
    exact_matches: list[str] = []
    base_matches: list[str] = []

    for index, scene_object in enumerate(scene_objects):
        exact_aliases = _scene_object_exact_aliases(scene_object)
        base_aliases = _scene_object_base_aliases(scene_object)
        object_key = _scene_object_key(index)
        if "|" in object_name and exact_query in exact_aliases:
            exact_matches.append(object_key)
        if base_query in base_aliases:
            base_matches.append(object_key)

    return tuple(exact_matches or base_matches)


def _scene_object_exact_aliases(scene_object: Mapping[str, object]) -> set[str]:
    aliases: set[str] = set()
    for candidate in (
        scene_object.get("objectType"),
        scene_object.get("name"),
        scene_object.get("objectId"),
    ):
        if isinstance(candidate, str) and candidate:
            aliases.add(_canonical_exact_text(candidate))
    return aliases


def _scene_object_base_aliases(scene_object: Mapping[str, object]) -> set[str]:
    aliases: set[str] = set()
    for candidate in (
        scene_object.get("objectType"),
        scene_object.get("name"),
        scene_object.get("objectId"),
    ):
        if isinstance(candidate, str) and candidate:
            aliases.add(_canonical_text(candidate))
    return aliases


def _held_object_label(
    held_object: str,
    scene_objects: Sequence[Mapping[str, object]],
) -> str:
    if not held_object.startswith("scene:"):
        return held_object.removeprefix("query:")
    try:
        scene_object = scene_objects[int(held_object.removeprefix("scene:"))]
    except (IndexError, ValueError):
        return held_object

    for candidate in (
        scene_object.get("name"),
        scene_object.get("objectId"),
        scene_object.get("objectType"),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return held_object


def _scene_object_key(index: int) -> str:
    return f"scene:{index}"


def _fallback_object_key(name: str) -> str:
    return f"query:{_canonical_exact_text(name)}"


def _canonical_exact_text(name: str) -> str:
    return name.strip().casefold()


def _canonical_text(name: str) -> str:
    return name.split("|", 1)[0].strip().casefold()


def _plan_payload(plan: TaskPlan) -> dict[str, object]:
    return {
        "planner_name": plan.planner_name,
        "notes": plan.notes,
        "phases": [
            {
                "label": phase.label,
                "actions": [
                    {
                        "robots": list(action.robots),
                        "skill": action.skill,
                        "object_name": action.object_name,
                        "receptacle_name": action.receptacle_name,
                    }
                    for action in phase.actions
                ],
            }
            for phase in plan.phases
        ],
    }


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SemanticVerificationPayloadError(f"Expected non-empty string for {key!r}")
    return value


def _optional_non_negative_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SemanticVerificationPayloadError(f"Expected non-negative integer for {key!r}")
    return value

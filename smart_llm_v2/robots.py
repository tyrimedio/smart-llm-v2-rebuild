from __future__ import annotations

from dataclasses import dataclass, replace


ALL_CORE_SKILLS = frozenset(
    {
        "GoToObject",
        "OpenObject",
        "CloseObject",
        "BreakObject",
        "SliceObject",
        "SwitchOn",
        "SwitchOff",
        "PickupObject",
        "PutObject",
        "DropHandObject",
        "ThrowObject",
        "PushObject",
        "PullObject",
    }
)


@dataclass(frozen=True, slots=True)
class RobotSpec:
    reference_id: int
    name: str
    skills: frozenset[str]
    mass_capacity: float
    reference_name: str | None = None

    def renamed(self, runtime_name: str) -> "RobotSpec":
        return replace(self, name=runtime_name)

    def can(self, skill_name: str) -> bool:
        return skill_name in self.skills


def load_reference_robots() -> tuple[RobotSpec, ...]:
    return REFERENCE_ROBOTS


def get_reference_robot(robot_id: int) -> RobotSpec:
    return REFERENCE_ROBOTS[robot_id - 1]


def build_task_robot_team(robot_ids: list[int] | tuple[int, ...]) -> tuple[RobotSpec, ...]:
    robots = []
    for index, robot_id in enumerate(robot_ids, start=1):
        robots.append(get_reference_robot(robot_id).renamed(f"robot{index}"))
    return tuple(robots)


def _robot(
    reference_id: int,
    *,
    skills: frozenset[str],
    mass_capacity: float,
) -> RobotSpec:
    name = f"robot{reference_id}"
    return RobotSpec(
        reference_id=reference_id,
        name=name,
        reference_name=name,
        skills=skills,
        mass_capacity=mass_capacity,
    )


REFERENCE_ROBOTS = (
    _robot(1, skills=ALL_CORE_SKILLS, mass_capacity=100.0),
    _robot(2, skills=ALL_CORE_SKILLS, mass_capacity=100.0),
    _robot(3, skills=ALL_CORE_SKILLS, mass_capacity=100.0),
    _robot(4, skills=ALL_CORE_SKILLS, mass_capacity=100.0),
    _robot(5, skills=ALL_CORE_SKILLS, mass_capacity=1.0),
    _robot(6, skills=ALL_CORE_SKILLS, mass_capacity=2.1),
    _robot(7, skills=ALL_CORE_SKILLS, mass_capacity=0.08),
    _robot(8, skills=ALL_CORE_SKILLS, mass_capacity=0.4),
    _robot(9, skills=ALL_CORE_SKILLS, mass_capacity=5.0),
    _robot(10, skills=ALL_CORE_SKILLS, mass_capacity=0.02),
    _robot(
        11,
        skills=frozenset(
            {
                "GoToObject",
                "BreakObject",
                "SliceObject",
                "PickupObject",
                "PutObject",
                "DropHandObject",
                "ThrowObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        12,
        skills=frozenset(
            {
                "GoToObject",
                "BreakObject",
                "SliceObject",
                "SwitchOn",
                "SwitchOff",
                "DropHandObject",
                "ThrowObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        13,
        skills=frozenset(
            {
                "GoToObject",
                "BreakObject",
                "SwitchOn",
                "SwitchOff",
                "PickupObject",
                "PutObject",
                "DropHandObject",
                "ThrowObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        14,
        skills=frozenset(
            {
                "GoToObject",
                "BreakObject",
                "SliceObject",
                "SwitchOn",
                "SwitchOff",
                "PickupObject",
                "PutObject",
                "DropHandObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        15,
        skills=frozenset(
            {
                "GoToObject",
                "BreakObject",
                "SliceObject",
                "SwitchOn",
                "SwitchOff",
                "PickupObject",
                "PutObject",
                "DropHandObject",
                "ThrowObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        16,
        skills=frozenset(
            {
                "GoToObject",
                "OpenObject",
                "CloseObject",
                "BreakObject",
                "SliceObject",
                "DropHandObject",
                "ThrowObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        17,
        skills=frozenset(
            {
                "GoToObject",
                "OpenObject",
                "CloseObject",
                "BreakObject",
                "PickupObject",
                "PutObject",
                "DropHandObject",
                "ThrowObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        18,
        skills=frozenset(
            {
                "GoToObject",
                "OpenObject",
                "CloseObject",
                "SliceObject",
                "PickupObject",
                "PutObject",
                "DropHandObject",
                "ThrowObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        19,
        skills=frozenset(
            {
                "GoToObject",
                "OpenObject",
                "CloseObject",
                "BreakObject",
                "SliceObject",
                "PickupObject",
                "PutObject",
                "DropHandObject",
                "ThrowObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        20,
        skills=frozenset(
            {
                "GoToObject",
                "OpenObject",
                "CloseObject",
                "BreakObject",
                "SwitchOn",
                "SwitchOff",
                "DropHandObject",
                "ThrowObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        21,
        skills=frozenset(
            {
                "GoToObject",
                "OpenObject",
                "CloseObject",
                "BreakObject",
                "SliceObject",
                "SwitchOn",
                "SwitchOff",
                "DropHandObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(
        22,
        skills=frozenset(
            {
                "GoToObject",
                "OpenObject",
                "CloseObject",
                "BreakObject",
                "SliceObject",
                "SwitchOn",
                "SwitchOff",
                "DropHandObject",
                "ThrowObject",
                "PushObject",
                "PullObject",
            }
        ),
        mass_capacity=100.0,
    ),
    _robot(23, skills=frozenset({"GoToObject", "OpenObject", "CloseObject"}), mass_capacity=100.0),
    _robot(24, skills=frozenset({"GoToObject", "SwitchOn", "SwitchOff"}), mass_capacity=100.0),
    _robot(25, skills=frozenset({"GoToObject", "PickupObject", "PutObject"}), mass_capacity=100.0),
    _robot(26, skills=frozenset({"GoToObject", "SliceObject", "PickupObject"}), mass_capacity=100.0),
    _robot(27, skills=frozenset({"GoToObject", "BreakObject", "ThrowObject"}), mass_capacity=100.0),
    _robot(28, skills=ALL_CORE_SKILLS, mass_capacity=0.9),
)

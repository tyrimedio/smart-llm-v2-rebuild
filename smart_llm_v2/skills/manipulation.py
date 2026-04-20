from __future__ import annotations

from smart_llm_v2.skills.base import SkillSpec

OPEN_OBJECT = SkillSpec(
    name="OpenObject",
    parameters=("robot", "object"),
    simulator_action="OpenObject",
)
CLOSE_OBJECT = SkillSpec(
    name="CloseObject",
    parameters=("robot", "object"),
    simulator_action="CloseObject",
)
BREAK_OBJECT = SkillSpec(
    name="BreakObject",
    parameters=("robot", "object"),
    simulator_action="BreakObject",
)
SLICE_OBJECT = SkillSpec(
    name="SliceObject",
    parameters=("robot", "object"),
    simulator_action="SliceObject",
)
SWITCH_ON = SkillSpec(
    name="SwitchOn",
    parameters=("robot", "object"),
    simulator_action="ToggleObjectOn",
)
SWITCH_OFF = SkillSpec(
    name="SwitchOff",
    parameters=("robot", "object"),
    simulator_action="ToggleObjectOff",
)
CLEAN_OBJECT = SkillSpec(
    name="CleanObject",
    parameters=("robot", "object"),
    simulator_action="CleanObject",
)
PICKUP_OBJECT = SkillSpec(
    name="PickupObject",
    parameters=("robot", "object"),
    simulator_action="PickupObject",
)
PUT_OBJECT = SkillSpec(
    name="PutObject",
    parameters=("robot", "object", "receptacleObject"),
    simulator_action="PutObject",
)
DROP_HAND_OBJECT = SkillSpec(
    name="DropHandObject",
    parameters=("robot", "object"),
    simulator_action="DropHandObject",
)
THROW_OBJECT = SkillSpec(
    name="ThrowObject",
    parameters=("robot", "object"),
    simulator_action="ThrowObject",
)

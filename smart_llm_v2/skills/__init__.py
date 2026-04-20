from smart_llm_v2.skills.base import SkillSpec, build_skill_registry
from smart_llm_v2.skills.manipulation import (
    BREAK_OBJECT,
    CLEAN_OBJECT,
    CLOSE_OBJECT,
    DROP_HAND_OBJECT,
    OPEN_OBJECT,
    PICKUP_OBJECT,
    PUT_OBJECT,
    SLICE_OBJECT,
    SWITCH_OFF,
    SWITCH_ON,
    THROW_OBJECT,
)
from smart_llm_v2.skills.mobile import GO_TO_OBJECT, PULL_OBJECT, PUSH_OBJECT

ALL_SKILLS = (
    GO_TO_OBJECT,
    OPEN_OBJECT,
    CLOSE_OBJECT,
    BREAK_OBJECT,
    SLICE_OBJECT,
    SWITCH_ON,
    SWITCH_OFF,
    CLEAN_OBJECT,
    PICKUP_OBJECT,
    PUT_OBJECT,
    DROP_HAND_OBJECT,
    THROW_OBJECT,
    PUSH_OBJECT,
    PULL_OBJECT,
)

SKILL_REGISTRY = build_skill_registry(*ALL_SKILLS)
REFERENCE_PROMPT_SIGNATURES = tuple(skill.prompt_signature for skill in ALL_SKILLS)


def get_skill(name: str) -> SkillSpec:
    return SKILL_REGISTRY[name]


__all__ = [
    "ALL_SKILLS",
    "REFERENCE_PROMPT_SIGNATURES",
    "SKILL_REGISTRY",
    "SkillSpec",
    "get_skill",
]

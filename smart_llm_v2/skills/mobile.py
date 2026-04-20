from __future__ import annotations

from smart_llm_v2.skills.base import SkillSpec

GO_TO_OBJECT = SkillSpec(name="GoToObject", parameters=("robot", "object"))
PUSH_OBJECT = SkillSpec(
    name="PushObject",
    parameters=("robot", "object"),
    simulator_action="PushObject",
)
PULL_OBJECT = SkillSpec(
    name="PullObject",
    parameters=("robot", "object"),
    simulator_action="PullObject",
)

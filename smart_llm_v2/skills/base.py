from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SkillSpec:
    name: str
    parameters: tuple[str, ...]
    simulator_action: str | None = None

    @property
    def prompt_signature(self) -> str:
        return f"{self.name} " + "".join(f"<{parameter}>" for parameter in self.parameters)


def build_skill_registry(*skills: SkillSpec) -> dict[str, SkillSpec]:
    return {skill.name: skill for skill in skills}

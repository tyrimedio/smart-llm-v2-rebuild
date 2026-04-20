"""Paper-style staged planner for the SMART-LLM control condition.

The original SMART-LLM paper used three LLM prompt stages before execution:
task decomposition, allocation reasoning, and final code generation. We keep
that structure here because the first benchmark milestone is reproduction, not
redesign. A later tool-calling planner can replace these string prompts while
still targeting the same `TaskPlan` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from smart_llm_v2.agents.plan import TaskPlan
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import RobotSpec
from smart_llm_v2.skills import REFERENCE_PROMPT_SIGNATURES

PAPER_PROMPT_DIR = Path(__file__).resolve().parents[2] / "SMART-LLM" / "data" / "pythonic_plans"


@dataclass(frozen=True, slots=True)
class PromptRequest:
    stage: str
    prompt: str
    max_tokens: int
    frequency_penalty: float = 0.0
    stop: tuple[str, ...] = ()
    system_message: str | None = None


@dataclass(frozen=True, slots=True)
class PaperPlannerArtifacts:
    decomposition: str
    allocation_solution: str
    code_solution: str


@dataclass(frozen=True, slots=True)
class PaperPromptAssets:
    decomposition_examples: str
    allocation_solution_examples: str
    allocation_code_examples: str

    @classmethod
    def load(cls, prompt_dir: Path | None = None) -> "PaperPromptAssets":
        base_dir = prompt_dir or PAPER_PROMPT_DIR
        return cls(
            decomposition_examples=(base_dir / "train_task_decompose.py").read_text(),
            allocation_solution_examples=(
                base_dir / "train_task_allocation_solution.py"
            ).read_text(),
            allocation_code_examples=(base_dir / "train_task_allocation_code.py").read_text(),
        )


class TextGenerationClient(Protocol):
    def complete(self, request: PromptRequest) -> str: ...


class PaperPlanParser(Protocol):
    def parse(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        artifacts: PaperPlannerArtifacts,
    ) -> TaskPlan: ...


class PaperStagedPlanner:
    def __init__(
        self,
        *,
        client: TextGenerationClient,
        parser: PaperPlanParser,
        prompt_assets: PaperPromptAssets | None = None,
    ) -> None:
        self.client = client
        self.parser = parser
        self.prompt_assets = prompt_assets or PaperPromptAssets.load()

    def build_plan(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        scene_objects: Sequence[Mapping[str, object]],
    ) -> TaskPlan:
        artifacts = self.generate_artifacts(
            task=task,
            robots=robots,
            scene_objects=scene_objects,
        )
        return self.parser.parse(task=task, robots=robots, artifacts=artifacts)

    def generate_artifacts(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        scene_objects: Sequence[Mapping[str, object]],
    ) -> PaperPlannerArtifacts:
        decomposition_request = self.decomposition_request(
            task=task,
            scene_objects=scene_objects,
        )
        decomposition = self.client.complete(decomposition_request)

        allocation_request = self.allocation_solution_request(
            task=task,
            robots=robots,
            scene_objects=scene_objects,
            decomposition=decomposition,
        )
        allocation_solution = self.client.complete(allocation_request)

        code_request = self.code_solution_request(
            robots=robots,
            scene_objects=scene_objects,
            decomposition=decomposition,
            allocation_solution=allocation_solution,
        )
        code_solution = self.client.complete(code_request)

        return PaperPlannerArtifacts(
            decomposition=decomposition,
            allocation_solution=allocation_solution,
            code_solution=code_solution,
        )

    def decomposition_request(
        self,
        *,
        task: BenchmarkTask,
        scene_objects: Sequence[Mapping[str, object]],
    ) -> PromptRequest:
        prompt = self._prompt_preamble(scene_objects)
        prompt += "\n\n" + self.prompt_assets.decomposition_examples
        prompt += f"\n\n# Task Description: {task.instruction}"
        return PromptRequest(
            stage="decomposition",
            prompt=prompt,
            max_tokens=1300,
            frequency_penalty=0.0,
        )

    def allocation_solution_request(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        scene_objects: Sequence[Mapping[str, object]],
        decomposition: str,
    ) -> PromptRequest:
        prompt = self._prompt_preamble(scene_objects, include_objects=False)
        prompt += "\n\n" + self.prompt_assets.allocation_solution_examples + "\n\n"
        prompt += decomposition
        prompt += "\n# TASK ALLOCATION"
        prompt += (
            f"\n# Scenario: There are {len(robots)} robots available, "
            "The task should be performed using the minimum number of robots necessary. "
            "Robots should be assigned to subtasks that match its skills and mass capacity. "
            "Using your reasoning come up with a solution to satisfy all contraints."
        )
        prompt += f"\n\nrobots = {serialize_prompt_robots(robots)}"
        prompt += f"\n{serialize_prompt_objects(scene_objects)}"
        prompt += (
            "\n\n# IMPORTANT: The AI should ensure that the robots assigned to the tasks "
            "have all the necessary skills to perform the tasks. IMPORTANT: Determine "
            "whether the subtasks must be performed sequentially or in parallel, or a "
            "combination of both and allocate robots based on availablitiy."
        )
        prompt += "\n# SOLUTION  \n"
        return PromptRequest(
            stage="allocation_solution",
            prompt=prompt,
            max_tokens=400,
            frequency_penalty=0.69,
            system_message=(
                "You are a Robot Task Allocation Expert. Determine whether the subtasks "
                "must be performed sequentially or in parallel, or a combination of both "
                "based on your reasoning. In the case of Task Allocation based on Robot "
                "Skills alone, first check if robot teams are required. Then ensure that "
                "robot skills or robot team skills match the required skills for the "
                "subtask when allocating. In the case of Task Allocation based on Mass "
                "alone, first check if robot teams are required. Then ensure that robot "
                "mass capacity or robot team combined mass capacity is greater than or "
                "equal to the mass for the object when allocating."
            ),
        )

    def code_solution_request(
        self,
        *,
        robots: Sequence[RobotSpec],
        scene_objects: Sequence[Mapping[str, object]],
        decomposition: str,
        allocation_solution: str,
    ) -> PromptRequest:
        prompt = self._prompt_preamble(scene_objects)
        prompt += "\n\n" + self.prompt_assets.allocation_code_examples + "\n\n"
        prompt += decomposition
        prompt += "\n# TASK ALLOCATION"
        prompt += f"\n\nrobots = {serialize_prompt_robots(robots)}"
        prompt += allocation_solution
        prompt += "\n# CODE Solution  \n"
        return PromptRequest(
            stage="code_solution",
            prompt=prompt,
            max_tokens=1400,
            frequency_penalty=0.4,
            system_message="You are a Robot Task Allocation Expert",
        )

    def _prompt_preamble(
        self,
        scene_objects: Sequence[Mapping[str, object]],
        *,
        include_objects: bool = True,
    ) -> str:
        prompt = "from skills import " + ", ".join(REFERENCE_PROMPT_SIGNATURES)
        prompt += "\nimport time"
        prompt += "\nimport threading"
        if include_objects:
            prompt += f"\n\n{serialize_prompt_objects(scene_objects)}"
        return prompt


class UnparsedPaperPlanParser:
    def parse(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        artifacts: PaperPlannerArtifacts,
    ) -> TaskPlan:
        raise NotImplementedError(
            "Parsing the paper-style generated Python into TaskPlan is not implemented yet."
        )


def serialize_prompt_robots(robots: Sequence[RobotSpec]) -> list[dict[str, object]]:
    return [
        {
            "name": robot.name,
            "no_skills": len(robot.skills),
            "skills": sorted(robot.skills),
            "mass": robot.mass_capacity,
        }
        for robot in robots
    ]


def serialize_prompt_objects(
    scene_objects: Sequence[Mapping[str, object]],
) -> str:
    objects = []
    for scene_object in scene_objects:
        name = (
            scene_object.get("objectType")
            or scene_object.get("name")
            or scene_object.get("objectId")
            or "UnknownObject"
        )
        objects.append(
            {
                "name": _normalize_object_name(str(name)),
                "mass": float(scene_object.get("mass", 1.0)),
            }
        )
    return f"objects = {objects}"


def _normalize_object_name(name: str) -> str:
    return name.split("|", 1)[0].strip()

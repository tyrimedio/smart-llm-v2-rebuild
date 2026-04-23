"""Legacy paper-style staged planner for the SMART-LLM control condition.

The original SMART-LLM paper used three LLM prompt stages before execution:
task decomposition, allocation reasoning, and final code generation. We keep
that structure here only as a historical ablation. The primary v2 planner path
uses structured JSON plans instead of generated Python.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from smart_llm_v2.agents.plan import ActionRequest, PlanPhase, PlanSubTask, TaskPlan
from smart_llm_v2.agents.planner import PlanBuildResult, PlanningImage
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import RobotSpec
from smart_llm_v2.skills import REFERENCE_PROMPT_SIGNATURES

PAPER_PROMPT_DIR = Path(__file__).resolve().parents[2] / "SMART-LLM" / "data" / "pythonic_plans"
PAPER_ACTION_NAMES = {signature.split(" ", 1)[0] for signature in REFERENCE_PROMPT_SIGNATURES}


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
            allocation_code_examples=_normalize_allocation_code_examples(
                (base_dir / "train_task_allocation_code.py").read_text()
            ),
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

    @property
    def uses_planning_images(self) -> bool:
        return False

    def build_plan(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        scene_objects: Sequence[Mapping[str, object]],
        planning_images: Sequence[PlanningImage] = (),
    ) -> PlanBuildResult:
        artifacts = self.generate_artifacts(
            task=task,
            robots=robots,
            scene_objects=scene_objects,
        )
        return PlanBuildResult(
            plan=self.parser.parse(task=task, robots=robots, artifacts=artifacts),
            provider="legacy",
            model=None,
            usage=None,
            profile_variant="legacy",
        )

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


class PaperPlanParseError(ValueError):
    """Raised when paper-style generated code cannot be converted into a TaskPlan."""


class AstPaperPlanParser:
    """Parse the paper's generated Python into a typed plan without executing it."""

    def parse(
        self,
        *,
        task: BenchmarkTask,
        robots: Sequence[RobotSpec],
        artifacts: PaperPlannerArtifacts,
    ) -> TaskPlan:
        parser = _PaperAstParser(
            source=_extract_python_source(artifacts.code_solution),
            robot_names=tuple(robot.name for robot in robots),
        )
        phases = parser.parse()
        if not phases:
            raise PaperPlanParseError(
                f"No executable actions were found in generated code for task {task.instruction!r}"
            )
        return TaskPlan(
            phases=tuple(phases),
            planner_name="paper-ast-parser",
            notes=artifacts.allocation_solution.strip() or None,
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


def _extract_python_source(text: str) -> str:
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return max(code_blocks, key=len).strip()
    return text.strip()


def _normalize_allocation_code_examples(text: str) -> str:
    return text.replace(
        "def throw_fork_in_trash():",
        "def throw_fork_in_trash(robot_list):",
    )


class _PaperAstParser:
    def __init__(self, *, source: str, robot_names: tuple[str, ...]) -> None:
        self.source = source
        self.robot_names = robot_names
        try:
            self.module = ast.parse(source)
        except SyntaxError as exc:
            raise PaperPlanParseError(f"Generated code is not valid Python: {exc.msg}") from exc
        self.functions = {
            node.name: node
            for node in self.module.body
            if isinstance(node, ast.FunctionDef)
        }

    def parse(self) -> list[PlanPhase]:
        phases: list[PlanPhase] = []
        thread_calls: dict[str, ast.Call] = {}
        active_phase_subtasks: list[PlanSubTask] = []

        for statement in self.module.body:
            if isinstance(statement, ast.FunctionDef):
                continue

            thread_assignment = self._thread_assignment(statement)
            if thread_assignment is not None:
                name, call = thread_assignment
                thread_calls[name] = call
                continue

            started_thread = self._thread_start_name(statement)
            if started_thread is not None:
                thread_call = thread_calls.get(started_thread)
                if thread_call is None:
                    raise PaperPlanParseError(f"Thread {started_thread!r} is started before assignment")
                actions = self._actions_from_thread_call(thread_call)
                if actions:
                    active_phase_subtasks.append(PlanSubTask(actions=tuple(actions)))
                continue

            if self._is_thread_join(statement):
                if active_phase_subtasks:
                    phases.append(PlanPhase(subtasks=tuple(active_phase_subtasks)))
                    active_phase_subtasks.clear()
                continue

            actions = self._actions_from_statement(statement, bindings={})
            if actions:
                if active_phase_subtasks:
                    active_phase_subtasks.append(PlanSubTask(actions=tuple(actions)))
                else:
                    phases.append(PlanPhase(actions=tuple(actions)))

        if active_phase_subtasks:
            phases.append(PlanPhase(subtasks=tuple(active_phase_subtasks)))

        return phases

    def _actions_from_statement(
        self,
        statement: ast.stmt,
        *,
        bindings: Mapping[str, object],
    ) -> list[ActionRequest]:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            return self._actions_from_call(statement.value, bindings=bindings)
        if isinstance(statement, ast.Pass):
            return []
        return []

    def _actions_from_call(
        self,
        call: ast.Call,
        *,
        bindings: Mapping[str, object],
    ) -> list[ActionRequest]:
        func_name = self._call_name(call)
        if func_name is None:
            return []

        if func_name in PAPER_ACTION_NAMES:
            return [self._action_from_skill_call(func_name, call, bindings=bindings)]

        if func_name in self.functions:
            return self._actions_from_function(func_name, call.args, bindings=bindings)

        if func_name in {"print", "sleep"}:
            return []

        return []

    def _actions_from_function(
        self,
        function_name: str,
        call_args: Sequence[ast.expr],
        *,
        bindings: Mapping[str, object],
    ) -> list[ActionRequest]:
        function = self.functions[function_name]
        parameters = [argument.arg for argument in function.args.args]
        if len(call_args) != len(parameters):
            raise PaperPlanParseError(
                f"Function {function_name!r} expected {len(parameters)} arguments, got {len(call_args)}"
            )
        local_bindings = dict(bindings)
        for parameter, value in zip(parameters, call_args):
            local_bindings[parameter] = self._resolve_value(value, bindings=bindings)

        actions: list[ActionRequest] = []
        for statement in function.body:
            actions.extend(self._actions_from_statement(statement, bindings=local_bindings))
        return actions

    def _actions_from_thread_call(self, thread_call: ast.Call) -> list[ActionRequest]:
        target_name = None
        target_args: list[ast.expr] = []
        for keyword in thread_call.keywords:
            if keyword.arg == "target" and isinstance(keyword.value, ast.Name):
                target_name = keyword.value.id
            if keyword.arg == "args":
                target_args = self._thread_args(keyword.value)

        if target_name is None:
            raise PaperPlanParseError("Thread target is missing")
        if target_name not in self.functions:
            raise PaperPlanParseError(f"Thread target {target_name!r} is not defined")

        return self._actions_from_function(target_name, target_args, bindings={})

    def _action_from_skill_call(
        self,
        skill_name: str,
        call: ast.Call,
        *,
        bindings: Mapping[str, object],
    ) -> ActionRequest:
        if not call.args:
            raise PaperPlanParseError(f"{skill_name} is missing its robot argument")

        robots = self._resolve_robot_value(call.args[0], bindings=bindings)
        object_name = None
        receptacle_name = None
        if len(call.args) >= 2:
            object_name = self._resolve_text_value(call.args[1], bindings=bindings)
        if len(call.args) >= 3:
            receptacle_name = self._resolve_text_value(call.args[2], bindings=bindings)
        return ActionRequest(
            robots=robots,
            skill=skill_name,
            object_name=object_name,
            receptacle_name=receptacle_name,
        )

    def _resolve_value(
        self,
        expression: ast.expr,
        *,
        bindings: Mapping[str, object],
    ) -> object:
        if isinstance(expression, ast.Subscript) and isinstance(expression.value, ast.Name):
            if expression.value.id == "robots":
                index = self._resolve_index(expression.slice, bindings=bindings)
                try:
                    return self.robot_names[index]
                except IndexError as exc:
                    raise PaperPlanParseError(f"Robot index {index} is out of range") from exc
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Name):
            if expression.id in bindings:
                return bindings[expression.id]
            raise PaperPlanParseError(f"Unknown name {expression.id!r} in generated code")
        if isinstance(expression, (ast.List, ast.Tuple)):
            return tuple(self._resolve_value(element, bindings=bindings) for element in expression.elts)
        if isinstance(expression, ast.Subscript):
            collection = self._resolve_value(expression.value, bindings=bindings)
            index = self._resolve_index(expression.slice, bindings=bindings)
            try:
                return collection[index]
            except (IndexError, TypeError) as exc:
                raise PaperPlanParseError("Generated code references an invalid list index") from exc
        if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
            if expression.value.id == "time" and expression.attr == "sleep":
                return "sleep"
        if isinstance(expression, ast.Call):
            return self._call_name(expression)
        raise PaperPlanParseError(
            f"Unsupported expression type in generated code: {type(expression).__name__}"
        )

    def _resolve_robot_value(
        self,
        expression: ast.expr,
        *,
        bindings: Mapping[str, object],
    ) -> tuple[str, ...]:
        if isinstance(expression, ast.Subscript) and isinstance(expression.value, ast.Name):
            if expression.value.id == "robots":
                index = self._resolve_index(expression.slice, bindings=bindings)
                try:
                    return (self.robot_names[index],)
                except IndexError as exc:
                    raise PaperPlanParseError(f"Robot index {index} is out of range") from exc

        value = self._resolve_value(expression, bindings=bindings)
        if isinstance(value, str):
            return (value,)
        if isinstance(value, tuple):
            flattened: list[str] = []
            for item in value:
                if isinstance(item, str):
                    flattened.append(item)
                    continue
                if isinstance(item, tuple):
                    flattened.extend(str(part) for part in item)
                    continue
                flattened.append(str(item))
            return tuple(flattened)
        raise PaperPlanParseError("Robot argument does not resolve to a robot or robot team")

    def _resolve_text_value(
        self,
        expression: ast.expr,
        *,
        bindings: Mapping[str, object],
    ) -> str:
        value = self._resolve_value(expression, bindings=bindings)
        if isinstance(value, str):
            return value
        raise PaperPlanParseError("Object argument does not resolve to a string")

    def _resolve_index(
        self,
        expression: ast.expr,
        *,
        bindings: Mapping[str, object],
    ) -> int:
        value = self._resolve_value(expression, bindings=bindings)
        if isinstance(value, int):
            return value
        raise PaperPlanParseError("List index does not resolve to an integer")

    def _thread_assignment(self, statement: ast.stmt) -> tuple[str, ast.Call] | None:
        if not isinstance(statement, ast.Assign):
            return None
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            return None
        if not isinstance(statement.value, ast.Call):
            return None
        call_name = self._call_name(statement.value)
        if call_name != "Thread":
            return None
        return statement.targets[0].id, statement.value

    def _thread_start_name(self, statement: ast.stmt) -> str | None:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            return None
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "start":
            return None
        if not isinstance(call.func.value, ast.Name):
            return None
        return call.func.value.id

    def _is_thread_join(self, statement: ast.stmt) -> bool:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            return False
        call = statement.value
        return isinstance(call.func, ast.Attribute) and call.func.attr == "join"

    def _thread_args(self, expression: ast.expr) -> list[ast.expr]:
        if isinstance(expression, ast.Tuple):
            return list(expression.elts)
        raise PaperPlanParseError("Thread args must be a tuple")

    def _call_name(self, call: ast.Call) -> str | None:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

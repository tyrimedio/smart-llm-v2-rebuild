from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from smart_llm_v2.benchmark.models import BenchmarkTask, GoalState

PAPER_BENCHMARK_TASK_COUNT = 36
REFERENCE_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "SMART-LLM" / "data" / "final_test"
)

_TASK_RE = re.compile(r'"task"\s*:\s*"(?P<value>.*?)"')
_ROBOT_LIST_RE = re.compile(r'"robot list"\s*:\s*\[(?P<value>[^\]]*)\]')
_TRANS_RE = re.compile(r'"trans"\s*:\s*(?P<value>-?\d+)')
_MAX_TRANS_RE = re.compile(r'"max_trans"\s*:\s*(?P<value>-?\d+)')


def load_reference_tasks(data_dir: Path | None = None) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    reference_dir = data_dir or REFERENCE_DATA_DIR

    for path in sorted(reference_dir.glob("FloorPlan*.json")):
        floor_plan = int(path.stem.removeprefix("FloorPlan"))
        lines = path.read_text().splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            if not raw_line.strip():
                continue
            tasks.append(
                _parse_reference_task_line(
                    floor_plan=floor_plan,
                    task_index=len(tasks) + 1,
                    raw_line=raw_line,
                    source_path=path,
                    source_line=line_number,
                )
            )

    return tasks


def summarize_reference_tasks(tasks: list[BenchmarkTask] | None = None) -> dict[int, int]:
    task_list = tasks or load_reference_tasks()
    counts = Counter(task.floor_plan for task in task_list)
    return dict(sorted(counts.items()))


def _parse_reference_task_line(
    *,
    floor_plan: int,
    task_index: int,
    raw_line: str,
    source_path: Path,
    source_line: int,
) -> BenchmarkTask:
    instruction = _extract_string_field(raw_line, _TASK_RE)
    robot_ids = _extract_robot_ids(raw_line)
    goal_states = _extract_goal_states(raw_line)
    transition_count = _extract_int(raw_line, _TRANS_RE, default=0)
    max_transition_count = _extract_int(raw_line, _MAX_TRANS_RE, default=0)

    return BenchmarkTask(
        floor_plan=floor_plan,
        task_index=task_index,
        instruction=instruction.strip(),
        robot_ids=robot_ids,
        goal_states=goal_states,
        transition_count=transition_count,
        max_transition_count=max_transition_count,
        source_path=source_path,
        source_line=source_line,
    )


def _extract_string_field(raw_line: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(raw_line)
    if match is None:
        raise ValueError(f"Could not parse task line: {raw_line}")
    return match.group("value")


def _extract_robot_ids(raw_line: str) -> tuple[int, ...]:
    match = _ROBOT_LIST_RE.search(raw_line)
    if match is None:
        return ()
    values = [chunk.strip() for chunk in match.group("value").split(",")]
    return tuple(int(value) for value in values if value)


def _extract_goal_states(raw_line: str) -> tuple[GoalState, ...]:
    object_states_literal = _extract_list_literal(raw_line, "object_states")
    if object_states_literal is None:
        return ()

    values = ast.literal_eval(object_states_literal)
    goal_states = []
    for value in values:
        goal_states.append(
            GoalState(
                name=_normalize_name(value["name"]),
                contains=tuple(_normalize_name(item) for item in value.get("contains", [])),
                state=_normalize_state(value.get("state")),
            )
        )
    return tuple(goal_states)


def _extract_int(raw_line: str, pattern: re.Pattern[str], *, default: int) -> int:
    match = pattern.search(raw_line)
    if match is None:
        return default
    return int(match.group("value"))


def _extract_list_literal(raw_line: str, field_name: str) -> str | None:
    key = f'"{field_name}"'
    key_index = raw_line.find(key)
    if key_index == -1:
        return None

    list_start = raw_line.find("[", key_index)
    if list_start == -1:
        return None

    depth = 0
    for index in range(list_start, len(raw_line)):
        char = raw_line[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return raw_line[list_start : index + 1]

    trans_index = raw_line.find(', "trans"', list_start)
    if trans_index == -1:
        trans_index = raw_line.find(',"trans"', list_start)
    if trans_index == -1:
        trans_index = len(raw_line)

    repaired = raw_line[list_start:trans_index].rstrip()
    return repaired if repaired.endswith("]") else f"{repaired}]"


def _normalize_name(value: str) -> str:
    return value.strip().rstrip(",.")


def _normalize_state(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if normalized == "NONE":
        return None
    return normalized

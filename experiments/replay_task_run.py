from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_llm_v2.agents.executor import BaselineExecutor, ExecutionRecord
from smart_llm_v2.agents.plan import ActionRequest, PlanPhase, PlanSubTask, TaskPlan
from smart_llm_v2.benchmark.metrics import compute_metrics
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.benchmark.tasks import load_reference_tasks
from smart_llm_v2.env.ai2thor_wrapper import Ai2ThorEnvironment
from smart_llm_v2.env.config import Ai2ThorConfig
from smart_llm_v2.robots import build_task_robot_team


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    task_runs = load_task_runs(run_dir / "task_runs.jsonl")

    if args.list_tasks:
        print_task_list(task_runs)
        return

    payload = select_task_run(task_runs=task_runs, task_id=args.task_id)
    plan = task_plan_from_payload(payload)
    if not plan.phases:
        raise SystemExit(f"{payload['task_id']} does not have a saved plan to replay.")

    task = resolve_task(str(payload["task_id"]))
    seed = args.seed if args.seed is not None else load_seed(run_dir / "config.json")
    replay_task_run(
        payload=payload,
        task=task,
        plan=plan,
        seed=seed,
        width=args.width,
        height=args.height,
        quality=args.quality,
        fullscreen=args.fullscreen,
        step_delay=args.step_delay,
        action_delay=args.action_delay,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=675)
    parser.add_argument("--quality", default="Very Low")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--step-delay", type=float, default=0.08)
    parser.add_argument("--action-delay", type=float, default=0.35)
    return parser.parse_args(argv)


def load_task_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Task run file not found: {path}")

    task_runs = []
    for line in path.read_text().splitlines():
        if line.strip():
            task_runs.append(json.loads(line))
    if not task_runs:
        raise SystemExit(f"No task runs found in {path}")
    return task_runs


def print_task_list(task_runs: Sequence[Mapping[str, Any]]) -> None:
    for payload in task_runs:
        metrics = payload["metrics"]
        status = "failed" if payload.get("error_message") else "ok"
        print(
            f"{payload['task_id']}\t{status}\t"
            f"SR={metrics['success_rate']:.2f}\t"
            f"Exe={metrics['executability']:.2f}\t"
            f"{payload['instruction']}"
        )


def select_task_run(
    *,
    task_runs: Sequence[Mapping[str, Any]],
    task_id: str | None,
) -> Mapping[str, Any]:
    if task_id is None:
        return task_runs[0]

    for payload in task_runs:
        if payload.get("task_id") == task_id:
            return payload
    raise SystemExit(f"Task {task_id!r} was not found in this run.")


def load_seed(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    seed = payload.get("seed")
    return seed if isinstance(seed, int) else None


def resolve_task(task_id: str) -> BenchmarkTask:
    for task in load_reference_tasks():
        if task.task_id == task_id:
            return task
    raise SystemExit(f"Task {task_id!r} was not found in the reference benchmark.")


def task_plan_from_payload(payload: Mapping[str, Any]) -> TaskPlan:
    phases = []
    for phase_payload in payload.get("plan", ()):
        subtasks = []
        for subtask_payload in phase_payload.get("subtasks", ()):
            actions = tuple(
                ActionRequest(
                    robots=tuple(action_payload["robots"]),
                    skill=action_payload["skill"],
                    object_name=action_payload.get("object_name"),
                    receptacle_name=action_payload.get("receptacle_name"),
                )
                for action_payload in subtask_payload.get("actions", ())
            )
            subtasks.append(
                PlanSubTask(
                    actions=actions,
                    assigned_robots=tuple(subtask_payload.get("assigned_robots", ())),
                    label=subtask_payload.get("label"),
                )
            )
        phases.append(PlanPhase(subtasks=tuple(subtasks), label=phase_payload.get("label")))

    planner_name = ":".join(
        value
        for value in (
            payload.get("planner_provider"),
            payload.get("planner_model"),
            payload.get("planner_profile_variant"),
        )
        if isinstance(value, str) and value
    )
    return TaskPlan(phases=tuple(phases), planner_name=planner_name or None)


def replay_task_run(
    *,
    payload: Mapping[str, Any],
    task: BenchmarkTask,
    plan: TaskPlan,
    seed: int | None,
    width: int,
    height: int,
    quality: str,
    fullscreen: bool,
    step_delay: float,
    action_delay: float,
) -> None:
    if step_delay < 0 or action_delay < 0:
        raise SystemExit("Delays must be non-negative.")

    robots = build_task_robot_team(task.robot_ids)
    environment = Ai2ThorEnvironment(
        config=Ai2ThorConfig(
            width=width,
            height=height,
            quality=quality,
            fullscreen=fullscreen,
            headless=False,
            step_delay_seconds=step_delay,
        )
    )
    environment.start(floor_plan=task.floor_plan, agent_count=len(robots), seed=seed)
    executor = BaselineExecutor(environment=environment, robots=robots)

    records: list[ExecutionRecord] = []
    try:
        print(f"Replaying {payload['task_id']}: {payload['instruction']}")
        print(f"Planner: {plan.planner_name or 'unknown'}")
        print(f"Seed: {seed}")
        for phase_index, phase in enumerate(plan.phases, start=1):
            if not phase.subtasks:
                continue
            print(f"Phase {phase_index}")
            for subtask_index, subtask in enumerate(phase.subtasks, start=1):
                print(
                    f"  Sub-task {subtask_index}: "
                    f"{', '.join(subtask.assigned_robots) or 'unassigned'}"
                )
                for action in subtask.actions:
                    print(f"    -> {format_action(action)}")
                    record = executor.execute_step(action)
                    records.append(record)
                    status = "ok" if record.succeeded else "failed"
                    detail = f": {record.error_message}" if record.error_message else ""
                    print(f"       {status}{detail}")
                    if action_delay:
                        time.sleep(action_delay)

        observed_objects = tuple(executor.scene_objects())
        metrics = compute_metrics(
            goal_states=task.goal_states,
            observed_objects=observed_objects,
            transition_count=plan.transition_count,
            transition_count_ground_truth=task.transition_count,
            max_transition_count=task.max_transition_count,
            successful_actions=sum(record.succeeded for record in records),
            total_actions=len(records),
        )
        print(
            "Replay metrics: "
            f"SR={metrics.success_rate:.2f}, "
            f"TCR={metrics.task_completion_rate:.2f}, "
            f"GCR={metrics.goal_condition_recall:.2f}, "
            f"RU={metrics.robot_utilization:.2f}, "
            f"Exe={metrics.executability:.2f}"
        )
    finally:
        executor.close()


def format_action(action: ActionRequest) -> str:
    args = [action.object_name]
    if action.receptacle_name is not None:
        args.append(action.receptacle_name)
    rendered_args = ", ".join(arg for arg in args if arg)
    return f"{'/'.join(action.robots)}.{action.skill}({rendered_args})"


if __name__ == "__main__":
    main()

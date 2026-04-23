from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import structlog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_llm_v2.agents.executor import BaselineExecutor
from smart_llm_v2.agents.model_profiles import ModelProfile, resolve_model_profile
from smart_llm_v2.agents.provider_factory import build_json_planner
from smart_llm_v2.benchmark.runner import BenchmarkRunner, TaskRunResult
from smart_llm_v2.benchmark.tasks import load_reference_tasks
from smart_llm_v2.env.ai2thor_wrapper import Ai2ThorEnvironment


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    logger = configure_logging()
    tasks = select_tasks(
        task_id=args.task_id,
        floor_plan=args.floor_plan,
        limit=args.limit,
    )
    if not tasks:
        raise SystemExit("No benchmark tasks matched the provided filters.")

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = resolve_planner_profile(args)
    planner = build_structured_planner(args=args, profile=profile)
    runner = BenchmarkRunner(
        planner=planner,
        executor_factory=lambda *, task, robots: build_executor(
            floor_plan=task.floor_plan,
            agent_count=len(robots),
            seed=args.seed,
            robots=robots,
        ),
    )

    logger.info(
        "structured_control_start",
        provider=profile.provider.value,
        model=profile.model,
        profile_variant=profile.variant.value,
        task_count=len(tasks),
        output_dir=str(output_dir),
    )
    summary = runner.run_benchmark(tasks)
    task_runs = summary.task_runs

    write_results(
        output_dir=output_dir,
        args=args,
        profile=profile,
        task_runs=task_runs,
        mean_metrics=summary.mean_metrics(),
    )
    logger.info(
        "structured_control_complete",
        provider=profile.provider.value,
        model=profile.model,
        profile_variant=profile.variant.value,
        task_count=len(task_runs),
        failed_task_count=summary.failed_task_count,
        mean_metrics=summary.mean_metrics(),
        output_dir=str(output_dir),
    )
    if summary.failed_task_count:
        raise SystemExit(
            f"{summary.failed_task_count} task runs failed. Results were written to {output_dir}."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("anthropic", "openai", "kimi"))
    parser.add_argument("--model")
    parser.add_argument("--profile-variant", choices=("symbolic", "multimodal"))
    parser.add_argument("--base-url")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-id")
    parser.add_argument("--floor-plan", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        parsed = parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").strip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, _parse_dotenv_value(value.strip())


def _parse_dotenv_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value.split(" #", 1)[0].strip()


def resolve_planner_profile(args: argparse.Namespace) -> ModelProfile:
    return resolve_model_profile(
        provider=args.provider,
        model=args.model,
        variant=args.profile_variant,
        base_url=args.base_url,
    )


def build_structured_planner(*, args: argparse.Namespace, profile: ModelProfile):
    return build_json_planner(
        profile,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


def configure_logging() -> structlog.stdlib.BoundLogger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()


def select_tasks(
    *,
    task_id: str | None,
    floor_plan: int | None,
    limit: int | None,
):
    tasks = load_reference_tasks()
    if task_id is not None:
        tasks = [task for task in tasks if task.task_id == task_id]
    if floor_plan is not None:
        tasks = [task for task in tasks if task.floor_plan == floor_plan]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def build_executor(
    *,
    floor_plan: int,
    agent_count: int,
    seed: int,
    robots,
) -> BaselineExecutor:
    environment = Ai2ThorEnvironment()
    environment.start(floor_plan=floor_plan, agent_count=agent_count, seed=seed)
    return BaselineExecutor(environment=environment, robots=robots)


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("results") / timestamp


def write_results(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    profile,
    task_runs: tuple[TaskRunResult, ...],
    mean_metrics: dict[str, float],
) -> None:
    config = {
        "provider": profile.provider.value,
        "model": profile.model,
        "profile_variant": profile.variant.value,
        "base_url": profile.base_url,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "task_id": args.task_id,
        "floor_plan": args.floor_plan,
        "limit": args.limit,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "task_count": len(task_runs),
                "failed_task_count": sum(task_run.error_message is not None for task_run in task_runs),
                "mean_metrics": mean_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    with (output_dir / "task_runs.jsonl").open("w") as handle:
        for task_run in task_runs:
            handle.write(json.dumps(task_run_payload(task_run), sort_keys=True) + "\n")


def task_run_payload(task_run: TaskRunResult) -> dict[str, object]:
    return {
        "task_id": task_run.task.task_id,
        "instruction": task_run.task.instruction,
        "planner_name": task_run.plan.planner_name,
        "planner_provider": task_run.planner_provider,
        "planner_model": task_run.planner_model,
        "planner_profile_variant": task_run.planner_profile_variant,
        "planner_usage": dict(task_run.planner_usage or {}),
        "error_message": task_run.error_message,
        "transition_count": task_run.execution.transition_count,
        "metrics": {
            "success_rate": task_run.metrics.success_rate,
            "task_completion_rate": task_run.metrics.task_completion_rate,
            "goal_condition_recall": task_run.metrics.goal_condition_recall,
            "robot_utilization": task_run.metrics.robot_utilization,
            "executability": task_run.metrics.executability,
        },
        "plan": [
            {
                "label": phase.label,
                "subtasks": [
                    {
                        "label": subtask.label,
                        "assigned_robots": list(subtask.assigned_robots),
                        "actions": [
                            {
                                "robots": list(action.robots),
                                "skill": action.skill,
                                "object_name": action.object_name,
                                "receptacle_name": action.receptacle_name,
                            }
                            for action in subtask.actions
                        ],
                    }
                    for subtask in phase.subtasks
                ],
            }
            for phase in task_run.plan.phases
        ],
        "records": [
            {
                "robots": list(record.request.robots),
                "skill": record.request.skill,
                "object_name": record.request.object_name,
                "receptacle_name": record.request.receptacle_name,
                "succeeded": record.succeeded,
                "error_message": record.error_message,
            }
            for record in task_run.execution.records
        ],
    }


if __name__ == "__main__":
    main()

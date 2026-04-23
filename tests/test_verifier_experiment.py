from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from smart_llm_v2.agents.executor import ExecutionReport
from smart_llm_v2.agents.plan import ActionRequest, TaskPlan
from smart_llm_v2.agents.verifier import VerificationIssue
from smart_llm_v2.benchmark.metrics import TaskMetrics
from smart_llm_v2.benchmark.models import BenchmarkTask, GoalState
from smart_llm_v2.benchmark.runner import TaskRunResult
from smart_llm_v2.robots import build_task_robot_team


def _load_experiment_module():
    try:
        import structlog
    except ModuleNotFoundError:
        structlog = SimpleNamespace(
            stdlib=SimpleNamespace(BoundLogger=object, LoggerFactory=object),
            processors=SimpleNamespace(
                add_log_level=object(),
                TimeStamper=lambda fmt: object(),
                JSONRenderer=lambda: object(),
            ),
            make_filtering_bound_logger=lambda level: object(),
            configure=lambda **kwargs: None,
            get_logger=lambda: object(),
        )
        sys.modules["structlog"] = structlog

    path = Path(__file__).resolve().parents[1] / "experiments" / "03_add_verifier.py"
    spec = importlib.util.spec_from_file_location("verifier_experiment", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_task_run_payload_includes_verifier_metadata() -> None:
    module = _load_experiment_module()
    task = BenchmarkTask(
        floor_plan=1,
        task_index=1,
        instruction="Turn on the laptop",
        robot_ids=(24,),
        goal_states=(GoalState(name="Laptop", state="ON"),),
    )
    plan = TaskPlan.sequential(
        ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="Laptop"),
        planner_name="kimi:kimi-k2.6:symbolic",
    )
    run = TaskRunResult(
        task=task,
        robots=tuple(build_task_robot_team(task.robot_ids)),
        plan=plan,
        execution=ExecutionReport(
            plan=plan,
            records=(),
            observed_objects=(),
            transition_count=0,
            successful_actions=0,
            total_actions=0,
        ),
        metrics=TaskMetrics(
            success_rate=0.0,
            task_completion_rate=0.0,
            goal_condition_recall=0.0,
            robot_utilization=0.0,
            executability=0.0,
        ),
        planner_provider="kimi",
        planner_model="kimi-k2.6",
        planner_usage={"total_tokens": 10},
        planner_profile_variant="symbolic",
        verifier_provider="kimi",
        verifier_model="kimi-k2.6",
        verifier_usage={"total_tokens": 7},
        verification_issues=(
            VerificationIssue(
                code="unsupported_skill",
                message="robot1 cannot execute OpenObject",
                phase_index=0,
                action_index=1,
            ),
        ),
        error_message="PlanVerificationError: robot1 cannot execute OpenObject",
    )

    payload = module.task_run_payload(run)

    assert payload["verifier_provider"] == "kimi"
    assert payload["verifier_model"] == "kimi-k2.6"
    assert payload["verifier_usage"] == {"total_tokens": 7}
    assert payload["verification_issues"] == [
        {
            "code": "unsupported_skill",
            "message": "robot1 cannot execute OpenObject",
            "source": "deterministic",
            "phase_index": 0,
            "action_index": 1,
        }
    ]


def test_verifier_experiment_wires_runner_with_plan_verifier(monkeypatch, tmp_path) -> None:
    module = _load_experiment_module()
    task = BenchmarkTask(
        floor_plan=1,
        task_index=1,
        instruction="Turn on the laptop",
        robot_ids=(24,),
        goal_states=(GoalState(name="Laptop", state="ON"),),
    )
    summary = SimpleNamespace(
        task_runs=(),
        failed_task_count=0,
        mean_metrics=lambda: {
            "success_rate": 0.0,
            "task_completion_rate": 0.0,
            "goal_condition_recall": 0.0,
            "robot_utilization": 0.0,
            "executability": 0.0,
        },
    )
    runner_init: dict[str, object] = {}

    monkeypatch.setattr(module, "load_dotenv", lambda path: None)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            task_id=None,
            floor_plan=None,
            limit=None,
            output_dir=tmp_path,
            provider=None,
            model=None,
            profile_variant=None,
            base_url=None,
            max_tokens=None,
            temperature=None,
            verifier_max_tokens=None,
            verifier_temperature=None,
            seed=0,
        ),
    )
    monkeypatch.setattr(module, "select_tasks", lambda **kwargs: [task])
    monkeypatch.setattr(module, "resolve_output_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        module,
        "resolve_planner_profile",
        lambda args: SimpleNamespace(
            provider=SimpleNamespace(value="kimi"),
            model="kimi-k2.6",
            variant=SimpleNamespace(value="symbolic"),
            base_url="https://api.moonshot.ai/v1",
        ),
    )
    monkeypatch.setattr(module, "build_json_planner", lambda *args, **kwargs: "planner")
    monkeypatch.setattr(module, "build_plan_verifier", lambda *args, **kwargs: "verifier")
    monkeypatch.setattr(module, "build_executor", lambda **kwargs: object())
    monkeypatch.setattr(module, "write_results", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "configure_logging",
        lambda: SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    class FakeRunner:
        def __init__(self, *, planner, verifier, executor_factory) -> None:
            runner_init["planner"] = planner
            runner_init["verifier"] = verifier
            runner_init["executor_factory"] = executor_factory

        def run_benchmark(self, tasks):
            assert tasks == [task]
            return summary

    monkeypatch.setattr(module, "BenchmarkRunner", FakeRunner)

    module.main()

    assert runner_init["planner"] == "planner"
    assert runner_init["verifier"] == "verifier"


def test_write_results_summarizes_verification_rejections(tmp_path) -> None:
    module = _load_experiment_module()
    task = BenchmarkTask(
        floor_plan=1,
        task_index=1,
        instruction="Turn on the laptop",
        robot_ids=(24,),
    )
    run = TaskRunResult.failed(
        task=task,
        robots=tuple(build_task_robot_team(task.robot_ids)),
        error_message="PlanVerificationError: rejected",
        verification_issues=(
            VerificationIssue(code="semantic_gap", message="Missing navigation"),
        ),
    )

    module.write_results(
        output_dir=tmp_path,
        args=SimpleNamespace(
            max_tokens=None,
            temperature=None,
            verifier_max_tokens=None,
            verifier_temperature=None,
            seed=0,
            task_id=None,
            floor_plan=None,
            limit=None,
        ),
        profile=SimpleNamespace(
            provider=SimpleNamespace(value="kimi"),
            model="kimi-k2.6",
            variant=SimpleNamespace(value="symbolic"),
            base_url="https://api.moonshot.ai/v1",
        ),
        task_runs=(run,),
        mean_metrics={
            "success_rate": 0.0,
            "task_completion_rate": 0.0,
            "goal_condition_recall": 0.0,
            "robot_utilization": 0.0,
            "executability": 0.0,
        },
    )

    assert '"verification_rejection_count": 1' in (tmp_path / "summary.json").read_text()

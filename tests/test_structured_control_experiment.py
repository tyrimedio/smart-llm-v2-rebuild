from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from smart_llm_v2.agents.executor import ExecutionRecord, ExecutionReport
from smart_llm_v2.agents.openai_client import OpenAICompatibleToolUseJsonClient
from smart_llm_v2.agents.plan import ActionRequest, TaskPlan
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

    path = Path(__file__).resolve().parents[1] / "experiments" / "01_structured_control.py"
    spec = importlib.util.spec_from_file_location("structured_control_experiment", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_structured_control_resolves_provider_from_cli_args() -> None:
    module = _load_experiment_module()

    args = module.parse_args(
        [
            "--provider",
            "kimi",
            "--model",
            "kimi-k2.6",
            "--profile-variant",
            "multimodal",
            "--base-url",
            "https://proxy.example/v1",
        ]
    )
    profile = module.resolve_planner_profile(args)
    planner = module.build_structured_planner(args=args, profile=profile)

    assert profile.provider.value == "kimi"
    assert profile.model == "kimi-k2.6"
    assert profile.variant.value == "multimodal"
    assert profile.base_url == "https://proxy.example/v1"
    assert planner.profile == profile
    assert isinstance(planner.client, OpenAICompatibleToolUseJsonClient)


def test_structured_control_profile_defaults_come_from_env(monkeypatch) -> None:
    module = _load_experiment_module()
    monkeypatch.setenv("SMART_LLM_V2_PROVIDER", "openai")
    monkeypatch.setenv("SMART_LLM_V2_MODEL", "gpt-5.4")
    monkeypatch.setenv("SMART_LLM_V2_PROFILE_VARIANT", "multimodal")

    args = module.parse_args([])
    profile = module.resolve_planner_profile(args)

    assert profile.provider.value == "openai"
    assert profile.model == "gpt-5.4"
    assert profile.variant.value == "multimodal"
    assert profile.vision_enabled is True


def test_load_dotenv_sets_missing_environment_values(monkeypatch, tmp_path) -> None:
    module = _load_experiment_module()
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "# local keys",
                "MOONSHOT_API_KEY=moonshot-local",
                "OPENAI_API_KEY='openai-local'",
                'ANTHROPIC_API_KEY="anthropic-local"',
                "SMART_LLM_V2_PROVIDER=kimi # cheap first",
                "export SMART_LLM_V2_PROFILE_VARIANT=symbolic",
            ]
        )
    )
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SMART_LLM_V2_PROVIDER", raising=False)
    monkeypatch.delenv("SMART_LLM_V2_PROFILE_VARIANT", raising=False)

    module.load_dotenv(dotenv_path)

    assert module.os.environ["MOONSHOT_API_KEY"] == "moonshot-local"
    assert module.os.environ["OPENAI_API_KEY"] == "openai-local"
    assert module.os.environ["ANTHROPIC_API_KEY"] == "anthropic-local"
    assert module.os.environ["SMART_LLM_V2_PROVIDER"] == "kimi"
    assert module.os.environ["SMART_LLM_V2_PROFILE_VARIANT"] == "symbolic"


def test_load_dotenv_does_not_override_exported_values(monkeypatch, tmp_path) -> None:
    module = _load_experiment_module()
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("MOONSHOT_API_KEY=from-file\n")
    monkeypatch.setenv("MOONSHOT_API_KEY", "from-shell")

    module.load_dotenv(dotenv_path)

    assert module.os.environ["MOONSHOT_API_KEY"] == "from-shell"


def test_task_run_payload_includes_provider_model_and_usage_metadata() -> None:
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
        planner_name="anthropic:claude-opus-4-7:symbolic",
    )
    run = TaskRunResult(
        task=task,
        robots=tuple(build_task_robot_team(task.robot_ids)),
        plan=plan,
        execution=ExecutionReport(
            plan=plan,
            records=(ExecutionRecord(request=plan.phases[0].actions[0], succeeded=True),),
            observed_objects=({"name": "Laptop|0", "isToggled": True},),
            transition_count=0,
            successful_actions=1,
            total_actions=1,
        ),
        metrics=TaskMetrics(
            success_rate=1.0,
            task_completion_rate=1.0,
            goal_condition_recall=1.0,
            robot_utilization=1.0,
            executability=1.0,
        ),
        planner_provider="anthropic",
        planner_model="claude-opus-4-7",
        planner_usage={"input_tokens": 10, "output_tokens": 5},
        planner_profile_variant="symbolic",
    )

    payload = module.task_run_payload(run)

    assert payload["planner_provider"] == "anthropic"
    assert payload["planner_model"] == "claude-opus-4-7"
    assert payload["planner_profile_variant"] == "symbolic"
    assert payload["planner_usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert payload["plan"][0]["subtasks"] == [
        {
            "label": None,
            "assigned_robots": ["robot1"],
            "actions": [
                {
                    "robots": ["robot1"],
                    "skill": "SwitchOn",
                    "object_name": "Laptop",
                    "receptacle_name": None,
                }
            ],
        }
    ]
    assert "verifier_provider" not in payload
    assert "verification_issues" not in payload


def test_structured_control_writes_results_before_reporting_failed_tasks(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_experiment_module()
    task = BenchmarkTask(
        floor_plan=1,
        task_index=1,
        instruction="Turn on the laptop",
        robot_ids=(24,),
        goal_states=(GoalState(name="Laptop", state="ON"),),
    )
    run = TaskRunResult(
        task=task,
        robots=tuple(build_task_robot_team(task.robot_ids)),
        plan=TaskPlan(phases=()),
        execution=ExecutionReport(
            plan=TaskPlan(phases=()),
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
        error_message="RuntimeError: boom",
    )
    summary = SimpleNamespace(
        task_runs=(run,),
        failed_task_count=1,
        mean_metrics=lambda: {
            "success_rate": 0.0,
            "task_completion_rate": 0.0,
            "goal_condition_recall": 0.0,
            "robot_utilization": 0.0,
            "executability": 0.0,
        },
    )
    writes: list[dict[str, object]] = []
    loaded_dotenv_paths: list[Path] = []

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
            seed=0,
        ),
    )
    monkeypatch.setattr(module, "load_dotenv", lambda path: loaded_dotenv_paths.append(path))
    monkeypatch.setattr(module, "select_tasks", lambda **kwargs: [task])
    monkeypatch.setattr(module, "resolve_output_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        module,
        "resolve_planner_profile",
        lambda args: SimpleNamespace(
            provider=SimpleNamespace(value="anthropic"),
            model="claude-opus-4-7",
            variant=SimpleNamespace(value="symbolic"),
            base_url=None,
        ),
    )
    monkeypatch.setattr(module, "build_structured_planner", lambda *, args, profile: object())
    monkeypatch.setattr(module, "build_executor", lambda **kwargs: object())
    monkeypatch.setattr(
        module,
        "configure_logging",
        lambda: SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    class FakeRunner:
        def __init__(self, *, planner, executor_factory) -> None:
            self.planner = planner
            self.executor_factory = executor_factory

        def run_benchmark(self, tasks):
            assert tasks == [task]
            return summary

    monkeypatch.setattr(module, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(module, "write_results", lambda **kwargs: writes.append(kwargs))

    with pytest.raises(SystemExit, match="1 task runs failed"):
        module.main()

    assert len(writes) == 1
    assert writes[0]["task_runs"] == summary.task_runs
    assert loaded_dotenv_paths == [module.PROJECT_ROOT / ".env"]

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_replay_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "replay_task_run.py"
    spec = importlib.util.spec_from_file_location("replay_task_run", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_task_plan_from_payload_rebuilds_saved_subtasks() -> None:
    module = _load_replay_module()
    payload = {
        "planner_provider": "kimi",
        "planner_model": "kimi-k2.6",
        "planner_profile_variant": "symbolic",
        "plan": [
            {
                "label": "first",
                "subtasks": [
                    {
                        "label": "lights",
                        "assigned_robots": ["robot1"],
                        "actions": [
                            {
                                "robots": ["robot1"],
                                "skill": "GoToObject",
                                "object_name": "LightSwitch",
                                "receptacle_name": None,
                            },
                            {
                                "robots": ["robot1"],
                                "skill": "SwitchOff",
                                "object_name": "LightSwitch",
                                "receptacle_name": None,
                            },
                        ],
                    }
                ],
            }
        ],
    }

    plan = module.task_plan_from_payload(payload)

    assert plan.planner_name == "kimi:kimi-k2.6:symbolic"
    assert len(plan.phases) == 1
    assert plan.phases[0].label == "first"
    assert plan.phases[0].subtasks[0].label == "lights"
    assert plan.phases[0].subtasks[0].assigned_robots == ("robot1",)
    assert plan.phases[0].subtasks[0].actions[1].skill == "SwitchOff"


def test_select_task_run_uses_requested_task_id() -> None:
    module = _load_replay_module()
    task_runs = [
        {"task_id": "FloorPlan15:1"},
        {"task_id": "FloorPlan15:2"},
    ]

    assert module.select_task_run(task_runs=task_runs, task_id="FloorPlan15:2") == {
        "task_id": "FloorPlan15:2"
    }


def test_format_action_includes_receptacle_when_present() -> None:
    module = _load_replay_module()
    action = module.ActionRequest(
        robots=("robot1",),
        skill="PutObject",
        object_name="Mug",
        receptacle_name="CoffeeMachine",
    )

    assert module.format_action(action) == "robot1.PutObject(Mug, CoffeeMachine)"

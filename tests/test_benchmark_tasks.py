from smart_llm_v2.benchmark.tasks import (
    PAPER_BENCHMARK_TASK_COUNT,
    load_reference_tasks,
    summarize_reference_tasks,
)


def test_load_reference_tasks_reads_current_snapshot() -> None:
    tasks = load_reference_tasks()

    assert len(tasks) == PAPER_BENCHMARK_TASK_COUNT


def test_load_reference_tasks_handles_malformed_reference_lines() -> None:
    tasks = load_reference_tasks()
    task_by_instruction = {task.instruction: task for task in tasks}

    break_task = task_by_instruction["Break the Cellphone and Close the blinds"]
    assert break_task.floor_plan == 303
    assert break_task.goal_states[0].state == "BROKEN"
    assert break_task.goal_states[1].state == "CLOSED"

    place_task = task_by_instruction["Place the laptop on the bed"]
    assert place_task.goal_states[0].contains == ("Laptop",)


def test_load_reference_tasks_defaults_missing_metadata() -> None:
    tasks = load_reference_tasks()
    task_by_instruction = {task.instruction: task for task in tasks}

    lamp_task = task_by_instruction["Turn off floor lamp"]
    assert lamp_task.goal_states == ()
    assert lamp_task.transition_count == 0
    assert lamp_task.max_transition_count == 0


def test_reference_task_summary_matches_floor_plan_files() -> None:
    assert summarize_reference_tasks() == {
        6: 3,
        15: 6,
        21: 6,
        201: 5,
        209: 5,
        303: 6,
        414: 5,
    }

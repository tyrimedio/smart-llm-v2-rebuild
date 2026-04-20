from smart_llm_v2.robots import build_task_robot_team, get_reference_robot, load_reference_robots


def test_reference_robot_catalog_preserves_original_count() -> None:
    robots = load_reference_robots()

    assert len(robots) == 28
    assert get_reference_robot(24).can("SwitchOn")
    assert not get_reference_robot(24).can("PickupObject")


def test_build_task_robot_team_renames_runtime_robots() -> None:
    team = build_task_robot_team((24, 25, 27))

    assert [robot.name for robot in team] == ["robot1", "robot2", "robot3"]
    assert [robot.reference_id for robot in team] == [24, 25, 27]
    assert team[0].reference_name == "robot24"

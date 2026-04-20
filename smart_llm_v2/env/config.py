from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Ai2ThorConfig:
    height: int = 300
    width: int = 400
    quality: str = "Very Low"
    fullscreen: bool = False
    headless: bool = False
    local_executable_path: str | None = None
    snap_grid: bool = False
    grid_size: float = 0.5
    rotate_step_degrees: int = 20
    visibility_distance: int = 100
    field_of_view: int = 90
    navigation_goal_threshold: float = 0.25
    navigation_stall_threshold: int = 8
    add_third_party_camera: bool = False

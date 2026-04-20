from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any

from smart_llm_v2.env.config import Ai2ThorConfig
from smart_llm_v2.env.profiles import recommended_ai2thor_config
from smart_llm_v2.env.state_extractor import extract_scene_objects

@dataclass(frozen=True, slots=True)
class ActionOutcome:
    action: str
    succeeded: bool
    error_message: str = ""


class Ai2ThorEnvironment:
    def __init__(self, config: Ai2ThorConfig | None = None) -> None:
        self.config = config or recommended_ai2thor_config()
        self._controller: Any | None = None
        self._reachable_positions: list[dict[str, float]] = []

    def start(self, *, floor_plan: int, agent_count: int, seed: int | None = None) -> None:
        controller_class = _load_controller_class()
        self._controller = controller_class(
            height=self.config.height,
            width=self.config.width,
            quality=self.config.quality,
            fullscreen=self.config.fullscreen,
            headless=self.config.headless,
            local_executable_path=self.config.local_executable_path,
        )
        self._controller.reset(f"FloorPlan{floor_plan}")
        self._controller.step(
            dict(
                action="Initialize",
                agentMode="default",
                snapGrid=self.config.snap_grid,
                gridSize=self.config.grid_size,
                rotateStepDegrees=self.config.rotate_step_degrees,
                visibilityDistance=self.config.visibility_distance,
                fieldOfView=self.config.field_of_view,
                agentCount=agent_count,
            )
        )
        if self.config.add_third_party_camera:
            map_view = self._controller.step(action="GetMapViewCameraProperties")
            self._controller.step(action="AddThirdPartyCamera", **map_view.metadata["actionReturn"])
        self._reachable_positions = self._controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        self._teleport_agents_randomly(agent_count=agent_count, seed=seed)

    def stop(self) -> None:
        if self._controller is not None:
            self._controller.stop()
            self._controller = None
            self._reachable_positions = []

    def navigate_to_object(self, *, agent_id: int, object_name: str) -> ActionOutcome:
        object_id, object_center = self._resolve_object_with_center(object_name)
        target_position = self._closest_reachable_position(object_center)
        previous_distance = math.inf
        stall_count = 0

        while True:
            current_pose = self.agent_pose(agent_id=agent_id)
            distance_to_goal = _distance_2d(
                (current_pose["x"], current_pose["z"]),
                (target_position["x"], target_position["z"]),
            )
            if distance_to_goal <= self.config.navigation_goal_threshold:
                break

            if abs(previous_distance - distance_to_goal) < 0.2:
                stall_count += 1
            else:
                stall_count = 0

            if stall_count >= self.config.navigation_stall_threshold:
                return ActionOutcome(
                    action="GoToObject",
                    succeeded=False,
                    error_message=f"Navigation stalled before reaching {object_name}",
                )

            event = self._step(
                action="ObjectNavExpertAction",
                position=target_position,
                agentId=agent_id,
            )
            next_action = event.metadata["actionReturn"]
            if next_action is not None:
                self._step(action=next_action, agentId=agent_id, forceAction=True)
            previous_distance = distance_to_goal

        self._align_agent_to_object(agent_id=agent_id, object_center=object_center)
        return ActionOutcome(action="GoToObject", succeeded=True)

    def perform_action(
        self,
        *,
        agent_id: int,
        action_name: str,
        target_name: str | None = None,
    ) -> ActionOutcome:
        controller_action, parameters = self._action_parameters(
            action_name=action_name,
            target_name=target_name,
            agent_id=agent_id,
        )
        event = self._step(action=controller_action, agentId=agent_id, forceAction=True, **parameters)
        error_message = event.metadata.get("errorMessage", "")
        return ActionOutcome(
            action=action_name,
            succeeded=error_message == "",
            error_message=error_message,
        )

    def scene_objects(self) -> tuple[dict[str, object], ...]:
        raw_objects = self._require_controller().last_event.metadata["objects"]
        return extract_scene_objects(raw_objects)

    def agent_pose(self, *, agent_id: int) -> dict[str, float]:
        metadata = self._require_controller().last_event.events[agent_id].metadata["agent"]
        return {
            "x": metadata["position"]["x"],
            "y": metadata["position"]["y"],
            "z": metadata["position"]["z"],
            "rotation": metadata["rotation"]["y"],
            "horizon": self._require_controller().last_event.events[agent_id].metadata["agent"]["cameraHorizon"],
        }

    def _teleport_agents_randomly(self, *, agent_count: int, seed: int | None) -> None:
        positions = self._reachable_positions
        if not positions:
            return

        rng = random.Random(seed)
        for agent_id in range(agent_count):
            self._step(action="Teleport", position=rng.choice(positions), agentId=agent_id)

    def _resolve_object_with_center(self, object_name: str) -> tuple[str, dict[str, float]]:
        objects = self._require_controller().last_event.metadata["objects"]
        for obj in objects:
            object_id = obj["objectId"]
            if re.match(object_name, object_id):
                center = obj["axisAlignedBoundingBox"]["center"]
                if center != {"x": 0.0, "y": 0.0, "z": 0.0}:
                    return object_id, center
        raise ValueError(f"Object {object_name!r} not found in scene")

    def _resolve_nearest_object_id(self, object_name: str, *, agent_id: int) -> str:
        objects = self._require_controller().last_event.metadata["objects"]
        matches: list[tuple[float, str]] = []
        for obj in objects:
            object_id = obj["objectId"]
            if re.match(object_name, object_id):
                matches.append((float(obj.get("distance", math.inf)), object_id))
        if not matches:
            raise ValueError(f"Object {object_name!r} not found in scene")
        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    def _closest_reachable_position(self, object_center: dict[str, float]) -> dict[str, float]:
        if not self._reachable_positions:
            raise RuntimeError("Reachable positions are not initialized")

        return min(
            self._reachable_positions,
            key=lambda position: _distance_2d(
                (position["x"], position["z"]),
                (object_center["x"], object_center["z"]),
            ),
        )

    def _align_agent_to_object(self, *, agent_id: int, object_center: dict[str, float]) -> None:
        pose = self.agent_pose(agent_id=agent_id)
        robot_to_object = (
            object_center["x"] - pose["x"],
            object_center["z"] - pose["z"],
        )
        if robot_to_object == (0.0, 0.0):
            return

        heading = math.degrees(math.atan2(robot_to_object[0], robot_to_object[1])) % 360
        rotation_delta = heading - pose["rotation"]
        if rotation_delta > 0:
            self._step(action="RotateRight", degrees=abs(rotation_delta), agentId=agent_id)
        elif rotation_delta < 0:
            self._step(action="RotateLeft", degrees=abs(rotation_delta), agentId=agent_id)

    def _action_parameters(
        self,
        *,
        action_name: str,
        target_name: str | None,
        agent_id: int,
    ) -> tuple[str, dict[str, Any]]:
        if action_name == "ThrowObject":
            return action_name, {"moveMagnitude": 7}

        if action_name == "DropHandObject":
            return action_name, {}

        if target_name is None:
            raise ValueError(f"{action_name} requires a target object name")

        object_id = self._resolve_nearest_object_id(target_name, agent_id=agent_id)
        return action_name, {"objectId": object_id}

    def _require_controller(self) -> Any:
        if self._controller is None:
            raise RuntimeError("AI2-THOR environment has not been started")
        return self._controller

    def _step(self, **kwargs: Any) -> Any:
        return self._require_controller().step(**kwargs)


def _load_controller_class() -> Any:
    try:
        from ai2thor.controller import Controller
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ai2thor is not installed. Install it before running the baseline executor."
        ) from exc
    return Controller


def _distance_2d(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.dist(point_a, point_b)

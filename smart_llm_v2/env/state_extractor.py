from __future__ import annotations

from typing import Mapping, Sequence


OBSERVED_OBJECT_FIELDS = (
    "name",
    "objectId",
    "objectType",
    "distance",
    "mass",
    "isBroken",
    "isCooked",
    "isOpen",
    "isPickedUp",
    "isSliced",
    "isToggled",
    "receptacleObjectIds",
    "temperature",
)


def extract_scene_objects(
    raw_objects: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    scene_objects = []
    for raw_object in raw_objects:
        scene_object = {
            field: raw_object.get(field)
            for field in OBSERVED_OBJECT_FIELDS
            if field in raw_object
        }
        scene_object["name"] = (
            raw_object.get("name")
            or raw_object.get("objectId")
            or raw_object.get("objectType")
            or ""
        )
        scene_object["receptacleObjectIds"] = tuple(
            raw_object.get("receptacleObjectIds") or ()
        )
        scene_objects.append(scene_object)
    return tuple(scene_objects)

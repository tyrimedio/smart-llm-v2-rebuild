from __future__ import annotations

from types import SimpleNamespace

from smart_llm_v2.env.ai2thor_wrapper import Ai2ThorEnvironment


class RecordingController:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.last_event = SimpleNamespace(
            metadata={
                "objects": [
                    {"objectId": "Mug|A", "distance": 1.0},
                    {"objectId": "Mug|B", "distance": 2.0},
                ]
            },
            events=[
                SimpleNamespace(
                    metadata={
                        "objects": [
                            {"objectId": "Mug|A", "distance": 1.0},
                            {"objectId": "Mug|B", "distance": 2.0},
                        ]
                    }
                ),
                SimpleNamespace(
                    metadata={
                        "objects": [
                            {"objectId": "Mug|A", "distance": 3.0},
                            {"objectId": "Mug|B", "distance": 0.5},
                        ]
                    }
                ),
            ],
        )

    def step(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(metadata={"errorMessage": ""})


def test_perform_action_uses_agent_specific_object_distances() -> None:
    environment = Ai2ThorEnvironment()
    controller = RecordingController()
    environment._controller = controller

    environment.perform_action(
        agent_id=0,
        action_name="PickupObject",
        target_name="Mug",
    )
    environment.perform_action(
        agent_id=1,
        action_name="PickupObject",
        target_name="Mug",
    )

    assert controller.calls == [
        {
            "action": "PickupObject",
            "agentId": 0,
            "forceAction": True,
            "objectId": "Mug|A",
        },
        {
            "action": "PickupObject",
            "agentId": 1,
            "forceAction": True,
            "objectId": "Mug|B",
        },
    ]

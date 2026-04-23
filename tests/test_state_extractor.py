from smart_llm_v2.env.state_extractor import extract_scene_objects


def test_state_extractor_preserves_openable_flag_for_verifier() -> None:
    objects = extract_scene_objects(
        (
            {
                "objectId": "DiningTable|0",
                "objectType": "DiningTable",
                "openable": False,
                "isOpen": False,
            },
        )
    )

    assert objects[0]["openable"] is False

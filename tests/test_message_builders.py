from __future__ import annotations

from smart_llm_v2.agents.json_planner import JsonPlannerRequest, task_plan_json_schema
from smart_llm_v2.agents.message_builders import (
    build_anthropic_messages,
    build_anthropic_verification_messages,
    build_openai_chat_messages,
    build_openai_verification_messages,
    build_semantic_verification_prompt,
)
from smart_llm_v2.agents.model_profiles import resolve_model_profile
from smart_llm_v2.agents.plan import ActionRequest, TaskPlan
from smart_llm_v2.agents.planner import PlanningImage
from smart_llm_v2.agents.verifier import SemanticVerificationRequest
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import build_task_robot_team


def _request(*, provider: str, variant: str) -> JsonPlannerRequest:
    return JsonPlannerRequest(
        task=BenchmarkTask(
            floor_plan=1,
            task_index=1,
            instruction="Turn on the laptop",
            robot_ids=(24,),
        ),
        robots=tuple(build_task_robot_team((24,))),
        scene_objects=(),
        profile=resolve_model_profile(provider=provider, variant=variant),
        response_schema=task_plan_json_schema(),
        system_message="Plan with structured tool calls.",
        images=(PlanningImage(data=b"png-bytes", agent_id=0, label="robot1_view"),),
    )


def _semantic_request(*, provider: str, variant: str) -> SemanticVerificationRequest:
    return SemanticVerificationRequest(
        task=BenchmarkTask(
            floor_plan=1,
            task_index=1,
            instruction="Turn on the laptop",
            robot_ids=(24,),
        ),
        robots=tuple(build_task_robot_team((24,))),
        scene_objects=(),
        plan=TaskPlan.sequential(
            ActionRequest(robots=("robot1",), skill="SwitchOn", object_name="Laptop"),
            planner_name=f"{provider}:model:{variant}",
        ),
        images=(
            (PlanningImage(data=b"png-bytes", agent_id=0, label="robot1_view"),)
            if variant == "multimodal"
            else ()
        ),
    )


def test_symbolic_profiles_do_not_emit_image_parts() -> None:
    anthropic_messages = build_anthropic_messages(_request(provider="anthropic", variant="symbolic"))
    anthropic_content = anthropic_messages[0]["content"]

    assert all(block["type"] != "image" for block in anthropic_content)
    assert "observation_images" not in anthropic_content[-1]["text"]

    for provider in ("openai", "kimi"):
        messages = build_openai_chat_messages(_request(provider=provider, variant="symbolic"))
        content = messages[1]["content"]

        assert messages[0] == {"role": "system", "content": "Plan with structured tool calls."}
        assert all(part["type"] != "image_url" for part in content)
        assert "observation_images" not in content[0]["text"]


def test_anthropic_multimodal_messages_include_base64_image_blocks() -> None:
    messages = build_anthropic_messages(_request(provider="anthropic", variant="multimodal"))
    content = messages[0]["content"]

    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Observation image, label=robot1_view, agent_id=0"
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["data"]


def test_openai_multimodal_messages_include_image_detail() -> None:
    messages = build_openai_chat_messages(_request(provider="openai", variant="multimodal"))
    content = messages[1]["content"]
    image_part = content[2]

    assert messages[0] == {"role": "system", "content": "Plan with structured tool calls."}
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["detail"] == "low"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_kimi_multimodal_messages_use_base64_urls_without_detail() -> None:
    messages = build_openai_chat_messages(_request(provider="kimi", variant="multimodal"))
    content = messages[1]["content"]
    image_part = content[2]

    assert image_part["type"] == "image_url"
    assert "detail" not in image_part["image_url"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert "https://" not in image_part["image_url"]["url"]


def test_semantic_verification_prompt_includes_candidate_plan_context() -> None:
    prompt = build_semantic_verification_prompt(
        _semantic_request(provider="openai", variant="symbolic")
    )

    assert "submit_plan_verification tool exactly once" in prompt
    assert '"candidate_plan"' in prompt
    assert '"deterministic_checks_passed"' in prompt


def test_openai_verification_messages_include_multimodal_image_detail() -> None:
    request = _semantic_request(provider="openai", variant="multimodal")
    messages = build_openai_verification_messages(
        request,
        system_message="Verify with structured tool calls.",
        provider=resolve_model_profile(provider="openai", variant="multimodal").provider,
        vision_enabled=True,
        image_detail="low",
    )
    image_part = messages[1]["content"][2]

    assert messages[0] == {"role": "system", "content": "Verify with structured tool calls."}
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["detail"] == "low"


def test_anthropic_verification_messages_include_base64_images() -> None:
    messages = build_anthropic_verification_messages(
        _semantic_request(provider="anthropic", variant="multimodal"),
        vision_enabled=True,
    )
    content = messages[0]["content"]

    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"
    assert content[-1]["text"].startswith("Review the candidate plan")

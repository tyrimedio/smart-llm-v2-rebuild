from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from smart_llm_v2.agents.json_planner import JsonPlannerRequest, task_plan_json_schema
from smart_llm_v2.agents.model_profiles import resolve_model_profile
from smart_llm_v2.agents.openai_client import (
    OpenAICompatibleSemanticVerifierClient,
    OpenAICompatiblePlanningError,
    OpenAICompatibleToolUseJsonClient,
)
from smart_llm_v2.agents.plan import ActionRequest, TaskPlan
from smart_llm_v2.agents.planner import PlanningImage
from smart_llm_v2.agents.verifier import PLAN_VERIFICATION_TOOL_NAME, SemanticVerificationRequest
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import build_task_robot_team


class FakeCompletionsAPI:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAIClient:
    def __init__(self, response: object) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletionsAPI(response))


def _request(*, provider: str, variant: str = "symbolic") -> JsonPlannerRequest:
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


def _semantic_request(*, provider: str, variant: str = "symbolic") -> SemanticVerificationRequest:
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


def _response(*, payload: dict[str, object], tool_name: str = "submit_task_plan") -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name=tool_name,
                                arguments=(
                                    '{"phases":[{"subtasks":[{"assigned_robots":["robot1"],"actions":[{"robots":["robot1"],"skill":"GoToObject","object_name":"Laptop"}]}]}]}'
                                    if not payload
                                    else json.dumps(payload)
                                ),
                            )
                        )
                    ]
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=18,
            completion_tokens=5,
            total_tokens=23,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )


def test_openai_client_forces_single_task_plan_tool_and_normalizes_usage() -> None:
    fake_client = FakeOpenAIClient(_response(payload={}))
    client = OpenAICompatibleToolUseJsonClient(
        provider="openai",
        model="gpt-5.4",
        api_key_env_var="OPENAI_API_KEY",
        client=fake_client,
        reasoning_effort="high",
    )

    result = client.complete(_request(provider="openai"))

    assert result.payload == {
        "phases": [
            {
                "subtasks": [
                    {
                        "assigned_robots": ["robot1"],
                        "actions": [
                            {
                                "robots": ["robot1"],
                                "skill": "GoToObject",
                                "object_name": "Laptop",
                            }
                        ],
                    }
                ]
            }
        ]
    }
    assert result.provider == "openai"
    assert result.model == "gpt-5.4"
    assert result.usage == {
        "prompt_tokens": 18,
        "completion_tokens": 5,
        "total_tokens": 23,
        "cached_tokens": 4,
        "reasoning_tokens": 2,
    }

    create_call = fake_client.chat.completions.calls[0]
    assert create_call["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_task_plan"},
    }
    assert create_call["parallel_tool_calls"] is False
    assert create_call["reasoning_effort"] == "high"
    assert create_call["messages"][0] == {
        "role": "system",
        "content": "Plan with structured tool calls.",
    }
    assert "submit_task_plan tool exactly once" in create_call["messages"][1]["content"][0]["text"]


def test_kimi_client_uses_same_transport_but_keeps_kimi_provider() -> None:
    fake_client = FakeOpenAIClient(_response(payload={}))
    client = OpenAICompatibleToolUseJsonClient(
        provider="kimi",
        model="kimi-k2.6",
        api_key_env_var="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.ai/v1",
        client=fake_client,
        temperature=1.0,
        top_p=0.95,
    )

    result = client.complete(_request(provider="kimi", variant="multimodal"))
    create_call = fake_client.chat.completions.calls[0]
    image_part = create_call["messages"][1]["content"][2]

    assert result.provider == "kimi"
    assert result.model == "kimi-k2.6"
    assert create_call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert "detail" not in image_part["image_url"]


def test_openai_client_rejects_prose_only_completions() -> None:
    fake_client = FakeOpenAIClient(
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content="Plan in prose"))]
        )
    )
    client = OpenAICompatibleToolUseJsonClient(
        provider="openai",
        model="gpt-5.4",
        api_key_env_var="OPENAI_API_KEY",
        client=fake_client,
    )

    with pytest.raises(OpenAICompatiblePlanningError, match="did not emit a tool call"):
        client.complete(_request(provider="openai"))


def test_openai_client_rejects_malformed_tool_json() -> None:
    fake_client = FakeOpenAIClient(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="submit_task_plan",
                                    arguments="{not-json",
                                )
                            )
                        ]
                    )
                )
            ]
        )
    )
    client = OpenAICompatibleToolUseJsonClient(
        provider="openai",
        model="gpt-5.4",
        api_key_env_var="OPENAI_API_KEY",
        client=fake_client,
    )

    with pytest.raises(OpenAICompatiblePlanningError, match="not valid JSON"):
        client.complete(_request(provider="openai"))


def test_openai_semantic_verifier_forces_verification_tool_and_normalizes_usage() -> None:
    fake_client = FakeOpenAIClient(
        _response(
            payload={
                "issues": [
                    {
                        "code": "semantic_gap",
                        "message": "The plan should navigate before toggling the laptop.",
                        "phase_index": 0,
                        "action_index": 0,
                    }
                ]
            },
            tool_name=PLAN_VERIFICATION_TOOL_NAME,
        )
    )
    client = OpenAICompatibleSemanticVerifierClient(
        provider="openai",
        model="gpt-5.4",
        api_key_env_var="OPENAI_API_KEY",
        client=fake_client,
        reasoning_effort="high",
        vision_enabled=True,
        image_detail="low",
    )

    result = client.review(_semantic_request(provider="openai", variant="multimodal"))

    assert result.provider == "openai"
    assert result.model == "gpt-5.4"
    assert result.issues[0].source == "semantic"
    assert result.usage == {
        "prompt_tokens": 18,
        "completion_tokens": 5,
        "total_tokens": 23,
        "cached_tokens": 4,
        "reasoning_tokens": 2,
    }
    create_call = fake_client.chat.completions.calls[0]
    assert create_call["tool_choice"] == {
        "type": "function",
        "function": {"name": PLAN_VERIFICATION_TOOL_NAME},
    }
    assert "submit_plan_verification tool exactly once" in create_call["messages"][1]["content"][0]["text"]
    assert create_call["messages"][1]["content"][2]["image_url"]["detail"] == "low"


def test_kimi_semantic_verifier_disables_thinking_for_specified_tool_choice() -> None:
    fake_client = FakeOpenAIClient(_response(payload={"issues": []}, tool_name=PLAN_VERIFICATION_TOOL_NAME))
    client = OpenAICompatibleSemanticVerifierClient(
        provider="kimi",
        model="kimi-k2.6",
        api_key_env_var="MOONSHOT_API_KEY",
        client=fake_client,
    )

    result = client.review(_semantic_request(provider="kimi"))

    assert result.provider == "kimi"
    assert fake_client.chat.completions.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }

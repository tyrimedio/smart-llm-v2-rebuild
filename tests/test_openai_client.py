from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from smart_llm_v2.agents.json_planner import JsonPlannerRequest, task_plan_json_schema
from smart_llm_v2.agents.model_profiles import resolve_model_profile
from smart_llm_v2.agents.openai_client import (
    OpenAICompatiblePlanningError,
    OpenAICompatibleToolUseJsonClient,
)
from smart_llm_v2.agents.planner import PlanningImage
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


def _response(*, payload: dict[str, object]) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="submit_task_plan",
                                arguments=(
                                    '{"phases":[{"actions":[{"robots":["robot1"],"skill":"GoToObject","object_name":"Laptop"}]}]}'
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
                "actions": [
                    {
                        "robots": ["robot1"],
                        "skill": "GoToObject",
                        "object_name": "Laptop",
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

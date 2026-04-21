from __future__ import annotations

import os

import pytest

from smart_llm_v2.agents.anthropic_client import (
    AnthropicPlanningError,
    AnthropicToolUseJsonClient,
    TASK_PLAN_TOOL_NAME,
    _task_plan_tool,
)
from smart_llm_v2.agents.json_planner import JsonPlannerRequest, task_plan_json_schema
from smart_llm_v2.agents.model_profiles import resolve_model_profile
from smart_llm_v2.agents.provider_factory import build_planning_client
from smart_llm_v2.benchmark.models import BenchmarkTask
from smart_llm_v2.robots import build_task_robot_team


class FakeToolUseBlock:
    def __init__(self, *, name: str, payload: dict[str, object]) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = payload


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeMessage:
    def __init__(self, *, content, stop_reason: str | None = "tool_use") -> None:
        self.content = content
        self.stop_reason = stop_reason


class FakeMessagesAPI:
    def __init__(self, response: FakeMessage) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeAnthropicClient:
    def __init__(self, response: FakeMessage) -> None:
        self.messages = FakeMessagesAPI(response)


LIVE_ANTHROPIC_SMOKE_ENV_VAR = "SMART_LLM_V2_RUN_ANTHROPIC_SMOKE"


def _request() -> JsonPlannerRequest:
    return JsonPlannerRequest(
        task=BenchmarkTask(
            floor_plan=1,
            task_index=1,
            instruction="Turn on the laptop",
            robot_ids=(24,),
        ),
        robots=tuple(build_task_robot_team((24,))),
        scene_objects=(),
        profile=resolve_model_profile(provider="anthropic", variant="symbolic"),
        response_schema=task_plan_json_schema(),
        system_message="Plan with JSON.",
    )


def test_anthropic_client_forces_single_task_plan_tool() -> None:
    payload = {
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
    fake_client = FakeAnthropicClient(
        FakeMessage(content=[FakeToolUseBlock(name=TASK_PLAN_TOOL_NAME, payload=payload)])
    )
    client = AnthropicToolUseJsonClient(
        model="claude-opus-4-7",
        client=fake_client,
    )

    result = client.complete(_request())

    assert result.payload == payload
    assert result.provider == "anthropic"
    assert result.model == "claude-opus-4-7"
    create_call = fake_client.messages.calls[0]
    assert create_call["model"] == "claude-opus-4-7"
    assert create_call["tool_choice"] == {
        "type": "tool",
        "name": TASK_PLAN_TOOL_NAME,
        "disable_parallel_tool_use": True,
    }
    assert create_call["tools"][0]["name"] == TASK_PLAN_TOOL_NAME
    assert create_call["tools"][0]["strict"] is True
    message_text = create_call["messages"][0]["content"][-1]["text"]
    assert "submit_task_plan tool exactly once" in message_text


def test_default_anthropic_path_uses_auto_tool_choice_when_thinking_is_enabled() -> None:
    payload = {
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
    fake_client = FakeAnthropicClient(
        FakeMessage(content=[FakeToolUseBlock(name=TASK_PLAN_TOOL_NAME, payload=payload)])
    )
    profile = resolve_model_profile(provider="anthropic", variant="symbolic")
    client = build_planning_client(profile, client=fake_client)

    result = client.complete(_request())

    assert result.payload == payload
    create_call = fake_client.messages.calls[0]
    assert create_call["tool_choice"] == {"type": "auto"}
    assert create_call["thinking"] == {
        "type": "enabled",
        "budget_tokens": profile.thinking_budget_tokens,
    }


def test_anthropic_client_rejects_missing_tool_use_block() -> None:
    fake_client = FakeAnthropicClient(
        FakeMessage(
            content=[FakeTextBlock("Here is a plan in prose.")],
            stop_reason="end_turn",
        )
    )
    client = AnthropicToolUseJsonClient(
        model="claude-opus-4-7",
        client=fake_client,
    )

    with pytest.raises(AnthropicPlanningError, match="did not emit"):
        client.complete(_request())


def test_anthropic_client_rejects_repeated_task_plan_tool_use_blocks() -> None:
    first_payload = {
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
    second_payload = {
        "phases": [
            {
                "actions": [
                    {
                        "robots": ["robot1"],
                        "skill": "SwitchOn",
                        "object_name": "Laptop",
                    }
                ]
            }
        ]
    }
    fake_client = FakeAnthropicClient(
        FakeMessage(
            content=[
                FakeToolUseBlock(name=TASK_PLAN_TOOL_NAME, payload=first_payload),
                FakeToolUseBlock(name=TASK_PLAN_TOOL_NAME, payload=second_payload),
            ]
        )
    )
    client = AnthropicToolUseJsonClient(
        model="claude-opus-4-7",
        client=fake_client,
    )

    with pytest.raises(AnthropicPlanningError, match="expected exactly one"):
        client.complete(_request())


@pytest.mark.integration
def test_live_anthropic_smoke_can_emit_repeated_submit_task_plan_blocks(record_property) -> None:
    if os.environ.get(LIVE_ANTHROPIC_SMOKE_ENV_VAR) != "1":
        pytest.skip(f"set {LIVE_ANTHROPIC_SMOKE_ENV_VAR}=1 to run live Anthropic smoke tests")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    anthropic = pytest.importorskip("anthropic")
    profile = resolve_model_profile(provider="anthropic", variant="symbolic")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=profile.model,
        max_tokens=1024,
        system="Return plans through tool calls only.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Call submit_task_plan twice. "
                            "Use one tool call for the first valid standalone plan and a second "
                            "tool call for a different valid standalone plan. "
                            "Do not combine them into one tool call."
                        ),
                    }
                ],
            }
        ],
        tools=[_task_plan_tool(task_plan_json_schema(), strict=profile.tool_strict)],
        tool_choice={"type": "tool", "name": TASK_PLAN_TOOL_NAME},
    )

    tool_use_blocks = [
        block
        for block in response.content
        if getattr(block, "type", None) == "tool_use"
        and getattr(block, "name", None) == TASK_PLAN_TOOL_NAME
    ]
    record_property("submit_task_plan_block_count", len(tool_use_blocks))
    record_property("stop_reason", getattr(response, "stop_reason", None))

    if len(tool_use_blocks) < 2:
        pytest.xfail("This live smoke run did not reproduce repeated submit_task_plan calls")

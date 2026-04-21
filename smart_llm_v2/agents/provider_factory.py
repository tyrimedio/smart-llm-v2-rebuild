"""Factory helpers for provider-backed planners."""

from __future__ import annotations

from typing import Any

from smart_llm_v2.agents.anthropic_client import AnthropicToolUseJsonClient
from smart_llm_v2.agents.json_planner import JsonPlanner, JsonPlanningClient
from smart_llm_v2.agents.model_profiles import ModelProfile, Transport
from smart_llm_v2.agents.openai_client import OpenAICompatibleToolUseJsonClient


def build_planning_client(
    profile: ModelProfile,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    client: Any | None = None,
) -> JsonPlanningClient:
    resolved_max_tokens = profile.max_tokens if max_tokens is None else max_tokens
    resolved_temperature = profile.temperature if temperature is None else temperature

    if profile.transport is Transport.ANTHROPIC_MESSAGES:
        return AnthropicToolUseJsonClient(
            model=profile.model,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            top_p=profile.top_p,
            thinking_budget_tokens=profile.thinking_budget_tokens,
            client=client,
        )

    if profile.transport is Transport.OPENAI_CHAT_COMPLETIONS:
        return OpenAICompatibleToolUseJsonClient(
            provider=profile.provider.value,
            model=profile.model,
            api_key_env_var=profile.api_key_env_var,
            base_url=profile.base_url,
            max_completion_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            top_p=profile.top_p,
            reasoning_effort=profile.reasoning_effort,
            client=client,
        )

    raise AssertionError(f"Unhandled transport: {profile.transport}")


def build_json_planner(
    profile: ModelProfile,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    client: Any | None = None,
) -> JsonPlanner:
    return JsonPlanner(
        client=build_planning_client(
            profile,
            max_tokens=max_tokens,
            temperature=temperature,
            client=client,
        ),
        profile=profile,
    )

"""OpenAI-compatible chat-completions client for structured planning."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

from smart_llm_v2.agents.anthropic_client import TASK_PLAN_TOOL_NAME
from smart_llm_v2.agents.json_planner import JsonPlannerRequest, JsonPlanningResult
from smart_llm_v2.agents.message_builders import (
    build_openai_chat_messages,
    build_openai_verification_messages,
)
from smart_llm_v2.agents.model_profiles import Provider
from smart_llm_v2.agents.verifier import (
    DEFAULT_SEMANTIC_VERIFIER_SYSTEM_MESSAGE,
    PLAN_VERIFICATION_TOOL_NAME,
    SemanticVerificationRequest,
    SemanticVerificationResult,
    semantic_verification_json_schema,
    semantic_verification_result_from_mapping,
)


class OpenAICompatiblePlanningError(RuntimeError):
    """Raised when an OpenAI-compatible provider does not return a usable tool payload."""


@dataclass(frozen=True, slots=True)
class OpenAICompatibleToolUseJsonClient:
    provider: str
    model: str
    api_key_env_var: str
    base_url: str | None = None
    max_completion_tokens: int = 4096
    temperature: float = 0.0
    top_p: float | None = None
    reasoning_effort: str | None = None
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")

    def complete(self, request: JsonPlannerRequest) -> JsonPlanningResult:
        create_kwargs: dict[str, object] = {
            "model": self.model,
            "messages": build_openai_chat_messages(request),
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "tools": [_task_plan_tool(request.response_schema, strict=request.profile.tool_strict)],
            "tool_choice": {
                "type": "function",
                "function": {"name": TASK_PLAN_TOOL_NAME},
            },
            "parallel_tool_calls": False,
        }
        if self.top_p is not None:
            create_kwargs["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            create_kwargs["reasoning_effort"] = self.reasoning_effort

        response = self._client().chat.completions.create(**create_kwargs)
        payload = _extract_tool_payload(response)
        return JsonPlanningResult(
            payload=payload,
            provider=request.profile.provider.value,
            model=self.model,
            usage=_usage_mapping(getattr(response, "usage", None)),
        )

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "openai is not installed. Install it before using OpenAI or Kimi planning."
            ) from exc

        api_key = os.environ.get(self.api_key_env_var)
        kwargs: dict[str, object] = {"api_key": api_key}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleSemanticVerifierClient:
    provider: str
    model: str
    api_key_env_var: str
    base_url: str | None = None
    max_completion_tokens: int = 4096
    temperature: float = 0.0
    top_p: float | None = None
    reasoning_effort: str | None = None
    strict: bool = True
    vision_enabled: bool = False
    image_detail: str | None = None
    system_message: str = DEFAULT_SEMANTIC_VERIFIER_SYSTEM_MESSAGE
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")

    def review(self, request: SemanticVerificationRequest) -> SemanticVerificationResult:
        create_kwargs: dict[str, object] = {
            "model": self.model,
            "messages": build_openai_verification_messages(
                request,
                system_message=self.system_message,
                provider=Provider(self.provider),
                vision_enabled=self.vision_enabled,
                image_detail=self.image_detail,
            ),
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "tools": [
                _tool(
                    tool_name=PLAN_VERIFICATION_TOOL_NAME,
                    description=(
                        "Return the semantic verification report as structured JSON. "
                        "List only execution-critical issues that remain after deterministic checks."
                    ),
                    schema=semantic_verification_json_schema(),
                    strict=self.strict,
                )
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": PLAN_VERIFICATION_TOOL_NAME},
            },
            "parallel_tool_calls": False,
        }
        if self.top_p is not None:
            create_kwargs["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            create_kwargs["reasoning_effort"] = self.reasoning_effort

        response = self._client().chat.completions.create(**create_kwargs)
        payload = _extract_named_tool_payload(response, tool_name=PLAN_VERIFICATION_TOOL_NAME)
        return semantic_verification_result_from_mapping(
            payload,
            provider=self.provider,
            model=self.model,
            usage=_usage_mapping(getattr(response, "usage", None)),
        )

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "openai is not installed. Install it before using the semantic verifier."
            ) from exc

        api_key = os.environ.get(self.api_key_env_var)
        kwargs: dict[str, object] = {"api_key": api_key}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)


def _task_plan_tool(schema: Mapping[str, object], *, strict: bool) -> dict[str, object]:
    return _tool(
        tool_name=TASK_PLAN_TOOL_NAME,
        description=(
            "Return the complete task plan as structured JSON. Use phases for temporal "
            "layers, sub-tasks for parallel work inside a phase, and ordered actions "
            "inside each sub-task. Include only actions that match the provided robot "
            "skills and task constraints."
        ),
        schema=schema,
        strict=strict,
    )


def _extract_tool_payload(response: Any) -> Mapping[str, object]:
    return _extract_named_tool_payload(response, tool_name=TASK_PLAN_TOOL_NAME)


def _tool(
    *,
    tool_name: str,
    description: str,
    schema: Mapping[str, object],
    strict: bool,
) -> dict[str, object]:
    function: dict[str, object] = {
        "name": tool_name,
        "description": description,
        "parameters": dict(schema),
    }
    if strict:
        function["strict"] = True
    return {
        "type": "function",
        "function": function,
    }


def _extract_named_tool_payload(response: Any, *, tool_name: str) -> Mapping[str, object]:
    choices = getattr(response, "choices", None)
    if not choices:
        raise OpenAICompatiblePlanningError("Provider did not return any completion choices")

    message = getattr(choices[0], "message", None)
    tool_calls = getattr(message, "tool_calls", None) if message is not None else None
    if not tool_calls:
        raise OpenAICompatiblePlanningError(
            f"Provider {getattr(response, 'model', None)!r} did not emit a tool call"
        )
    if len(tool_calls) != 1:
        raise OpenAICompatiblePlanningError("Provider returned more than one tool call")

    tool_call = tool_calls[0]
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None)
    if name != tool_name:
        raise OpenAICompatiblePlanningError(
            f"Provider emitted unexpected tool {name!r} instead of {tool_name!r}"
        )
    arguments = getattr(function, "arguments", None)
    if not isinstance(arguments, str):
        raise OpenAICompatiblePlanningError("Provider tool call did not include JSON arguments")
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise OpenAICompatiblePlanningError("Provider tool call arguments were not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise OpenAICompatiblePlanningError("Provider returned a tool payload that is not a mapping")
    return payload


def _usage_mapping(usage: Any) -> Mapping[str, object] | None:
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        return dict(usage)

    data: dict[str, object] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            data[key] = value

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached_tokens = getattr(prompt_details, "cached_tokens", None)
        if cached_tokens is not None:
            data["cached_tokens"] = cached_tokens

    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details is not None:
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        if reasoning_tokens is not None:
            data["reasoning_tokens"] = reasoning_tokens

    return data or None

"""Anthropic client for the JSON-first planner path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from smart_llm_v2.agents.json_planner import JsonPlannerRequest, JsonPlanningResult
from smart_llm_v2.agents.message_builders import build_anthropic_messages

TASK_PLAN_TOOL_NAME = "submit_task_plan"


class AnthropicPlanningError(RuntimeError):
    """Raised when Claude does not return a usable structured task plan."""


@dataclass(frozen=True, slots=True)
class AnthropicToolUseJsonClient:
    model: str
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float | None = None
    thinking_budget_tokens: int | None = None
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")

    def complete(self, request: JsonPlannerRequest) -> JsonPlanningResult:
        create_kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": request.system_message,
            "messages": build_anthropic_messages(request),
            "tools": [_task_plan_tool(request.response_schema, strict=request.profile.tool_strict)],
            "tool_choice": _task_plan_tool_choice(
                thinking_enabled=self.thinking_budget_tokens is not None
            ),
        }
        if self.top_p is not None:
            create_kwargs["top_p"] = self.top_p
        if self.thinking_budget_tokens is not None:
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }

        response = self._client().messages.create(
            **create_kwargs,
        )
        payload = _extract_tool_input(response)
        if not isinstance(payload, Mapping):
            raise AnthropicPlanningError("Claude returned a tool payload that is not a mapping")
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
            import anthropic
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "anthropic is not installed. Install it before using the Anthropic JSON planner."
            ) from exc
        return anthropic.Anthropic()


def _task_plan_tool_choice(*, thinking_enabled: bool) -> dict[str, object]:
    # Anthropic rejects forced named tool choice when extended thinking is enabled.
    if thinking_enabled:
        return {"type": "auto"}
    return {
        "type": "tool",
        "name": TASK_PLAN_TOOL_NAME,
        "disable_parallel_tool_use": True,
    }


def _task_plan_tool(schema: Mapping[str, object], *, strict: bool) -> dict[str, object]:
    return {
        "name": TASK_PLAN_TOOL_NAME,
        "description": (
            "Return the complete task plan as structured JSON. Use phased actions, "
            "assign one or more robots to each action, and include only actions that "
            "match the provided robot skills and task constraints."
        ),
        "input_schema": dict(schema),
        "strict": strict,
        "input_examples": [
            {
                "notes": "Robot 1 reaches the switch while Robot 2 handles the laptop.",
                "phases": [
                    {
                        "label": "prepare",
                        "actions": [
                            {
                                "robots": ["robot1"],
                                "skill": "GoToObject",
                                "object_name": "LightSwitch",
                            },
                            {
                                "robots": ["robot2"],
                                "skill": "GoToObject",
                                "object_name": "Laptop",
                            },
                        ],
                    },
                    {
                        "actions": [
                            {
                                "robots": ["robot2"],
                                "skill": "SwitchOn",
                                "object_name": "Laptop",
                            }
                        ]
                    },
                ],
            }
        ],
    }


def _extract_tool_input(message: Any) -> Any:
    blocks = getattr(message, "content", None)
    if not isinstance(blocks, list):
        raise AnthropicPlanningError("Claude response did not contain content blocks")

    payloads: list[Any] = []
    for block in blocks:
        if _block_type(block) != "tool_use":
            continue
        if _block_name(block) != TASK_PLAN_TOOL_NAME:
            continue
        payloads.append(_block_input(block))

    if len(payloads) == 1:
        return payloads[0]
    if len(payloads) > 1:
        raise AnthropicPlanningError(
            f"Claude emitted {len(payloads)} {TASK_PLAN_TOOL_NAME!r} tool calls, expected exactly one"
        )

    stop_reason = getattr(message, "stop_reason", None)
    raise AnthropicPlanningError(
        f"Claude did not emit the {TASK_PLAN_TOOL_NAME!r} tool call (stop_reason={stop_reason!r})"
    )


def _block_type(block: Any) -> str | None:
    if isinstance(block, Mapping):
        value = block.get("type")
        return value if isinstance(value, str) else None
    value = getattr(block, "type", None)
    return value if isinstance(value, str) else None


def _block_name(block: Any) -> str | None:
    if isinstance(block, Mapping):
        value = block.get("name")
        return value if isinstance(value, str) else None
    value = getattr(block, "name", None)
    return value if isinstance(value, str) else None


def _block_input(block: Any) -> Any:
    if isinstance(block, Mapping):
        return block.get("input")
    return getattr(block, "input", None)


def _usage_mapping(usage: Any) -> Mapping[str, object] | None:
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        return dict(usage)

    data: dict[str, object] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            data[key] = value
    return data or None

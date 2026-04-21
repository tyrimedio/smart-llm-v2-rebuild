"""Provider-specific prompt and multimodal message builders."""

from __future__ import annotations

import base64
import json

from smart_llm_v2.agents.json_planner import JsonPlannerRequest
from smart_llm_v2.agents.model_profiles import Provider


def build_planning_prompt(request: JsonPlannerRequest) -> str:
    return "\n".join(
        (
            "Plan the task by calling the submit_task_plan tool exactly once.",
            "Do not answer in prose. Use the provided task context, robot catalog, and observations.",
            "Task context JSON:",
            json.dumps(request.context, indent=2, sort_keys=True),
        )
    )


def build_anthropic_messages(request: JsonPlannerRequest) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    content.extend(_anthropic_image_blocks(request))
    content.append({"type": "text", "text": build_planning_prompt(request)})
    return [{"role": "user", "content": content}]


def build_openai_chat_messages(request: JsonPlannerRequest) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [{"type": "text", "text": build_planning_prompt(request)}]
    content.extend(_openai_image_blocks(request))
    return [
        {"role": "system", "content": request.system_message},
        {"role": "user", "content": content},
    ]


def _anthropic_image_blocks(request: JsonPlannerRequest) -> list[dict[str, object]]:
    if not request.profile.vision_enabled:
        return []
    blocks: list[dict[str, object]] = []
    for image in request.images:
        blocks.append(
            {
                "type": "text",
                "text": _image_caption(image.label, image.agent_id),
            }
        )
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": _base64_ascii(image.data),
                },
            }
        )
    return blocks


def _openai_image_blocks(request: JsonPlannerRequest) -> list[dict[str, object]]:
    if not request.profile.vision_enabled:
        return []
    blocks: list[dict[str, object]] = []
    for image in request.images:
        blocks.append({"type": "text", "text": _image_caption(image.label, image.agent_id)})
        image_part: dict[str, object] = {
            "type": "image_url",
            "image_url": {
                "url": _data_url(image.media_type, image.data),
            },
        }
        if request.profile.provider is Provider.OPENAI and request.profile.image_detail is not None:
            image_part["image_url"]["detail"] = request.profile.image_detail
        blocks.append(image_part)
    return blocks


def _image_caption(label: str | None, agent_id: int | None) -> str:
    parts = ["Observation image"]
    if label:
        parts.append(f"label={label}")
    if agent_id is not None:
        parts.append(f"agent_id={agent_id}")
    return ", ".join(parts)


def _data_url(media_type: str, data: bytes) -> str:
    return f"data:{media_type};base64,{_base64_ascii(data)}"


def _base64_ascii(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

"""Provider and model profile resolution for structured planning."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import StrEnum

PROVIDER_ENV_VAR = "SMART_LLM_V2_PROVIDER"
MODEL_ENV_VAR = "SMART_LLM_V2_MODEL"
PROFILE_VARIANT_ENV_VAR = "SMART_LLM_V2_PROFILE_VARIANT"
BASE_URL_ENV_VAR = "SMART_LLM_V2_BASE_URL"

DEFAULT_KIMI_BASE_URL = "https://api.moonshot.ai/v1"


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    KIMI = "kimi"


class ProfileVariant(StrEnum):
    SYMBOLIC = "symbolic"
    MULTIMODAL = "multimodal"


class Transport(StrEnum):
    ANTHROPIC_MESSAGES = "anthropic_messages"
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    provider: Provider
    model: str
    variant: ProfileVariant
    api_key_env_var: str
    transport: Transport
    base_url: str | None = None
    prompt_variant: str = "structured_control"
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    thinking_budget_tokens: int | None = None
    vision_enabled: bool = False
    image_detail: str | None = None
    tool_strict: bool = True

    @property
    def planner_name(self) -> str:
        return f"{self.provider}:{self.model}:{self.variant}"


_BASE_PROFILES: dict[tuple[Provider, ProfileVariant], ModelProfile] = {
    (Provider.ANTHROPIC, ProfileVariant.SYMBOLIC): ModelProfile(
        provider=Provider.ANTHROPIC,
        model="claude-opus-4-7",
        variant=ProfileVariant.SYMBOLIC,
        api_key_env_var="ANTHROPIC_API_KEY",
        transport=Transport.ANTHROPIC_MESSAGES,
        thinking_budget_tokens=1024,
    ),
    (Provider.ANTHROPIC, ProfileVariant.MULTIMODAL): ModelProfile(
        provider=Provider.ANTHROPIC,
        model="claude-opus-4-7",
        variant=ProfileVariant.MULTIMODAL,
        api_key_env_var="ANTHROPIC_API_KEY",
        transport=Transport.ANTHROPIC_MESSAGES,
        thinking_budget_tokens=1024,
        vision_enabled=True,
    ),
    (Provider.OPENAI, ProfileVariant.SYMBOLIC): ModelProfile(
        provider=Provider.OPENAI,
        model="gpt-5.4",
        variant=ProfileVariant.SYMBOLIC,
        api_key_env_var="OPENAI_API_KEY",
        transport=Transport.OPENAI_CHAT_COMPLETIONS,
        reasoning_effort="high",
    ),
    (Provider.OPENAI, ProfileVariant.MULTIMODAL): ModelProfile(
        provider=Provider.OPENAI,
        model="gpt-5.4",
        variant=ProfileVariant.MULTIMODAL,
        api_key_env_var="OPENAI_API_KEY",
        transport=Transport.OPENAI_CHAT_COMPLETIONS,
        reasoning_effort="high",
        vision_enabled=True,
        image_detail="low",
    ),
    (Provider.KIMI, ProfileVariant.SYMBOLIC): ModelProfile(
        provider=Provider.KIMI,
        model="kimi-k2.6",
        variant=ProfileVariant.SYMBOLIC,
        api_key_env_var="MOONSHOT_API_KEY",
        transport=Transport.OPENAI_CHAT_COMPLETIONS,
        base_url=DEFAULT_KIMI_BASE_URL,
        temperature=1.0,
        top_p=0.95,
        tool_strict=False,
    ),
    (Provider.KIMI, ProfileVariant.MULTIMODAL): ModelProfile(
        provider=Provider.KIMI,
        model="kimi-k2.6",
        variant=ProfileVariant.MULTIMODAL,
        api_key_env_var="MOONSHOT_API_KEY",
        transport=Transport.OPENAI_CHAT_COMPLETIONS,
        base_url=DEFAULT_KIMI_BASE_URL,
        temperature=1.0,
        top_p=0.95,
        vision_enabled=True,
        tool_strict=False,
    ),
}


def resolve_model_profile(
    *,
    provider: str | Provider | None = None,
    model: str | None = None,
    variant: str | ProfileVariant | None = None,
    base_url: str | None = None,
) -> ModelProfile:
    selected_model = model or os.environ.get(MODEL_ENV_VAR)
    selected_provider = provider or os.environ.get(PROVIDER_ENV_VAR)
    selected_variant = variant or os.environ.get(
        PROFILE_VARIANT_ENV_VAR,
        ProfileVariant.SYMBOLIC.value,
    )
    selected_base_url = base_url or os.environ.get(BASE_URL_ENV_VAR)

    resolved_provider = _resolve_provider(selected_provider, selected_model)
    resolved_variant = ProfileVariant(str(selected_variant))
    template = _BASE_PROFILES[(resolved_provider, resolved_variant)]

    return replace(
        template,
        model=selected_model or template.model,
        base_url=selected_base_url or template.base_url,
    )


def _resolve_provider(provider: str | Provider | None, model: str | None) -> Provider:
    if provider is not None:
        return Provider(str(provider))
    if model is not None:
        return infer_provider_from_model(model)
    return Provider.ANTHROPIC


def infer_provider_from_model(model: str) -> Provider:
    normalized = model.casefold()
    if normalized.startswith("claude-"):
        return Provider.ANTHROPIC
    if normalized.startswith("gpt-") or normalized.startswith("o"):
        return Provider.OPENAI
    if normalized.startswith("kimi-") or normalized.startswith("moonshot-"):
        return Provider.KIMI
    raise ValueError(f"Cannot infer provider from model {model!r}")

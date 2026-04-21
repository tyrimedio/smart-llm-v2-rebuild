from __future__ import annotations

import pytest

from smart_llm_v2.agents.model_profiles import (
    BASE_URL_ENV_VAR,
    DEFAULT_KIMI_BASE_URL,
    MODEL_ENV_VAR,
    PROFILE_VARIANT_ENV_VAR,
    PROVIDER_ENV_VAR,
    ProfileVariant,
    Provider,
    Transport,
    infer_provider_from_model,
    resolve_model_profile,
)


def test_resolve_model_profile_defaults_to_anthropic_symbolic(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        PROVIDER_ENV_VAR,
        MODEL_ENV_VAR,
        PROFILE_VARIANT_ENV_VAR,
        BASE_URL_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)

    profile = resolve_model_profile()

    assert profile.provider is Provider.ANTHROPIC
    assert profile.model == "claude-opus-4-7"
    assert profile.variant is ProfileVariant.SYMBOLIC
    assert profile.transport is Transport.ANTHROPIC_MESSAGES
    assert profile.vision_enabled is False


def test_resolve_model_profile_uses_env_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "gpt-5.4")
    monkeypatch.setenv(PROFILE_VARIANT_ENV_VAR, "multimodal")

    profile = resolve_model_profile()

    assert profile.provider is Provider.OPENAI
    assert profile.model == "gpt-5.4"
    assert profile.variant is ProfileVariant.MULTIMODAL
    assert profile.transport is Transport.OPENAI_CHAT_COMPLETIONS
    assert profile.vision_enabled is True
    assert profile.image_detail == "low"


def test_resolve_model_profile_keeps_kimi_default_base_url_and_allows_override() -> None:
    default_profile = resolve_model_profile(provider="kimi", variant="symbolic")
    override_profile = resolve_model_profile(
        provider="kimi",
        variant="symbolic",
        base_url="https://proxy.example/v1",
    )

    assert default_profile.base_url == DEFAULT_KIMI_BASE_URL
    assert override_profile.base_url == "https://proxy.example/v1"


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("claude-opus-4-7", Provider.ANTHROPIC),
        ("gpt-5.4", Provider.OPENAI),
        ("kimi-k2.6", Provider.KIMI),
    ],
)
def test_infer_provider_from_model(model: str, provider: Provider) -> None:
    assert infer_provider_from_model(model) is provider

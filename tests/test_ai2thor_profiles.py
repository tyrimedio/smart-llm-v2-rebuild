import pytest

import smart_llm_v2.env.profiles as profiles
from smart_llm_v2.env import (
    Ai2ThorProfile,
    LOCAL_EXECUTABLE_ENV_VAR,
    detect_host_architecture,
    recommended_ai2thor_config,
    resolve_ai2thor_profile,
)


def test_detect_host_architecture_normalizes_machine_names() -> None:
    assert detect_host_architecture("arm64") == "arm64"
    assert detect_host_architecture("aarch64") == "arm64"
    assert detect_host_architecture("x86_64") == "x86_64"
    assert detect_host_architecture("amd64") == "x86_64"


def test_detect_host_architecture_prefers_mac_hardware_signal(monkeypatch) -> None:
    monkeypatch.setattr(profiles.platform, "system", lambda: "Darwin")

    class Result:
        stdout = "1\n"

    monkeypatch.setattr(profiles.subprocess, "run", lambda *args, **kwargs: Result())

    assert detect_host_architecture() == "arm64"


def test_auto_profile_prefers_demo_low_on_apple_silicon() -> None:
    assert resolve_ai2thor_profile(machine="arm64") is Ai2ThorProfile.DEMO_LOW


def test_auto_profile_prefers_intel_default_on_x86_64() -> None:
    assert resolve_ai2thor_profile(machine="x86_64") is Ai2ThorProfile.INTEL_DEFAULT


def test_recommended_demo_low_config_is_lightweight() -> None:
    config = recommended_ai2thor_config(Ai2ThorProfile.DEMO_LOW)

    assert config.quality == "Very Low"
    assert config.width == 400
    assert config.height == 300
    assert config.add_third_party_camera is False


def test_benchmark_headless_profile_sets_headless_mode() -> None:
    config = recommended_ai2thor_config(Ai2ThorProfile.BENCHMARK_HEADLESS)

    assert config.headless is True
    assert config.quality == "Very Low"
    assert config.width == 300
    assert config.height == 300


def test_local_native_arm64_profile_requires_executable_path() -> None:
    with pytest.raises(ValueError, match=LOCAL_EXECUTABLE_ENV_VAR):
        recommended_ai2thor_config(Ai2ThorProfile.LOCAL_NATIVE_ARM64)


def test_local_native_arm64_profile_accepts_custom_executable_path() -> None:
    config = recommended_ai2thor_config(
        Ai2ThorProfile.LOCAL_NATIVE_ARM64,
        local_executable_path="/Applications/AI2-THOR-arm64.app/Contents/MacOS/AI2-THOR",
    )

    assert config.local_executable_path == "/Applications/AI2-THOR-arm64.app/Contents/MacOS/AI2-THOR"
    assert config.quality == "Medium"
    assert config.width == 800

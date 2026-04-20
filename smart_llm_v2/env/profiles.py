"""AI2-THOR runtime profiles.

The official macOS AI2-THOR release currently resolves to an Intel build, so
Apple Silicon machines benefit from lighter visual defaults even though the
planner and executor code stays the same. These profiles keep the swap between
machines explicit and reproducible.
"""

from __future__ import annotations

import os
import platform
import subprocess
from enum import StrEnum

from smart_llm_v2.env.config import Ai2ThorConfig

PROFILE_ENV_VAR = "SMART_LLM_V2_AI2THOR_PROFILE"
LOCAL_EXECUTABLE_ENV_VAR = "SMART_LLM_V2_AI2THOR_LOCAL_EXECUTABLE"


class Ai2ThorProfile(StrEnum):
    AUTO = "auto"
    DEMO_LOW = "demo_low"
    BENCHMARK_HEADLESS = "benchmark_headless"
    INTEL_DEFAULT = "intel_default"
    LOCAL_NATIVE_ARM64 = "local_native_arm64"


def detect_host_architecture(machine: str | None = None) -> str:
    if machine is None and _darwin_reports_arm64():
        return "arm64"

    value = (machine or platform.machine()).casefold()
    if value in {"arm64", "aarch64"}:
        return "arm64"
    if value in {"x86_64", "amd64"}:
        return "x86_64"
    return value


def resolve_ai2thor_profile(
    profile: str | Ai2ThorProfile | None = None,
    *,
    machine: str | None = None,
) -> Ai2ThorProfile:
    if profile is None:
        profile = os.environ.get(PROFILE_ENV_VAR, Ai2ThorProfile.AUTO.value)

    selected = Ai2ThorProfile(str(profile))
    if selected is not Ai2ThorProfile.AUTO:
        return selected

    architecture = detect_host_architecture(machine)
    if architecture == "arm64":
        return Ai2ThorProfile.DEMO_LOW
    return Ai2ThorProfile.INTEL_DEFAULT


def recommended_ai2thor_config(
    profile: str | Ai2ThorProfile | None = None,
    *,
    machine: str | None = None,
    local_executable_path: str | None = None,
) -> Ai2ThorConfig:
    selected = resolve_ai2thor_profile(profile, machine=machine)
    executable_path = local_executable_path or os.environ.get(LOCAL_EXECUTABLE_ENV_VAR)

    if selected is Ai2ThorProfile.DEMO_LOW:
        return Ai2ThorConfig(
            width=400,
            height=300,
            quality="Very Low",
            fullscreen=False,
            headless=False,
            add_third_party_camera=False,
        )

    if selected is Ai2ThorProfile.BENCHMARK_HEADLESS:
        return Ai2ThorConfig(
            width=300,
            height=300,
            quality="Very Low",
            fullscreen=False,
            headless=True,
            add_third_party_camera=False,
        )

    if selected is Ai2ThorProfile.INTEL_DEFAULT:
        return Ai2ThorConfig(
            width=800,
            height=600,
            quality="Medium",
            fullscreen=False,
            headless=False,
            add_third_party_camera=False,
        )

    if selected is Ai2ThorProfile.LOCAL_NATIVE_ARM64:
        if not executable_path:
            raise ValueError(
                "local_native_arm64 requires a local executable path. Set "
                f"{LOCAL_EXECUTABLE_ENV_VAR} or pass local_executable_path."
            )
        return Ai2ThorConfig(
            width=800,
            height=600,
            quality="Medium",
            fullscreen=False,
            headless=False,
            local_executable_path=executable_path,
            add_third_party_camera=False,
        )

    raise AssertionError(f"Unhandled AI2-THOR profile: {selected}")


def _darwin_reports_arm64() -> bool:
    if platform.system() != "Darwin":
        return False

    try:
        result = subprocess.run(
            ["sysctl", "-in", "hw.optional.arm64"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

    return result.stdout.strip() == "1"

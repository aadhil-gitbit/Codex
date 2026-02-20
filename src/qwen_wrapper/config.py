from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the wrapper and loop guardrails."""

    qwen_base_url: str = "http://127.0.0.1:8000"
    qwen_model: str = "qwen2.5:latest"
    request_timeout_s: float = 30.0
    request_retries: int = 2

    max_agent_steps: int = 4
    max_history_turns: int = 12
    max_consecutive_failures: int = 2

    @property
    def chat_endpoint(self) -> str:
        return f"{self.qwen_base_url.rstrip('/')}/v1/chat/completions"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_settings() -> Settings:
    """Load runtime settings from environment variables."""

    return Settings(
        qwen_base_url=os.getenv("QWEN_BASE_URL", Settings.qwen_base_url),
        qwen_model=os.getenv("QWEN_MODEL", Settings.qwen_model),
        request_timeout_s=_env_float("QWEN_TIMEOUT_S", Settings.request_timeout_s),
        request_retries=_env_int("QWEN_RETRIES", Settings.request_retries),
        max_agent_steps=_env_int("QWEN_MAX_STEPS", Settings.max_agent_steps),
        max_history_turns=_env_int("QWEN_MAX_HISTORY_TURNS", Settings.max_history_turns),
        max_consecutive_failures=_env_int(
            "QWEN_MAX_CONSECUTIVE_FAILURES", Settings.max_consecutive_failures
        ),
    )

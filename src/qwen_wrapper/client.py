"""Model client adapter for local Qwen-compatible endpoints."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelResponse:
    """Normalized response from model backends."""

    content: str
    tool_calls: List[Dict[str, Any]]
    token_usage: Dict[str, int]
    raw: Dict[str, Any]


class LocalQwenClient:
    """Client adapter for local Qwen endpoints.

    Supports OpenAI-compatible chat/completions and a custom JSON endpoint
    returning: {"content": ..., "tool_calls": [...], "usage": {...}}.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_s: float = 30.0,
        openai_compatible: bool = True,
        max_tokens: int = 1024,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.openai_compatible = openai_compatible
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
    ) -> ModelResponse:
        payload: Dict[str, Any]
        if self.openai_compatible:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
            }
            if tools:
                payload["tools"] = tools
        else:
            payload = {
                "messages": messages,
                "tools": tools or [],
                "temperature": temperature,
                "max_tokens": self.max_tokens,
            }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Qwen endpoint HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach Qwen endpoint: {exc}") from exc

        return self._normalize_response(raw)

    def _normalize_response(self, raw: Dict[str, Any]) -> ModelResponse:
        if self.openai_compatible:
            choices = raw.get("choices") or []
            if not choices:
                raise RuntimeError("Missing choices in OpenAI-compatible response")
            message = choices[0].get("message", {})
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            usage = raw.get("usage") or {}
            token_usage = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
            return ModelResponse(
                content=content,
                tool_calls=tool_calls,
                token_usage=token_usage,
                raw=raw,
            )

        content = raw.get("content") or ""
        tool_calls = raw.get("tool_calls") or []
        usage = raw.get("usage") or {}
        token_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage=token_usage,
            raw=raw,
        )

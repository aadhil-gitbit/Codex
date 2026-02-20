"""Tool registry and safe execution utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Set

from .schemas import ToolCall, ToolResult

ToolFn = Callable[..., Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    function: ToolFn
    json_schema: Dict[str, Any]


class ToolRegistry:
    """In-memory registry of tools the agent can execute."""

    def __init__(self, allowed_tools: Optional[Iterable[str]] = None) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._allowed_tools: Optional[Set[str]] = set(allowed_tools) if allowed_tools else None

    def register(
        self,
        name: str,
        description: str,
        json_schema: Dict[str, Any],
    ) -> Callable[[ToolFn], ToolFn]:
        def decorator(func: ToolFn) -> ToolFn:
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                function=func,
                json_schema=json_schema,
            )
            return func

        return decorator

    def get_openai_tools(self) -> list[Dict[str, Any]]:
        tools = []
        for spec in self._tools.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.json_schema,
                    },
                }
            )
        return tools

    def execute(self, call: ToolCall) -> ToolResult:
        if self._allowed_tools is not None and call.name not in self._allowed_tools:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                success=False,
                error=f"Tool '{call.name}' is not in the allowed-tools whitelist",
            )

        spec = self._tools.get(call.name)
        if spec is None:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                success=False,
                error=f"Tool '{call.name}' is not registered",
            )

        started = time.perf_counter()
        try:
            output = spec.function(**call.arguments)
            latency_ms = (time.perf_counter() - started) * 1000
            return ToolResult(
                call_id=call.id,
                name=call.name,
                success=True,
                output=output,
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000
            return ToolResult(
                call_id=call.id,
                name=call.name,
                success=False,
                error=str(exc),
                latency_ms=latency_ms,
            )

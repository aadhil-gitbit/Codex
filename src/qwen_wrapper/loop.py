"""Agent loop orchestration for local Qwen tool-using interactions."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .client import LocalQwenClient
from .schemas import AgentTurn, FinalAnswer, ToolCall, ToolResult
from .tools import ToolRegistry


@dataclass
class LoopConfig:
    max_steps: int = 8
    timeout_s: float = 45.0
    max_total_tokens: int = 12_000
    request_retries: int = 2
    retry_backoff_s: float = 0.5


class AgentLoop:
    """Iterative loop that alternates between model calls and tool execution."""

    def __init__(
        self,
        client: LocalQwenClient,
        tools: ToolRegistry,
        config: Optional[LoopConfig] = None,
        log_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.config = config or LoopConfig()
        self.log_hook = log_hook

    def run(self, query: str, context: Optional[str] = None) -> FinalAnswer:
        started = time.perf_counter()
        total_tokens = 0
        turns: List[AgentTurn] = []
        tool_results: List[ToolResult] = []

        messages: List[Dict[str, Any]] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": query})

        for step in range(self.config.max_steps):
            elapsed = time.perf_counter() - started
            if elapsed > self.config.timeout_s:
                return FinalAnswer(
                    answer="Timed out while solving the request.",
                    termination_reason="timeout",
                    steps=step,
                    tool_results=tool_results,
                )

            response, latency_ms = self._call_model_with_retries(messages)
            total_tokens += int(response.token_usage.get("total_tokens", 0))

            if total_tokens > self.config.max_total_tokens:
                return FinalAnswer(
                    answer="Token budget exceeded before reaching a final answer.",
                    termination_reason="token_budget_exceeded",
                    steps=step + 1,
                    tool_results=tool_results,
                )

            parsed_calls = self._parse_tool_calls(response.tool_calls, response.content)
            turn = AgentTurn(
                step_index=step,
                prompt=messages[-1]["content"],
                raw_response=response.content,
                tool_calls=parsed_calls,
                token_usage=response.token_usage,
                latency_ms=latency_ms,
            )

            self._emit_log(
                {
                    "step_index": step,
                    "latency_ms": latency_ms,
                    "token_usage": response.token_usage,
                    "tool_count": len(parsed_calls),
                }
            )

            if not parsed_calls:
                turns.append(turn)
                return FinalAnswer(
                    answer=response.content.strip(),
                    termination_reason="final_answer",
                    steps=step + 1,
                    tool_results=tool_results,
                )

            assistant_message: Dict[str, Any] = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_message["tool_calls"] = response.tool_calls
            messages.append(assistant_message)

            for call in parsed_calls:
                result = self.tools.execute(call)
                tool_results.append(result)
                turn.tool_results.append(result)

                self._emit_log(
                    {
                        "step_index": step,
                        "tool_name": call.name,
                        "tool_success": result.success,
                        "tool_latency_ms": result.latency_ms,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(
                            {
                                "success": result.success,
                                "output": result.output,
                                "error": result.error,
                            }
                        ),
                    }
                )

            turns.append(turn)

        return FinalAnswer(
            answer="Stopped due to max loop iterations.",
            termination_reason="max_steps",
            steps=self.config.max_steps,
            tool_results=tool_results,
        )

    def _call_model_with_retries(self, messages: List[Dict[str, Any]]) -> tuple[Any, float]:
        last_error: Optional[Exception] = None
        for attempt in range(self.config.request_retries + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat(messages=messages, tools=self.tools.get_openai_tools())
                latency_ms = (time.perf_counter() - started) * 1000
                return response, latency_ms
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._emit_log(
                    {
                        "event": "model_error",
                        "attempt": attempt + 1,
                        "error": str(exc),
                    }
                )
                if attempt < self.config.request_retries:
                    time.sleep(self.config.retry_backoff_s * (attempt + 1))

        raise RuntimeError(f"Model request failed after retries: {last_error}")

    def _parse_tool_calls(self, raw_calls: List[Dict[str, Any]], content: str) -> List[ToolCall]:
        if raw_calls:
            calls: List[ToolCall] = []
            for raw in raw_calls:
                function = raw.get("function", {})
                raw_args = function.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = raw_args or {}
                calls.append(
                    ToolCall(
                        id=raw.get("id") or f"call_{uuid.uuid4().hex}",
                        name=function.get("name", ""),
                        arguments=args,
                    )
                )
            return calls

        # Optional fallback parser for custom text protocols.
        marker = "TOOL_CALL:"
        if marker not in content:
            return []
        try:
            payload = content.split(marker, maxsplit=1)[1].strip()
            parsed = json.loads(payload)
            return [
                ToolCall(
                    id=parsed.get("id") or f"call_{uuid.uuid4().hex}",
                    name=parsed["name"],
                    arguments=parsed.get("arguments") or {},
                )
            ]
        except Exception:  # noqa: BLE001
            return []

    def _emit_log(self, payload: Dict[str, Any]) -> None:
        if self.log_hook:
            self.log_hook(payload)

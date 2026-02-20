from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import json
from urllib import request
from urllib.error import HTTPError, URLError

from .config import Settings, load_settings
from .memory import InMemorySessionStore, SessionMemoryStore, Turn


@dataclass
class AgentRunResult:
    answer: str
    steps: List[Dict[str, Any]]
    tool_trace: List[Dict[str, Any]]
    termination_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentRunner:
    """Small, reliable loop wrapper for driving a local Qwen model endpoint."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        memory_store: Optional[SessionMemoryStore] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.memory_store = memory_store or InMemorySessionStore(
            max_turns=self.settings.max_history_turns
        )

    def run(
        self,
        user_input: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentRunResult:
        metadata = metadata or {}
        steps: List[Dict[str, Any]] = []
        tool_trace: List[Dict[str, Any]] = []
        consecutive_failures = 0
        termination_reason = "max_steps_reached"
        answer = ""

        self.memory_store.append_turn(session_id, Turn(role="user", content=user_input))

        for step_idx in range(1, self.settings.max_agent_steps + 1):
            messages = self._build_messages(session_id, metadata)
            try:
                answer = self._invoke_with_retries(messages, tool_trace)
                steps.append({"step": step_idx, "status": "ok"})
                self.memory_store.append_turn(session_id, Turn(role="assistant", content=answer))
                termination_reason = "completed"
                break
            except Exception as exc:  # pragma: no cover - handled return path
                consecutive_failures += 1
                steps.append({"step": step_idx, "status": "error", "error": str(exc)})
                if consecutive_failures >= self.settings.max_consecutive_failures:
                    termination_reason = "guardrail_consecutive_failures"
                    break

        if not answer:
            answer = ""

        return AgentRunResult(
            answer=answer,
            steps=steps,
            tool_trace=tool_trace,
            termination_reason=termination_reason,
        )

    def _build_messages(self, session_id: str, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        system_prompt = (
            "You are an internal agent assistant. Respond clearly and concisely. "
            "Treat metadata as contextual hints and do not fabricate tool execution."
        )
        messages = [{"role": "system", "content": system_prompt}]
        if metadata:
            messages.append({"role": "system", "content": f"metadata={metadata}"})

        for turn in self.memory_store.get_recent_turns(session_id):
            messages.append({"role": turn.role, "content": turn.content})

        return messages

    def _invoke_with_retries(
        self, messages: List[Dict[str, str]], tool_trace: List[Dict[str, Any]]
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.settings.request_retries + 2):
            try:
                response_text = self._call_qwen(messages)
                tool_trace.append(
                    {
                        "kind": "model_call",
                        "attempt": attempt,
                        "endpoint": self.settings.chat_endpoint,
                        "status": "ok",
                    }
                )
                return response_text
            except Exception as exc:  # pragma: no cover - exercised by runtime failures
                last_error = exc
                tool_trace.append(
                    {
                        "kind": "model_call",
                        "attempt": attempt,
                        "endpoint": self.settings.chat_endpoint,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        raise RuntimeError(f"Model call failed after retries: {last_error}")

    def _call_qwen(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.settings.qwen_model,
            "messages": messages,
            "stream": False,
        }
        encoded = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.settings.chat_endpoint,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.settings.request_timeout_s) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(f"HTTP error from Qwen server: {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Connection error to Qwen server: {exc.reason}") from exc

        data = json.loads(body)
        return self._extract_message_text(data)

    @staticmethod
    def _extract_message_text(data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            if "content" in message:
                return str(message["content"])

        if "response" in data:
            return str(data["response"])

        raise ValueError("Unable to parse model response payload")

"""Structured schemas for the Qwen wrapper agent loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A model-requested tool invocation."""

    id: str = Field(..., description="Unique call id for tool invocation")
    name: str = Field(..., description="Registered tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Execution result for a tool call."""

    call_id: str
    name: str
    success: bool = True
    output: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class FinalAnswer(BaseModel):
    """Terminal answer emitted by the loop."""

    answer: str
    termination_reason: str
    steps: int
    tool_results: List[ToolResult] = Field(default_factory=list)


class AgentTurn(BaseModel):
    """A full step exchanged between orchestrator and model."""

    step_index: int
    prompt: str
    raw_response: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    token_usage: Optional[Dict[str, int]] = None
    latency_ms: Optional[float] = None

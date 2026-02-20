# Codex

Built using chatgpt codex.

## Qwen wrapper package

A new Python package is available at `src/qwen_wrapper/` with:

- `client.py`: `LocalQwenClient` for OpenAI-compatible or custom local HTTP endpoints.
- `loop.py`: `AgentLoop` orchestration with retries, guardrails, and tool-feedback iterations.
- `tools.py`: `ToolRegistry` for registration, whitelist enforcement, and safe execution.
- `schemas.py`: structured `AgentTurn`, `ToolCall`, `ToolResult`, and `FinalAnswer` models.

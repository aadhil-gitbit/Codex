from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import AgentRunner
from .config import load_settings
from .memory import InMemorySessionStore


class AgentRunRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class AgentRunResponse(BaseModel):
    answer: str
    steps: List[Dict[str, Any]]
    tool_trace: List[Dict[str, Any]]
    termination_reason: str


def create_app() -> FastAPI:
    settings = load_settings()
    memory = InMemorySessionStore(max_turns=settings.max_history_turns)
    runner = AgentRunner(settings=settings, memory_store=memory)

    app = FastAPI(title="Qwen Wrapper API", version="0.1.0")

    @app.post("/agent/run", response_model=AgentRunResponse)
    def run_agent(request: AgentRunRequest) -> AgentRunResponse:
        result = runner.run(
            user_input=request.user_input,
            session_id=request.session_id,
            metadata=request.metadata,
        )
        return AgentRunResponse(**result.to_dict())

    return app


app = create_app()

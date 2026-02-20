from __future__ import annotations

import argparse
import json
import uuid
from typing import Any, Dict

from .agent import AgentRunner
from .config import load_settings
from .memory import InMemorySessionStore


def _parse_metadata(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run qwen wrapper agent loop from the CLI.")
    parser.add_argument("--prompt", required=True, help="User prompt to send into agent loop")
    parser.add_argument(
        "--session-id",
        default=f"cli-{uuid.uuid4()}",
        help="Session identifier for memory scoping",
    )
    parser.add_argument(
        "--metadata-json",
        default="{}",
        help="Optional metadata object as JSON",
    )
    args = parser.parse_args()

    settings = load_settings()
    memory = InMemorySessionStore(max_turns=settings.max_history_turns)
    runner = AgentRunner(settings=settings, memory_store=memory)

    metadata = _parse_metadata(args.metadata_json)
    result = runner.run(user_input=args.prompt, session_id=args.session_id, metadata=metadata)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

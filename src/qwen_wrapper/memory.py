from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, List, Protocol


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


class SessionMemoryStore(Protocol):
    """Memory abstraction to support in-memory and future persistent backends."""

    def get_recent_turns(self, session_id: str) -> List[Turn]:
        ...

    def append_turn(self, session_id: str, turn: Turn) -> None:
        ...

    def clear(self, session_id: str) -> None:
        ...


class InMemorySessionStore:
    """Thread-safe in-memory implementation for session-scoped turns."""

    def __init__(self, max_turns: int = 12) -> None:
        self._max_turns = max_turns
        self._store: Dict[str, Deque[Turn]] = defaultdict(lambda: deque(maxlen=self._max_turns))
        self._lock = Lock()

    def get_recent_turns(self, session_id: str) -> List[Turn]:
        with self._lock:
            return list(self._store[session_id])

    def append_turn(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            self._store[session_id].append(turn)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

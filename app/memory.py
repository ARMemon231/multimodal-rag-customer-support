from collections import defaultdict, deque


class ConversationMemory:
    def __init__(self, max_turns: int = 8) -> None:
        self._store: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=max_turns))

    def add_turn(self, user_id: str, text: str) -> None:
        self._store[user_id].append(text)

    def get_context(self, user_id: str) -> str:
        turns = list(self._store[user_id])
        if not turns:
            return ""
        return "\n".join(turns)

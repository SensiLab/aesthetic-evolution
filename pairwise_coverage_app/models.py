from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


CSV_COLUMNS = ["timestamp", "session_id", "image_a", "image_b", "outcome"]
VALID_OUTCOMES = {"first", "second", "draw"}


@dataclass(frozen=True)
class VoteRecord:
    timestamp: str
    session_id: str
    image_a: str
    image_b: str
    outcome: str

    @classmethod
    def create(cls, session_id: str, image_a: str, image_b: str, outcome: str) -> "VoteRecord":
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"Invalid outcome '{outcome}'. Expected one of {sorted(VALID_OUTCOMES)}")
        return cls(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            session_id=session_id,
            image_a=image_a,
            image_b=image_b,
            outcome=outcome,
        )

    def to_row(self) -> dict[str, str]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "image_a": self.image_a,
            "image_b": self.image_b,
            "outcome": self.outcome,
        }

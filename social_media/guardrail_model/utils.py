from dataclasses import dataclass, asdict
from typing import List

@dataclass
class ModerationResult:
    prompt: str or List[dict]
    model: str
    flagged: bool = None
    prob_flagged: float = None
    score: List[float] = None
    reason: str = ""

    def to_dict(self):
        return asdict(self)

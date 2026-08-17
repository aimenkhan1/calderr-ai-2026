"""
procedural.py

PROCEDURAL MEMORY -- "how the agent has learned to behave." Stores
behavioral corrections/rules distilled from feedback -- not facts about the
world, but instructions about HOW the agent should act going forward (e.g.
"when writing Python, always include type hints", "never delete a file
without asking first"). Each correction tracks how many times it's actually
been applied, so rules that never fire can be identified/pruned later.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class Correction:
    trigger_context: str   # when this rule applies, e.g. "writing python code"
    instruction: str       # what to do differently, e.g. "always include type hints"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    times_applied: int = 0
    active: bool = True


class ProceduralMemory:
    def __init__(self):
        self.corrections: List[Correction] = []

    def add_correction(self, trigger_context: str, instruction: str) -> Correction:
        correction = Correction(trigger_context=trigger_context, instruction=instruction)
        self.corrections.append(correction)
        return correction

    def get_active(self) -> List[Correction]:
        return [c for c in self.corrections if c.active]

    def deactivate(self, correction_id: str) -> None:
        for c in self.corrections:
            if c.id == correction_id:
                c.active = False

    def record_application(self, correction_id: str) -> None:
        for c in self.corrections:
            if c.id == correction_id:
                c.times_applied += 1

    def applicable_to(self, context: str, limit: int = 5) -> List[Correction]:
        """Active corrections whose trigger_context overlaps with the given context."""
        c_words = set(context.lower().split())
        scored = []
        for correction in self.get_active():
            trigger_words = set(correction.trigger_context.lower().split())
            overlap = len(c_words & trigger_words)
            if overlap > 0:
                scored.append((overlap, correction))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [c for _, c in scored[:limit]]

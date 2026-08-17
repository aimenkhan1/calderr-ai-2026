"""
semantic.py

SEMANTIC MEMORY -- "stable facts the agent knows, independent of any one
event." A profile of key -> value facts (name, preferences, role, etc.),
each with a confidence score and where it came from. Unlike episodic memory,
semantic facts don't decay with time  they get OVERWRITTEN when a newer,
more confident statement of the same fact comes in, which is how a profile
should behave (your current favorite language replaces your old one, it
doesn't just fade away).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class SemanticFact:
    key: str
    value: str
    confidence: float = 1.0  
    source: str = "user"   
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class SemanticMemory:
    def __init__(self):
        self._facts: Dict[str, SemanticFact] = {} 

    def set_fact(self, key: str, value: str, confidence: float = 1.0,
                 source: str = "user") -> SemanticFact:
        """Adds a new fact, or overwrites an existing one for the same key
        (this is what makes a 'profile' one current value per key)."""
        norm_key = key.strip().lower()
        fact = SemanticFact(key=norm_key, value=value, confidence=confidence, source=source)
        self._facts[norm_key] = fact
        return fact

    def get_fact(self, key: str) -> Optional[SemanticFact]:
        return self._facts.get(key.strip().lower())

    def get_profile(self) -> List[SemanticFact]:
        """All known facts, most recently updated first."""
        return sorted(self._facts.values(), key=lambda f: f.updated_at, reverse=True)

    def search(self, query: str, limit: int = 5) -> List[SemanticFact]:
        q_words = set(query.lower().split())
        scored = []
        for fact in self._facts.values():
            text = f"{fact.key} {fact.value}".lower()
            overlap = len(q_words & set(text.split()))
            if overlap > 0:
                scored.append((overlap, fact))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [fact for _, fact in scored[:limit]]

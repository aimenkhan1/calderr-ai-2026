"""
episodic.py

EPISODIC MEMORY - "what happened, and when."

Stores a raw, timestamped log of individual events/interactions. Each entry
(an Episode) has an importance score that decays over time -- trivial old
episodes eventually get pruned, important ones stick around longer. When the
log grows past a threshold, the oldest episodes get compressed into a
summary MemoryBlock instead of being deleted outright, so the agent keeps a
gist of its older history without keeping every raw detail forever.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional, Tuple


@dataclass
class Episode:
    content: str
    importance: float  # 0.0 - 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"importance must be in [0.0, 1.0], got {self.importance}")


@dataclass
class MemoryBlock:
    summary: str
    source_episode_ids: List[str]
    episode_count: int
    time_range: Tuple[datetime, datetime]
    importance: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


def exponential_decay(base_importance: float, elapsed_seconds: float,
                       half_life_seconds: float = 6 * 3600) -> float:
    if elapsed_seconds <= 0:
        return base_importance
    scaled_half_life = half_life_seconds * (0.5 + base_importance)
    decay_rate = math.log(2) / scaled_half_life
    return base_importance * math.exp(-decay_rate * elapsed_seconds)


def default_summarizer(episodes: List[Episode]) -> str:
    bullets = "; ".join(e.content.strip() for e in episodes)
    return f"[{len(episodes)} older episodes summarized] {bullets}"


Summarizer = Callable[[List[Episode]], str]


class EpisodicMemory:
    def __init__(
        self,
        consolidation_threshold: int = 12,
        consolidation_batch_size: int = 6,
        forget_threshold: float = 0.05,
        half_life_seconds: float = 6 * 3600,
        summarizer: Summarizer = default_summarizer,
    ):
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_batch_size = consolidation_batch_size
        self.forget_threshold = forget_threshold
        self.half_life_seconds = half_life_seconds
        self.summarizer = summarizer

        self.episodes: List[Episode] = []
        self.memory_blocks: List[MemoryBlock] = []

    def add_episode(self, content: str, importance: float = 0.5,
                     tags: Optional[List[str]] = None,
                     created_at: Optional[datetime] = None) -> Episode:
        episode = Episode(
            content=content, importance=importance, tags=tags or [],
            **({"created_at": created_at} if created_at is not None else {}),
        )
        self.episodes.append(episode)
        self.episodes.sort(key=lambda e: e.created_at)
        if len(self.episodes) > self.consolidation_threshold:
            self.consolidate()
        return episode

    def consolidate(self) -> Optional[MemoryBlock]:
        if len(self.episodes) < self.consolidation_batch_size:
            return None
        batch = self.episodes[: self.consolidation_batch_size]
        self.episodes = self.episodes[self.consolidation_batch_size:]
        block = MemoryBlock(
            summary=self.summarizer(batch),
            source_episode_ids=[e.id for e in batch],
            episode_count=len(batch),
            time_range=(batch[0].created_at, batch[-1].created_at),
            importance=max(e.importance for e in batch),
        )
        self.memory_blocks.append(block)
        return block

    def effective_importance(self, episode: Episode, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        elapsed = (now - episode.created_at).total_seconds()
        return exponential_decay(episode.importance, elapsed, self.half_life_seconds)

    def decay_and_forget(self, now: Optional[datetime] = None) -> List[Episode]:
        now = now or datetime.now(timezone.utc)
        keep, forgotten = [], []
        for e in self.episodes:
            (forgotten if self.effective_importance(e, now) < self.forget_threshold else keep).append(e)
        self.episodes = keep
        return forgotten

    def search(self, query: str, limit: int = 5) -> List[Episode]:
        """Very simple keyword-overlap search over the active log (most recent first)."""
        q_words = set(query.lower().split())
        scored = [
            (len(q_words & set(e.content.lower().split())), e)
            for e in self.episodes
        ]
        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [e for score, e in scored if score > 0][:limit]

    def recent(self, limit: int = 20) -> List[Episode]:
        return sorted(self.episodes, key=lambda e: e.created_at, reverse=True)[:limit]

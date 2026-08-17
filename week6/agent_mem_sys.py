"""

 AGENT MEMORY SYSTEM 

This file gives an AI agent a simple "memory" system with two features:

1. MEMORY CONSOLIDATION
   The agent stores every event as a small "Episode" (like a diary entry).
   If the agent collects too many episodes (more than 50), the oldest 25
   of them get squashed into ONE summary called a "MemoryBlock".
   This keeps memory from growing forever and endlessly.

2. IMPORTANCE-BASED FORGETTING (DECAY)
   Every episode has an "importance" score (0.0 = not important,
   1.0 = very important). As time passes, importance slowly fades
   (this is called "decay"). Low-importance memories fade fast and
   eventually get deleted ("forgotten"). High-importance memories fade
   slowly, so they stick around much longer.


FILE ARCHITECTURE


1. DATA MODELS
   - Episode      -> one single memory (raw event)
   - MemoryBlock  -> one compressed summary made from many old episodes

2. DECAY FUNCTION
   - exponential_decay() -> the math formula that fades importance over time

3. SUMMARIZER
   - default_summarizer() -> turns a batch of episodes into one summary text

4. MemoryStore CLASS  (the main engine, does everything above together)
   - add_episode()             -> add a new memory
   - consolidate()              -> squash oldest 25 episodes into 1 summary
   - decay_and_forget()         -> delete memories that faded below the limit
   - effective_importance()     -> check how "important" a memory is RIGHT NOW
   - get_context_snapshot()     -> get everything in a clean format
                                    (useful to feed into an LLM prompt)
   - stats()                    -> quick numbers: how many memories, etc.

5. DEMO / TEST CODE 
   - Shows consolidation happening automatically after 50+ episodes
   - Shows old, unimportant memories getting forgotten
   - Shows old, important memories surviving

"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Tuple



#data models

@dataclass
class Episode:
    content: str                 
    importance: float          
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))   
    tags: List[str] = field(default_factory=list)   

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"importance must be between 0.0 and 1.0, got {self.importance}")


@dataclass
class MemoryBlock:
    summary: str                       
    source_episode_ids: List[str]     
    episode_count: int                  
    time_range: Tuple[datetime, datetime] 
    importance: float                 
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))



#decay func(over time fades formula)
def exponential_decay(
    base_importance: float,
    elapsed_seconds: float,
    half_life_seconds: float = 6 * 3600,   
) -> float:

    if elapsed_seconds <= 0:
        return base_importance   


    scaled_half_life = half_life_seconds * (0.5 + base_importance)

    decay_rate = math.log(2) / scaled_half_life
    return base_importance * math.exp(-decay_rate * elapsed_seconds)



Summarizer = Callable[[List[Episode]], str]


def default_summarizer(episodes: List[Episode]) -> str:

    bullet_points = "\n".join(f"- {e.content.strip()}" for e in episodes)
    start_time = episodes[0].created_at.isoformat()
    end_time = episodes[-1].created_at.isoformat()
    return (
        f"[Summary of {len(episodes)} episodes, from {start_time} to {end_time}]\n"
        f"{bullet_points}"
    )


#main agent - memory system

class MemoryStore:


    def __init__(
        self,
        consolidation_threshold: int = 50,    
        consolidation_batch_size: int = 25,   
        forget_threshold: float = 0.05,      
        half_life_seconds: float = 6 * 3600,  
        summarizer: Summarizer = default_summarizer,
    ):
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_batch_size = consolidation_batch_size
        self.forget_threshold = forget_threshold
        self.half_life_seconds = half_life_seconds
        self.summarizer = summarizer

        self.episodes: List[Episode] = []           # active memories
        self.memory_blocks: List[MemoryBlock] = []  # compressed summaries


#ADDING NEW MEM

    def add_episode(
        self,
        content: str,
        importance: float,
        tags: Optional[List[str]] = None,
        created_at: Optional[datetime] = None,  
    ) -> Episode:
        """Add one new memory. Automatically triggers consolidation if needed."""
        episode = Episode(
            content=content,
            importance=importance,
            tags=tags or [],
            **({"created_at": created_at} if created_at is not None else {}),
        )
        self.episodes.append(episode)
        self.episodes.sort(key=lambda e: e.created_at)  # always keep oldest-first order

        # If we now have too many memories, squash the oldest ones automatically.
        if len(self.episodes) > self.consolidation_threshold:
            self.consolidate()

        return episode

#consolidation

    def consolidate(self) -> MemoryBlock:

        if len(self.episodes) < self.consolidation_batch_size:
            raise ValueError(
                f"Not enough memories to consolidate. Need "
                f"{self.consolidation_batch_size}, only have {len(self.episodes)}."
            )

        # Take the oldest batch, and remove them from the active list.
        oldest_batch = self.episodes[: self.consolidation_batch_size]
        self.episodes = self.episodes[self.consolidation_batch_size :]

        # Turn that batch into one summary text.
        summary_text = self.summarizer(oldest_batch)

        # Build the compressed MemoryBlock.
        block = MemoryBlock(
            summary=summary_text,
            source_episode_ids=[e.id for e in oldest_batch],
            episode_count=len(oldest_batch),
            time_range=(oldest_batch[0].created_at, oldest_batch[-1].created_at),
            importance=max(e.importance for e in oldest_batch),  # keep the highest score
        )
        self.memory_blocks.append(block)
        return block


#decay and forgetting

    def effective_importance(self, episode: Episode, now: Optional[datetime] = None) -> float:
  
        now = now or datetime.now(timezone.utc)
        elapsed_seconds = (now - episode.created_at).total_seconds()
        return exponential_decay(episode.importance, elapsed_seconds, self.half_life_seconds)

    def decay_and_forget(self, now: Optional[datetime] = None) -> List[Episode]:

        now = now or datetime.now(timezone.utc)
        keep_these: List[Episode] = []
        forgotten_these: List[Episode] = []

        for episode in self.episodes:
            current_importance = self.effective_importance(episode, now)
            if current_importance < self.forget_threshold:
                forgotten_these.append(episode)   # too faded, delete it
            else:
                keep_these.append(episode)        # still worth keeping

        self.episodes = keep_these
        return forgotten_these


#view mem

    def get_active_episodes_sorted_by_importance(
        self, now: Optional[datetime] = None
    ) -> List[Episode]:
        """Returns active memories sorted from most important to least important."""
        now = now or datetime.now(timezone.utc)
        return sorted(
            self.episodes,
            key=lambda e: self.effective_importance(e, now),
            reverse=True,
        )

    def get_context_snapshot(self, now: Optional[datetime] = None) -> dict:

        now = now or datetime.now(timezone.utc)
        return {
            "active_episodes": [
                {
                    "id": e.id,
                    "content": e.content,
                    "base_importance": e.importance,
                    "effective_importance": round(self.effective_importance(e, now), 4),
                    "created_at": e.created_at.isoformat(),
                }
                for e in self.episodes
            ],
            "memory_blocks": [
                {
                    "id": b.id,
                    "summary": b.summary,
                    "episode_count": b.episode_count,
                    "importance": b.importance,
                    "time_range": [t.isoformat() for t in b.time_range],
                }
                for b in self.memory_blocks
            ],
        }

    def stats(self) -> dict:
        """Quick numbers: how many active memories, how many summaries, etc."""
        return {
            "active_episode_count": len(self.episodes),
            "memory_block_count": len(self.memory_blocks),
            "total_consolidated_episodes": sum(b.episode_count for b in self.memory_blocks),
        }


#main demo

def _demo_consolidation() -> None:
    """Shows that adding more than 50 memories triggers automatic consolidation."""
    print()
    print("DEMO 1: Consolidation (triggers automatically after 50 memories)")
    print()

    store = MemoryStore(consolidation_threshold=50, consolidation_batch_size=25)

    # Add 60 memories one by one.
    for i in range(60):
        store.add_episode(
            content=f"Episode #{i}: agent did task {i}",
            importance=0.3 + (i % 5) * 0.1,   # just some varying importance values
        )

    print("Stats after adding 60 memories:", store.stats())
    print(f"Active memories left: {len(store.episodes)}  (expected: 35, since 60 - 25 = 35)")
    print(f"Summary blocks created: {len(store.memory_blocks)}  (expected: 1)")

    first_block = store.memory_blocks[0]
    print(f"\nThat one summary block covers {first_block.episode_count} old memories.")
    print("Preview of the summary text:")
    preview = first_block.summary[:250]
    print(preview + ("..." if len(first_block.summary) > 250 else ""))


def _demo_decay_and_forgetting() -> None:
    """Shows that unimportant old memories get deleted, but important ones survive."""
    print()
    print("DEMO 2: Decay & forgetting (importance fades, weak memories get deleted)")
    print()

    store = MemoryStore(half_life_seconds=6 * 3600, forget_threshold=0.05)
    now = datetime.now(timezone.utc)

    # Create a mix of old/new and important/unimportant fake memories.
    test_memories = [
        ("low_importance_old_memory",    0.1, now - timedelta(hours=10)),
        ("low_importance_fresh_memory",  0.1, now - timedelta(minutes=5)),
        ("high_importance_old_memory",   0.9, now - timedelta(hours=10)),
        ("high_importance_fresh_memory", 0.9, now - timedelta(minutes=5)),
        ("medium_importance_ancient_memory", 0.4, now - timedelta(hours=30)),
    ]

    for label, importance, created_at in test_memories:
        store.add_episode(content=label, importance=importance, created_at=created_at)

    print("\nImportance BEFORE running decay_and_forget():")
    for e in store.episodes:
        current = store.effective_importance(e, now)
        print(f"  {e.content:32s} started at {e.importance:.2f}  ->  now worth {current:.4f}")

    forgotten = store.decay_and_forget(now=now)

    print(f"\nDeleted (forgotten) memories ({len(forgotten)}):")
    for e in forgotten:
        print(f"  - {e.content}")

    print(f"\nMemories that survived ({len(store.episodes)}):")
    for e in store.episodes:
        print(f"  - {e.content}")


if __name__ == "__main__":
    _demo_consolidation()
    _demo_decay_and_forgetting()
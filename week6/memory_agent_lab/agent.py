"""
agent.py

MemoryAgent -- integrates all four memory types into one object:

    episodic    (episodic.py)         "what happened, when"
    semantic    (semantic.py)         "stable facts / profile"
    procedural  (procedural.py)       "learned behavioral corrections"
    graph       (knowledge_graph.py)  "how entities relate to each other"

Two things make this "one agent" rather than four separate stores glued
together:

  1. `observe(text)` -- a single entry point. EVERY observation always gets
     logged to episodic memory (the raw log of everything that happened).
     On top of that, simple rule-based pattern matching checks whether the
     text ALSO looks like a fact ("my name is..."), a correction
     ("always...", "never...", "from now on..."), or a relationship
     ("X works on Y") -- and if so, distills it into the matching store(s)
     too. This mirrors how memory actually works: most of what happens is
     just logged, but some of it gets extracted into more durable,
     structured knowledge.

  2. `get_context_for_query(query)` -- pulls relevant material from ALL
     FOUR stores for a given query, so a real response-generation step
     would have everything it needs in one call.

"""

from __future__ import annotations

import re
from typing import Dict, List

from episodic import EpisodicMemory, Episode
from semantic import SemanticMemory, SemanticFact
from procedural import ProceduralMemory, Correction
from knowledge_graph import KnowledgeGraphMemory


#extraction pattern 

CORRECTION_PATTERNS = [
    re.compile(r"^when (?P<trigger>.+?),\s*(?:please\s+)?(?P<instruction>.+)$", re.I),
    re.compile(r"^from now on,?\s*(?P<instruction>.+)$", re.I),
    re.compile(r"^please always\s+(?P<instruction>.+)$", re.I),
    re.compile(r"^always\s+(?P<instruction>.+)$", re.I),
    re.compile(r"^never\s+(?P<instruction>.+)$", re.I),
    re.compile(r"^remember to\s+(?P<instruction>.+)$", re.I),
    re.compile(r"^don'?t\s+(?P<instruction>.+)$", re.I),
]

FACT_PATTERNS = [
    ("name", re.compile(r"\bmy name is (?P<value>[\w\s]+)", re.I)),
    ("role", re.compile(r"\bi(?:'m| am) an? (?P<value>[\w\s]+?)(?:\.|$)", re.I)),
    ("location", re.compile(r"\bi live in (?P<value>[\w\s]+)", re.I)),
    ("employer", re.compile(r"\bi work at (?P<value>[\w\s]+)", re.I)),
    ("timezone", re.compile(r"\bmy timezone is (?P<value>[\w\s/+-]+)", re.I)),
    ("preference", re.compile(r"\bi prefer (?P<value>[\w\s]+)", re.I)),
]
FAVORITE_PATTERN = re.compile(r"\bmy favorite (?P<thing>\w+) is (?P<value>[\w\s]+)", re.I)

RELATION_PATTERNS = [
    ("works_on", re.compile(r"(?P<subj>[\w\s]+?)\s+works on\s+(?P<obj>[\w\s]+)", re.I)),
    ("uses", re.compile(r"(?P<subj>[\w\s]+?)\s+uses\s+(?P<obj>[\w\s]+)", re.I)),
    ("depends_on", re.compile(r"(?P<subj>[\w\s]+?)\s+depends on\s+(?P<obj>[\w\s]+)", re.I)),
    ("part_of", re.compile(r"(?P<subj>[\w\s]+?)\s+is part of\s+(?P<obj>[\w\s]+)", re.I)),
    ("works_with", re.compile(r"(?P<subj>[\w\s]+?)\s+works with\s+(?P<obj>[\w\s]+)", re.I)),
    ("built_by", re.compile(r"(?P<subj>[\w\s]+?)\s+was built by\s+(?P<obj>[\w\s]+)", re.I)),
]


def _clean(text: str) -> str:
    return text.strip().rstrip(".").strip()


def _normalize_entity(text: str) -> str:

    cleaned = _clean(text)
    return re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.I).strip()


class MemoryAgent:
    def __init__(self):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.graph = KnowledgeGraphMemory()



    def observe(self, text: str, importance: float = 0.5) -> Dict[str, list]:

        text = text.strip()
        if not text:
            return {}

        stored: Dict[str, list] = {"episodic": [], "semantic": [], "procedural": [], "graph": []}

        correction = self._try_extract_correction(text)
        if correction:
            stored["procedural"].append(correction)
            episode = self.episodic.add_episode(text, importance=max(importance, 0.75), tags=["correction"])
            stored["episodic"].append(episode)
            return stored  # a correction is treated as its own distinct kind of observation

        # 2) Otherwise, always log the raw episode.
        episode = self.episodic.add_episode(text, importance=importance, tags=[])
        stored["episodic"].append(episode)

        # 3) Try to extract a semantic fact.
        fact = self._try_extract_fact(text)
        if fact:
            stored["semantic"].append(fact)

        # 4) Try to extract a graph relationship.
        relation = self._try_extract_relation(text)
        if relation:
            stored["graph"].append(relation)

        return stored



    def _try_extract_correction(self, text: str):
        for pattern in CORRECTION_PATTERNS:
            m = pattern.match(text)
            if m:
                groups = m.groupdict()
                trigger = _clean(groups.get("trigger", "general"))
                instruction = _clean(groups["instruction"])
                return self.procedural.add_correction(trigger_context=trigger, instruction=instruction)
        return None

    def _try_extract_fact(self, text: str):
        m = FAVORITE_PATTERN.search(text)
        if m:
            key = f"favorite_{_clean(m.group('thing')).lower()}"
            return self.semantic.set_fact(key, _clean(m.group("value")), source="user_stated")

        for key, pattern in FACT_PATTERNS:
            m = pattern.search(text)
            if m:
                return self.semantic.set_fact(key, _clean(m.group("value")), source="user_stated")
        return None

    def _try_extract_relation(self, text: str):
        for relation_name, pattern in RELATION_PATTERNS:
            m = pattern.search(text)
            if m:
                subj = _normalize_entity(m.group("subj"))
                obj = _normalize_entity(m.group("obj"))
                if subj and obj:
                    self.graph.add_relation(subj, relation_name, obj)
                    return (subj, relation_name, obj)
        return None


    def get_context_for_query(self, query: str) -> Dict[str, list]:
        """Pulls relevant material from all four memory stores for `query`."""
        relevant_episodes = self.episodic.search(query, limit=5)
        relevant_facts = self.semantic.search(query, limit=10) or self.semantic.get_profile()
        applicable_corrections = self.procedural.applicable_to(query, limit=5) or self.procedural.get_active()

        graph_triples: List[tuple] = []
        for entity in self.graph.search_entities(query):
            graph_triples.extend(self.graph.neighbors_of(entity))

        return {
            "episodic": relevant_episodes,
            "semantic": relevant_facts,
            "procedural": applicable_corrections,
            "graph": graph_triples,
        }



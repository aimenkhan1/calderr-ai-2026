"""
knowledge_graph.py

KNOWLEDGE GRAPH MEMORY -- "how things the agent knows about are connected."
A directed graph of entities (people, projects, tools, companies...)
connected by typed relationships (e.g. "Alex" -works_on-> "Project Nimbus").
Backed by NetworkX. Complements semantic memory: semantic memory holds
flat key/value facts about ONE subject (the user), while the graph captures
relationships BETWEEN many different entities.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import networkx as nx


class KnowledgeGraphMemory:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_relation(self, subject: str, relation: str, obj: str, **attrs) -> None:
        self.graph.add_node(subject)
        self.graph.add_node(obj)
        self.graph.add_edge(
            subject, obj, relation=relation,
            added_at=datetime.now(timezone.utc).isoformat(), **attrs,
        )

    def entities(self) -> List[str]:
        return list(self.graph.nodes())

    def get_triples(self) -> List[Tuple[str, str, str]]:
        return [
            (u, data["relation"], v)
            for u, v, data in self.graph.edges(data=True)
        ]

    def neighbors_of(self, entity: str) -> List[Tuple[str, str, str]]:
        """All triples touching `entity`, either as subject or object."""
        triples = []
        if entity not in self.graph:
            return triples
        for _, v, data in self.graph.out_edges(entity, data=True):
            triples.append((entity, data["relation"], v))
        for u, _, data in self.graph.in_edges(entity, data=True):
            triples.append((u, data["relation"], entity))
        return triples

    def search_entities(self, query: str) -> List[str]:
        q_lower = query.lower()
        return [e for e in self.graph.nodes() if e.lower() in q_lower]

    def to_dot(self) -> str:
        """Renders the graph as a Graphviz DOT string, for st.graphviz_chart()."""
        lines = ["digraph KnowledgeGraph {", '  rankdir="LR";', '  node [shape=box, style="rounded,filled", fillcolor="#EEF3FF", fontname="Helvetica"];']
        for u, v, data in self.graph.edges(data=True):
            label = data.get("relation", "")
            lines.append(f'  "{u}" -> "{v}" [label="{label}", fontsize=10];')
        if not self.graph.edges():
            lines.append('  empty [label="(no relations learned yet)", shape=plaintext];')
        lines.append("}")
        return "\n".join(lines)

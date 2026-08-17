"""
graph_store.py

Wraps NetworkX to do the "graph retrieval" half of the pipeline.

How it works:
  1. Build a directed graph from GRAPH_EDGES (person/company/product/city nodes,
     connected by typed relationships like "founded", "acquired", "worked_at").
  2. Given a question, find which known entities are mentioned in it (simple
     substring matching against node names good enough for this lab; a
     production system would use a proper NER model).
  3. Starting from those "seed" entities, expand outward through the graph
     up to `max_hops` steps (a "neighbourhood expansion" / breadth-first
     search), collecting every edge touched along the way.
  4. Turn each collected edge back into a plain-English sentence, so it can
     be merged with vector-retrieved text later.
"""

from __future__ import annotations
from typing import List, Set
import networkx as nx


class GraphRetriever:
    def __init__(self, edges: List[tuple]):
        self.graph = nx.MultiDiGraph()
        for source, relation, target, attrs in edges:
            self.graph.add_edge(source, target, relation=relation, **attrs)


        self.known_entities = sorted(self.graph.nodes(), key=len, reverse=True)

    # Step 1: find which entities the question is talking about

    def extract_entities(self, question: str) -> List[str]:
        found = []
        q_lower = question.lower()
        for entity in self.known_entities:
            if entity.lower() in q_lower and entity not in found:
                found.append(entity)
        return found

    # Step 2: expand outward from those entities

    def _edge_to_sentence(self, source: str, target: str, data: dict) -> str:
        relation = data["relation"]
        extra_bits = {k: v for k, v in data.items() if k != "relation"}

        phrase_map = {
            "founded": f"{source} founded {target}",
            "co_founded": f"{source} co-founded {target}",
            "acquired": f"{source} acquired {target}",
            "created_product": f"{source} created the product {target}",
            "based_in": f"{source} is based in {target}",
            "left": f"{source} left {target}",
            "worked_at": f"{source} worked at {target}",
            "works_at": f"{source} works at {target}",
            "partnered_with": f"{source} partnered with {target}",
        }
        sentence = phrase_map.get(relation, f"{source} {relation} {target}")

        if extra_bits:
            details = ", ".join(f"{k}={v}" for k, v in extra_bits.items())
            sentence += f" ({details})"
        return sentence + "."

    def expand_neighborhood(self, seed_entities: List[str], max_hops: int = 2) -> List[str]:
        """
        Breadth-first expansion from the seed entities. Collects every edge
        (in either direction) touched within `max_hops` steps, and returns
        them as plain-English sentences (deduplicated, seed entities first).
        """
        visited_nodes: Set[str] = set(seed_entities)
        frontier: Set[str] = set(seed_entities)
        collected_sentences: List[str] = []
        seen_edges: Set[tuple] = set()

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node not in self.graph:
                    continue
                # outgoing edges: node -> neighbor
                for _, neighbor, data in self.graph.out_edges(node, data=True):
                    edge_key = (node, data["relation"], neighbor)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        collected_sentences.append(self._edge_to_sentence(node, neighbor, data))
                    if neighbor not in visited_nodes:
                        next_frontier.add(neighbor)
                # incoming edges: neighbor -> node
                for neighbor, _, data in self.graph.in_edges(node, data=True):
                    edge_key = (neighbor, data["relation"], node)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        collected_sentences.append(self._edge_to_sentence(neighbor, node, data))
                    if neighbor not in visited_nodes:
                        next_frontier.add(neighbor)

            visited_nodes |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        return collected_sentences

    #main point

    def retrieve(self, question: str, max_hops: int = 2) -> List[dict]:

        seeds = self.extract_entities(question)
        if not seeds:
            return []

        sentences = self.expand_neighborhood(seeds, max_hops=max_hops)
        out = []
        for i, sentence in enumerate(sentences):
            out.append({"id": f"g{i}", "text": sentence, "score": round(1.0 - i * 0.02, 4)})
        return out


if __name__ == "__main__":
    # Quick test.
    from domain_data import GRAPH_EDGES

    retriever = GraphRetriever(GRAPH_EDGES)
    for q in [
        "Who co-founded QuantumLeap Robotics alongside Marcus Bell?",
        "What product did the company founded by the person who previously worked at Verdant Energy Co go on to create?",
    ]:
        print(f"\nQuery: {q}")
        print("Seed entities:", retriever.extract_entities(q))
        for hit in retriever.retrieve(q, max_hops=2):
            print(f"  [{hit['score']}] {hit['text']}")

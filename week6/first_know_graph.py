"""
Knowledge Graph Builder — 20 paragraphs -> entities -> relationships -> graph

Two things happen to the paragraphs:
  1. an LLM (or an offline rule-based fallback) reads them and pulls out
     structured entities and relationships, validated against Pydantic
     models — so the extraction can never come back as "some free text",
     always a clean typed object.
  2. those entities/relationships get added into a NetworkX MultiDiGraph
     — MultiDiGraph because two entities can be connected by more than
     one kind of relationship (e.g. Einstein "born_in" Germany AND
     "emigrated_from" Germany), and direction matters (X founded Y is not
     the same as Y founded X).

architecture:

20 paragraphs -> ONE batched call to call_groq_extract_all() (all 20 sent
in a single request, not 20 separate ones) -> LLM returns BatchExtraction
(Pydantic) with one ParagraphExtraction per paragraph -> add_to_graph()
per paragraph -> nodes get merged by name (so "Einstein" mentioned in 5
paragraphs is still ONE node) -> edges carry relation_type + which
paragraph they came from -> final MultiDiGraph

Why batched instead of one call per paragraph: 20 separate Groq calls
means 20 separate network round trips. Each one has its own latency, and
if any single call is a bit slow or gets rate-limited, that delay stacks
on top of the other 19 -- that's what was timing out. One call carrying
all 20 paragraphs in the prompt is a SINGLE round trip, so there's
nothing to stack.

analysis -> uses NetworkX functions/algorithms (degree, centrality) to show
which entities are most central to the extracted knowledge graph

LLM backend: real Groq if GROQ_API_KEY is set, else an offline rule-based
extractor (per paragraph, no network calls at all), so this can be built
and tested without an API key or internet.

paragraph source: real Wikipedia fetch if requests/internet available,
else a bundled offline dataset (sample_para_of_first_know_graph.py) of 20
original paragraphs about an interconnected topic cluster (Einstein/
Curie/Nobel Prize/Princeton), so this file always has something to run on.
"""

import os
import re
import json
import urllib.request               
from typing import Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv           

from sample_para_of_first_know_graph import SAMPLE_PARAGRAPHS

load_dotenv()


try:
    from groq import Groq
    _GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else None
except ImportError:
    client = None

MODEL = "llama-3.1-8b-instant"

ENTITY_TYPES = Literal["person", "organization", "location", "concept", "event", "other"]
RELATION_TYPES = Literal[
    "born_in", "died_in", "works_at", "studied_at", "founded",
    "located_in", "part_of", "awarded", "discovered", "married_to",
    "related_to",
]


#pydantic models 
class Entity(BaseModel):
    name: str
    entity_type: ENTITY_TYPES = "other"


class Relationship(BaseModel):
    source: str = Field(..., description="the entity the relationship starts from")
    target: str = Field(..., description="the entity the relationship points to")
    relation_type: RELATION_TYPES = "related_to"



class ParagraphExtraction(BaseModel):
    paragraph: int
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


class BatchExtraction(BaseModel):
    results: list[ParagraphExtraction] = Field(default_factory=list)



def fetch_wikipedia_paragraphs(topics: list[str]) -> list[str]:
    paragraphs = []
    try:
        for topic in topics:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            extract = data.get("extract", "").strip()
            if extract:
                paragraphs.append(extract)
        if len(paragraphs) < len(topics):
            raise RuntimeError("some topics returned no extract")
        return paragraphs
    except Exception:
        print("[offline] Wikipedia fetch unavailable — using bundled sample paragraphs.\n")
        return SAMPLE_PARAGRAPHS




EXTRACTION_SYSTEM_PROMPT = """You will receive MULTIPLE numbered paragraphs.

For EACH paragraph, extract:
- entities
- relationships

Return ONLY valid JSON matching this exact shape, nothing else:

{"results": [
  {"paragraph": 1,
   "entities": [{"name": "...", "entity_type": "person"|"organization"|"location"|"concept"|"event"|"other"}],
   "relationships": [{"source": "...", "target": "...", "relation_type": "born_in"|"died_in"|"works_at"|"studied_at"|"founded"|"located_in"|"part_of"|"awarded"|"discovered"|"married_to"|"related_to"}]},
  {"paragraph": 2, "entities": [...], "relationships": [...]}
]}

Include one result object per paragraph number, even if it has no
entities/relationships (use empty lists). Only extract entities/
relationships actually stated in the text. Entity names in "source"/
"target" must exactly match a name in that paragraph's "entities".
"""


def call_groq_extract_all(paragraphs: list[str]) -> BatchExtraction:

    text = ""
    for i, p in enumerate(paragraphs, start=1):
        text += f"Paragraph {i}:\n{p}\n\n"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = response.choices[0].message.content
    return BatchExtraction.model_validate_json(raw)


def stub_extract(paragraph: str, paragraph_index: int) -> ParagraphExtraction:
    STOPWORDS_START = {"The", "In", "He", "She", "They", "It", "His", "Her", "This"}

    entity_pattern = re.compile(
        r"\b[A-Z][a-zA-Z]+(?:\s+(?:of|for|the|and)\s+[A-Z][a-zA-Z]+|\s+[A-Z][a-zA-Z]+)*\b"
    )

    mentions = []   # list of (name, start_pos)
    names = set()
    for m in entity_pattern.finditer(paragraph):
        name = m.group().strip()
        first_word = name.split()[0] if name.split() else ""
        if len(name) < 3 or first_word in STOPWORDS_START:
            continue
        mentions.append((name, m.start()))
        names.add(name)

    entities = []
    for name in names:
        lower = name.lower()
        if any(w in lower for w in ["university", "institute", "academy", "prize", "society"]):
            etype = "organization"
        elif any(w in lower for w in ["germany", "poland", "sweden", "france", "switzerland",
                                        "paris", "stockholm", "princeton", "zurich", "warsaw",
                                        "new jersey", "united states"]):
            etype = "location"
        elif any(w in lower for w in ["relativity", "mechanics", "gravitation", "radioactivity",
                                        "physics", "chemistry"]):
            etype = "concept"
        else:
            etype = "person"
        entities.append(Entity(name=name, entity_type=etype))

    relationships = []
    patterns = [
        (r"born in", "born_in"),
        (r"died in", "died_in"),
        (r"works? at|worked at|joined", "works_at"),
        (r"studied at|trained at", "studied_at"),
        (r"founded|established", "founded"),
        (r"located in|based in", "located_in"),
        (r"part of|separate institution", "part_of"),
        (r"won|awarded|received", "awarded"),
        (r"discover(?:ed|y)", "discovered"),
        (r"married|husband|wife", "married_to"),
    ]
    for cue_pattern, relation_type in patterns:
        for m in re.finditer(cue_pattern, paragraph, re.IGNORECASE):
            before = [n for n in mentions if n[1] < m.start()]
            after = [n for n in mentions if n[1] >= m.end()]
            src = max(before, key=lambda x: x[1])[0] if before else None   # closest before
            tgt = min(after, key=lambda x: x[1])[0] if after else None     # closest after
            if src and tgt and src != tgt:
                relationships.append(Relationship(source=src, target=tgt, relation_type=relation_type))

    return ParagraphExtraction(paragraph=paragraph_index, entities=entities, relationships=relationships)


def extract_all(paragraphs: list[str]) -> BatchExtraction:

    if client is not None:
        try:
            return call_groq_extract_all(paragraphs)
        except (ValidationError, Exception) as e:
            print(f"[offline] Groq batch extraction failed ({e}) — falling back to rule-based extraction.\n")
    return BatchExtraction(results=[
        stub_extract(p, i) for i, p in enumerate(paragraphs, start=1)
    ])


#graph building

def add_to_graph(G: nx.MultiDiGraph, result: ParagraphExtraction) -> None:
    for entity in result.entities:
        # merge by name: if this entity already exists (from an earlier
        # paragraph), don't overwrite it with a duplicate node
        if G.has_node(entity.name):
            continue
        G.add_node(entity.name, entity_type=entity.entity_type)

    for rel in result.relationships:
        # make sure both endpoints exist as nodes even if the entity list
        # missed one of them
        if not G.has_node(rel.source):
            G.add_node(rel.source, entity_type="other")
        if not G.has_node(rel.target):
            G.add_node(rel.target, entity_type="other")

        G.add_edge(
            rel.source, rel.target,
            relation_type=rel.relation_type,
            paragraph=result.paragraph,
        )


def build_knowledge_graph(paragraphs: list[str]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    batch = extract_all(paragraphs)
    for result in batch.results:
        add_to_graph(G, result)
    return G


#analysis of graph thru network x func

def print_graph_summary(G: nx.MultiDiGraph) -> None:
    print(f"Nodes (entities): {G.number_of_nodes()}")
    print(f"Edges (relationships): {G.number_of_edges()}\n")

    print("--- Entity type breakdown ---")
    type_counts = {}
    for _, etype in G.nodes(data="entity_type"):
        type_counts[etype] = type_counts.get(etype, 0) + 1
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")

    print("\n--- Most central entities (degree centrality) ---")
    centrality = nx.degree_centrality(G)
    top = sorted(centrality.items(), key=lambda x: -x[1])[:5]
    for name, score in top:
        etype = G.nodes[name]["entity_type"]
        print(f"  {name} ({etype}): {score:.3f}")

    print("\n--- Sample relationships ---")
    for u, v, data in list(G.edges(data=True))[:10]:
        print(f"  {u} --[{data['relation_type']}]--> {v}  (from paragraph {data['paragraph']})")


#main

if __name__ == "__main__":
    topics = [
        "Albert Einstein", "Marie Curie", "Nobel Prize",
        "Princeton University", "Theory of relativity",
    ]

    print("===== Loading 20 paragraphs =====")
    paragraphs = fetch_wikipedia_paragraphs(topics)
    # if the live fetch only returned a handful of intro paragraphs
    # (one per topic), pad out to 20 using the offline dataset
    if len(paragraphs) < 20:
        paragraphs = SAMPLE_PARAGRAPHS
    print(f"Loaded {len(paragraphs)} paragraphs.\n")

    print("===== Extracting entities + relationships (one batched call), building graph =====")
    print(f"Extraction backend: {'Groq (batched)' if client is not None else 'offline rule-based'}\n")

    G = build_knowledge_graph(paragraphs)

    print("===== Knowledge graph summary =====")
    print_graph_summary(G)
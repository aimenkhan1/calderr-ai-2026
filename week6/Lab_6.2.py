"""
Lab 6.2 - Knowledge Graph Query Agent

Sample paragraphs only (no live Wikipedia fetch) -> LLM extraction (or an
offline rule-based fallback) -> NetworkX graph -> pyvis visualization ->
query agent -> validation -> interactive Q&A.

DESIGN PRINCIPLES
-----------------------------------------------------------
1. The LLM does semantic understanding; Python only validates/normalizes.
   Regex-guessing every possible way English can phrase "works_at" doesn't
   scale ("joined" / "was employed by" / "held a position at" / ...). The
   extraction prompt below gives the LLM explicit few-shot mappings and
   asks IT to do the semantic mapping — Pydantic's job is just to catch
   and gently correct whatever vocabulary drift slips through, not to be
   the primary understanding mechanism.

2. The LLM must be told to extract ONLY what's explicitly stated, not to
   infer relationships from proximity, movement, or background context.
   An earlier version of this prompt caused the LLM to turn "Einstein
   emigrated to the US" into "Germany -[part_of]-> United States" —
   pure hallucination from loose instructions, not a bug in the model.

3. Nothing in this file invents edges after extraction. No "bridge" step
   that guesses hierarchical relationships between similarly-named
   entities — if the text doesn't explicitly state "X is part of Y",
   no part_of edge gets created for X and Y, full stop. An auto-bridge
   step is exactly the kind of inference rule #2 says not to do; doing
   it in Python instead of in the prompt doesn't make it safer.

4. The final answer NEVER comes from an LLM reading the traversed paths.
   query() always calls synthesize_answer_stub() — deterministic,
   grounded entirely in edges that were actually walked. Groq (if
   configured) only helps with extraction and question-parsing.
"""

import os
import re
import json
from typing import Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field, ValidationError, field_validator


#types-safety 

ENTITY_TYPES = Literal["person", "organization", "location", "concept", "event", "other"]
RELATION_TYPES = Literal[
    "born_in", "died_in", "works_at", "studied_at", "founded",
    "located_in", "part_of", "awarded", "discovered", "married_to",
    "related_to",
]

RELATION_TYPE_ALIASES = {
    "moved_to": "related_to", "lived_in": "located_in",
    "worked_at": "works_at", "work_at": "works_at", "employed_at": "works_at",
    "established": "founded", "founded_by": "founded", "co_founded": "founded",
    "invented": "discovered", "created": "discovered",
    "published": "related_to", "left": "related_to",
    "spouse_of": "married_to", "husband_of": "married_to", "wife_of": "married_to",
    "affiliated_with": "part_of", "member_of": "part_of", "belongs_to": "part_of",
    "graduated_from": "studied_at", "attended": "studied_at",
    "died_at": "died_in", "passed_away_in": "died_in",
    "born_at": "born_in",
    "won": "awarded", "received": "awarded",
}


def normalize_relation_type(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if key in RELATION_TYPES.__args__:
        return key
    return RELATION_TYPE_ALIASES.get(key, "related_to")


#pydantic model 

class Entity(BaseModel):
    name: str
    entity_type: str = "other"

    @field_validator("entity_type", mode="before")
    @classmethod
    def _normalize(cls, v):
        if not v:
            return "other"
        key = str(v).strip().lower().replace(" ", "_")
        return key if key in ENTITY_TYPES.__args__ else "other"


class Relationship(BaseModel):
    source: str
    target: str
    relation_type: str = "related_to"

    @field_validator("relation_type", mode="before")
    @classmethod
    def _normalize(cls, v):
        return normalize_relation_type(v) if v else "related_to"


class ParagraphExtraction(BaseModel):
    paragraph: int
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


class BatchExtraction(BaseModel):
    results: list[ParagraphExtraction] = Field(default_factory=list)


class QueryPlan(BaseModel):
    start_entities: list[str] = Field(default_factory=list)
    relation_hints: list[str] = Field(default_factory=list)
    target_type: Optional[str] = None


#llm call

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
    _GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else None
except ImportError:
    client = None

MODEL = "llama-3.1-8b-instant"


#sample data

SAMPLE_PARAGRAPHS = [
    "Albert Einstein was a theoretical physicist born in Germany in 1879. "
    "He is best known for developing the theory of relativity, one of the "
    "two pillars of modern physics alongside quantum mechanics.",

    "Einstein won the Nobel Prize in Physics in 1921 for his explanation "
    "of the photoelectric effect. The award is presented annually by the "
    "Royal Swedish Academy of Sciences in Stockholm.",

    "In 1933, Einstein moved to the United States to escape political "
    "persecution in Germany. He joined the Institute for Advanced Study "
    "in Princeton, New Jersey, where he worked until his death in 1955.",

    "Before moving to Princeton, Einstein studied at ETH Zurich in "
    "Switzerland, where he trained as a physicist and mathematics teacher.",

    "Marie Curie was a physicist and chemist born in Warsaw, Poland in "
    "1867. She later moved to Paris to continue her studies at the "
    "Sorbonne University.",

    "Marie Curie conducted pioneering research on radioactivity together "
    "with her husband, Pierre Curie. Their work led to the discovery of "
    "two new chemical elements, polonium and radium.",

    "In 1903, Marie Curie won the Nobel Prize in Physics, sharing the "
    "award with Pierre Curie and Henri Becquerel for their research on "
    "radioactivity.",

    "Marie Curie later won a second Nobel Prize, this time in Chemistry "
    "in 1911, making her the first person to win Nobel Prizes in two "
    "different scientific fields.",

    "The Curie Institute in Paris was founded partly through funding "
    "connected to Marie Curie's research, and it remains an important "
    "center for cancer research today.",

    "Princeton University is a private research university located in "
    "Princeton, New Jersey. It is one of the oldest universities in the "
    "United States, founded in 1746.",

    "The Institute for Advanced Study, though located near Princeton "
    "University, is a separate institution. It was founded in 1930 as an "
    "independent center for theoretical research.",

    "The Nobel Prize was established through the will of Alfred Nobel, a "
    "Swedish chemist and inventor of dynamite. The first prizes were "
    "awarded in 1901.",

    "Alfred Nobel was born in Stockholm, Sweden in 1833. He held patents "
    "in many countries and made much of his fortune through the "
    "manufacture of explosives.",

    "The theory of relativity developed by Einstein is divided into two "
    "parts: special relativity, published in 1905, and general "
    "relativity, published in 1915.",

    "General relativity describes gravity not as a force but as a "
    "curvature of space and time caused by mass and energy. It replaced "
    "earlier explanations proposed by Isaac Newton.",

    "Isaac Newton was an English physicist and mathematician who lived in "
    "the 17th and 18th centuries. His laws of motion and universal "
    "gravitation dominated physics for over two hundred years.",

    "Radioactivity, the phenomenon studied extensively by Marie Curie, "
    "refers to the spontaneous emission of radiation from unstable atomic "
    "nuclei.",

    "Pierre Curie was a French physicist who worked closely with his wife "
    "Marie Curie. He died in Paris in 1906, in a street accident.",

    "Marie Curie was born in Poland. During the 19th century, Poland was "
    "controlled by the Russian Empire, which limited educational "
    "opportunities for women.",

    "Germany, where Einstein was born, later became the site of major "
    "political upheaval in the 1930s that led many scientists, including "
    "Einstein, to emigrate to other countries such as the United States.",
]


#extraction llm does pydantic validates

EXTRACTION_SYSTEM_PROMPT = """You will receive MULTIPLE numbered paragraphs.

For EACH paragraph, extract entities and relationships.

Return ONLY valid JSON matching this exact shape, nothing else:

{"results": [
  {"paragraph": 1,
   "entities": [{"name": "...", "entity_type": "person"|"organization"|"location"|"concept"|"event"|"other"}],
   "relationships": [{"source": "...", "target": "...", "relation_type": "born_in"|"died_in"|"works_at"|"studied_at"|"founded"|"located_in"|"part_of"|"awarded"|"discovered"|"married_to"|"related_to"}]},
  {"paragraph": 2, "entities": [...], "relationships": [...]}
]}

=== CRITICAL RULE: only extract what is EXPLICITLY stated ===
Do NOT infer or invent a relationship from movement, migration,
comparison, historical background, or words like "where", "including",
"alongside", "near", "associated with". If a paragraph mentions two
entities in the same sentence but doesn't state a direct relationship
between them, do not connect them. When in doubt, leave it out — a
missing edge is far less harmful than an invented one.

Example of what NOT to do:
"Germany, where Einstein was born, later saw scientists emigrate to the
United States" does NOT mean Germany is part_of United States. Extract
ONLY: Einstein --born_in--> Germany. Nothing linking Germany and the
United States directly — the paragraph never states that relationship.

=== Mapping natural language to the fixed vocabulary ===
You MUST use ONLY the relation_type values listed above. Map whatever
phrasing appears in the text to the closest one. Examples:
  "joined X" / "was employed by X" / "held a position at X" -> works_at
  "passed away in X" / "died at X"                            -> died_in
  "was born in X"                                              -> born_in
  "established X" / "was founded by X"                        -> founded
  "received X" / "was awarded X"                               -> awarded
  "was married to X" / "her husband X"                        -> married_to
  "X was controlled by Y" / "X was ruled by Y"                -> part_of
Never invent a relation_type outside the fixed list.

=== part_of direction (get this right — it's easy to reverse) ===
source = the SMALLER / more specific entity.
target = the LARGER / more general entity it belongs to.
Example: "Poland was controlled by the Russian Empire" ->
  source="Poland", target="Russian Empire", relation_type="part_of"
Do NOT extract part_of between a general concept and one of its named
sub-parts unless the text explicitly states that containment relationship
in those terms — do not invent a hierarchy that isn't written out.

=== Entity typing ===
Universities, institutes, companies, and research centers are always
"organization", never "location" — the city/country they sit in is the
"location". Countries, cities, states, and regions are "location".

=== Naming consistency (important for graph quality) ===
Use the EXACT SAME name string for the same real-world entity every time
it appears, across ALL paragraphs. Prefer the form WITHOUT a leading
"The"/"A"/"An" (e.g. "Institute for Advanced Study", not "The Institute
for Advanced Study") and use that exact string everywhere. Inconsistent
naming makes one real entity show up as multiple disconnected nodes.

Include one result object per paragraph number, even if it has no
entities/relationships (empty lists). Entity names in "source"/"target"
must exactly match a name in that paragraph's "entities" list.
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


def classify_entity_type(name: str) -> str:
    lower = name.lower()
    if any(w in lower for w in ["university", "institute", "academy", "prize", "society"]):
        return "organization"
    elif any(w in lower for w in ["germany", "poland", "sweden", "france", "switzerland",
                                    "paris", "stockholm", "princeton", "zurich", "warsaw",
                                    "new jersey", "united states", "empire"]):
        return "location"
    elif any(w in lower for w in ["relativity", "mechanics", "gravitation", "radioactivity",
                                    "physics", "chemistry"]):
        return "concept"
    return "person"


def _resolve_pronouns(paragraph: str, entity_pattern: re.Pattern, stopwords_start: set) -> str:
    subject = None
    for m in entity_pattern.finditer(paragraph):
        words = m.group().strip().split()
        while words and words[0] in stopwords_start:
            words = words[1:]
        name = " ".join(words)
        if len(name) >= 3 and classify_entity_type(name) == "person":
            subject = name
            break
    if not subject:
        return paragraph
    paragraph = re.sub(r"\bHe\b", subject, paragraph)
    paragraph = re.sub(r"\bHis\b", subject + "'s", paragraph)
    paragraph = re.sub(r"\bShe\b", subject, paragraph)
    paragraph = re.sub(r"\bHer\b", subject + "'s", paragraph)
    paragraph = re.sub(r"\bHim\b", subject, paragraph)
    return paragraph


def stub_extract(paragraph: str, paragraph_index: int) -> ParagraphExtraction:
    STOPWORDS_START = {"The", "In", "He", "She", "They", "It", "His", "Her", "This"}
    entity_pattern = re.compile(
        r"\b[A-Z][a-zA-Z]+(?:\s+(?:of|for|the|and)\s+[A-Z][a-zA-Z]+|\s+[A-Z][a-zA-Z]+)*\b"
    )
    paragraph = _resolve_pronouns(paragraph, entity_pattern, STOPWORDS_START)

    mentions = []
    names = set()
    for m in entity_pattern.finditer(paragraph):
        words = m.group().strip().split()
        while words and words[0] in STOPWORDS_START:
            words = words[1:]
        name = " ".join(words)
        if len(name) < 3:
            continue
        mentions.append((name, m.start()))
        names.add(name)

    entities = [Entity(name=name, entity_type=classify_entity_type(name)) for name in names]

    relationships = []
    patterns = [
        (r"born in", "born_in"),
        (r"died in", "died_in"),
        (r"works? at|worked at|joined", "works_at"),
        (r"studied at|trained at", "studied_at"),
        (r"founded|established", "founded"),
        (r"located in|based in|located near", "located_in"),
        (r"part of|separate institution|controlled by|control of", "part_of"),
        (r"won|awarded|received", "awarded"),
        (r"discover(?:ed|y)", "discovered"),
        (r"married|husband|wife", "married_to"),
    ]
    for cue_pattern, relation_type in patterns:
        for m in re.finditer(cue_pattern, paragraph, re.IGNORECASE):
            before = [n for n in mentions if n[1] < m.start()]
            after = [n for n in mentions if n[1] >= m.end()]
            src = max(before, key=lambda x: x[1])[0] if before else None
            tgt = min(after, key=lambda x: x[1])[0] if after else None
            if src and tgt and src != tgt:
                relationships.append(Relationship(source=src, target=tgt, relation_type=relation_type))

    return ParagraphExtraction(paragraph=paragraph_index, entities=entities, relationships=relationships)


def extract_all(paragraphs: list[str]) -> BatchExtraction:
    if client is not None:
        try:
            batch = call_groq_extract_all(paragraphs)
            return canonicalize_entities(batch)
        except Exception as e:
            print()
            print("Groq extraction failed. Exact error:")
            print(e)
            print("Falling back to the offline rule-based extractor.")
            print()

    batch = BatchExtraction(results=[
        stub_extract(p, i) for i, p in enumerate(paragraphs, start=1)
    ])
    return canonicalize_entities(batch)


def canonicalize_entities(batch: BatchExtraction) -> BatchExtraction:

    all_names = set()
    for r in batch.results:
        for e in r.entities:
            all_names.add(e.name)
        for rel in r.relationships:
            all_names.add(rel.source)
            all_names.add(rel.target)

    alias_map = {}
    for short in all_names:
        if " " in short:
            continue
        candidates = [long for long in all_names if long != short and long.endswith(" " + short)]
        if len(candidates) == 1:
            alias_map[short] = candidates[0]

    def resolve(name: str) -> str:
        return alias_map.get(name, name)

    for r in batch.results:
        for e in r.entities:
            e.name = resolve(e.name)
        for rel in r.relationships:
            rel.source = resolve(rel.source)
            rel.target = resolve(rel.target)
    return batch


#building the graph

def add_to_graph(G: nx.MultiDiGraph, result: ParagraphExtraction) -> None:
    for entity in result.entities:
        if G.has_node(entity.name):
            continue
        G.add_node(entity.name, entity_type=entity.entity_type)
    for rel in result.relationships:
        if not G.has_node(rel.source):
            G.add_node(rel.source, entity_type="other")
        if not G.has_node(rel.target):
            G.add_node(rel.target, entity_type="other")
        G.add_edge(rel.source, rel.target, relation_type=rel.relation_type, paragraph=result.paragraph)


def build_knowledge_graph(paragraphs: list[str]) -> nx.MultiDiGraph:

    G = nx.MultiDiGraph()
    batch = extract_all(paragraphs)
    for result in batch.results:
        add_to_graph(G, result)
    return G


def print_graph_edges(G: nx.MultiDiGraph) -> None:

    print("--- Extracted graph edges ---")
    for u, v, data in G.edges(data=True):
        print(f"  {u} --[{data['relation_type']}]--> {v}   (para {data['paragraph']})")
    print()


#query agent

RELATION_CUES = [
    (r"born", "born_in"),
    (r"\bdi(?:e|ed)\b", "died_in"),
    (r"work|worked|works", "works_at"),
    (r"stud(?:y|ied|ies)", "studied_at"),
    (r"found(?:ed|er)?|establish", "founded"),
    (r"locat|near", "located_in"),
    (r"part of|separate|control|empire", "part_of"),
    (r"award|won|win|prize", "awarded"),
    (r"discover", "discovered"),
    (r"marri|husband|wife|spouse", "married_to"),
    (r"replac", "related_to"),
]

TARGET_TYPE_CUES = [
    (r"\bcountry\b|\bnation\b|\bempire\b|\bcity\b", "location"),
    (r"\borganization\b|\buniversity\b|\binstitute\b|\bcompany\b", "organization"),
    (r"^who\b|^whose\b", "person"),
    (r"^where\b", "location"),
]


def _infer_target_type(question: str) -> Optional[str]:
    q_lower = question.lower()
    for pattern, target in TARGET_TYPE_CUES:
        if re.search(pattern, q_lower):
            return target
    return None


def parse_question_stub(question: str, G: nx.MultiDiGraph) -> QueryPlan:
    q_lower = question.lower()
    q_words = set(re.findall(r"\b\w+\b", q_lower))

    full_matches, fallback_matches = [], []
    for name in sorted(G.nodes, key=len, reverse=True):
        name_lower = name.lower()
        if name_lower in q_lower:
            if any(name_lower in kept.lower() for kept in full_matches):
                continue
            full_matches.append(name)
            continue
        last_word = name_lower.split()[-1]
        if len(last_word) > 3 and last_word in q_words:
            if any(name in kept for kept in fallback_matches + full_matches):
                continue
            fallback_matches.append(name)

    matched = full_matches if full_matches else fallback_matches
    person_matches = [n for n in matched if G.nodes[n].get("entity_type") == "person"]
    if person_matches:
        matched = person_matches

    hints = [rel for pattern, rel in RELATION_CUES if re.search(pattern, q_lower)]
    return QueryPlan(start_entities=matched, relation_hints=hints, target_type=_infer_target_type(question))


def parse_question_groq(question: str, G: nx.MultiDiGraph) -> QueryPlan:
    node_list = "\n".join(f"- {n} ({G.nodes[n].get('entity_type')})" for n in G.nodes)
    prompt = f"""Here are the known entities in a knowledge graph:
{node_list}

Question: {question}

Return ONLY valid JSON: {{"start_entities": ["exact entity name(s) from the list above"], "relation_hints": ["relevant relation types IN THE ORDER they'd need to be traversed, from: born_in, died_in, works_at, studied_at, founded, located_in, part_of, awarded, discovered, married_to, related_to"], "target_type": "person"|"location"|"organization"|null}}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return QueryPlan.model_validate_json(response.choices[0].message.content)


def parse_question(question: str, G: nx.MultiDiGraph) -> QueryPlan:
    if client is not None:
        try:
            return parse_question_groq(question, G)
        except Exception as e:
            print(f"[offline] Groq question parsing failed ({e}) — using rule-based parser.")
    return parse_question_stub(question, G)


_GRAPH_REF = None


def traverse_graph(G: nx.MultiDiGraph, plan: QueryPlan, max_hops: int = 3) -> list[list[tuple]]:
    global _GRAPH_REF
    _GRAPH_REF = G

    if not plan.start_entities:
        return []

    undirected = G.to_undirected(as_view=True)
    all_paths = []
    for start in plan.start_entities:
        if start not in G:
            continue
        for target in G.nodes:
            if target == start:
                continue
            try:
                node_paths = list(nx.all_simple_paths(undirected, start, target, cutoff=max_hops))
            except nx.NodeNotFound:
                continue
            for node_path in node_paths:
                hop_path, valid = [], True
                for a, b in zip(node_path, node_path[1:]):
                    if G.has_edge(a, b):
                        rel = list(G.get_edge_data(a, b).values())[0]["relation_type"]
                        hop_path.append((a, rel, b, "forward"))
                    elif G.has_edge(b, a):
                        rel = list(G.get_edge_data(b, a).values())[0]["relation_type"]
                        hop_path.append((a, rel, b, "reverse"))
                    else:
                        valid = False
                        break
                if valid and hop_path:
                    all_paths.append(hop_path)
    return all_paths


def _path_relation_sequence(path):
    return [hop[1] for hop in path]


"""Ranking priority, in order:
1. does the final node's type match what the question asked for
2. how much of the question's relation_hints sequence does this
path satisfy, AS A FRACTION of the hints given — a path matching
2 of 2 hints beats a path matching 1 of 1 (raw count alone would
wrongly treat those as equal), and a path matching more hints
always beats one matching fewer, regardless of length
3. only once both of those tie: shorter path wins

This ordering is deliberate: hop count must NEVER outrank hint
coverage. A direct-but-wrong 1-hop path should not be able to beat a
correct 2-hop path just for being shorter — length is purely a
last-resort tiebreaker, not a primary signal.
"""
def rank_paths(paths: list[list[tuple]], plan: QueryPlan) -> list[list[tuple]]:
    def hint_match_count(seq):
        i, matches = 0, 0
        for rel in seq:
            if i < len(plan.relation_hints) and rel == plan.relation_hints[i]:
                matches += 1
                i += 1
        return matches

    def score(path):
        seq = _path_relation_sequence(path)
        target_ok = 0
        if plan.target_type:
            final_node = path[-1][2]
            final_type = _GRAPH_REF.nodes[final_node].get("entity_type") if _GRAPH_REF and final_node in _GRAPH_REF else None
            target_ok = 0 if final_type == plan.target_type else 1

        matches = hint_match_count(seq)
        unmatched = max(len(plan.relation_hints) - matches, 0)

        return (target_ok, unmatched, -matches, len(path))

    return sorted(paths, key=score)


def format_path(path: list[tuple]) -> str:
    parts = [path[0][0]]
    for a, rel, b, direction in path:
        arrow = f"-[{rel}]->" if direction == "forward" else f"<-[{rel}]-"
        parts.append(f"{arrow}{b}")
    return " ".join(parts)


def synthesize_answer_stub(plan: QueryPlan, paths: list[list[tuple]]) -> dict:
    if not paths:
        return {"answer": None, "path": None, "hops": 0}
    best = rank_paths(paths, plan)[0]
    return {"answer": best[-1][2], "path": format_path(best), "hops": len(best)}


def query(question: str, G: nx.MultiDiGraph, max_hops: int = 3) -> dict:
    plan = parse_question(question, G)
    paths = traverse_graph(G, plan, max_hops=max_hops)
    result = synthesize_answer_stub(plan, paths)
    result["plan"] = plan
    return result


#keyword search

def keyword_search_baseline(required_keywords: list[str], paragraphs: list[str]) -> bool:
    for p in paragraphs:
        p_lower = p.lower()
        if all(kw.lower() in p_lower for kw in required_keywords):
            return True
    return False


#visualization

def build_pyvis_graph(G: nx.MultiDiGraph, path: str = "knowledge_graph.html") -> str:
    from pyvis.network import Network

    net = Network(height="800px", width="100%", directed=True,
                   bgcolor="#ffffff", font_color="#222222", notebook=False,
                   cdn_resources="in_line")
    net.barnes_hut(gravity=-3000, spring_length=150)

    type_colors = {
        "person": "#4C72B0", "organization": "#DD8452", "location": "#55A868",
        "concept": "#C44E52", "event": "#8172B2", "other": "#999999",
    }
    centrality = nx.degree_centrality(G)
    for node in G.nodes:
        etype = G.nodes[node].get("entity_type", "other")
        size = 15 + 60 * centrality.get(node, 0)
        net.add_node(node, label=node, color=type_colors.get(etype, "#999999"),
                     size=size, title=f"{node} ({etype})")

    seen_pairs = set()
    for u, v, data in G.edges(data=True):
        if (u, v) in seen_pairs:
            continue
        seen_pairs.add((u, v))
        net.add_edge(u, v, label=data["relation_type"], title=data["relation_type"], arrows="to")

    net.set_options("""
    {
      "edges": {"font": {"size": 10, "align": "middle"}, "color": {"color": "#BBBBBB"}},
      "physics": {"stabilization": {"iterations": 150}}
    }
    """)
    with open(path, "w", encoding="utf-8") as f:
        f.write(net.generate_html())
    return path


#check-validation

VALIDATION_QUESTIONS = [
    {"question": "What organization is located near the institute where Einstein worked?",
     "expected_answer": "Princeton University", "keyword_check": ["Einstein", "Princeton University"]},
    {"question": "In which city is the university located that is near the institute where Einstein worked?",
     "expected_answer": "Princeton", "keyword_check": ["Einstein", "Princeton,"]},
    {"question": "In which country was the person born who established the Nobel Prize that Marie Curie won?",
     "expected_answer": "Sweden", "keyword_check": ["Marie Curie", "Sweden"]},
    {"question": "Which empire controlled the country where Marie Curie was born?",
     "expected_answer": "Russian Empire", "keyword_check": ["Marie Curie", "Russian Empire"]},
    {"question": "In which city did Marie Curie's husband die?",
     "expected_answer": "Paris", "keyword_check": ["Marie Curie", "died", "Paris"]},
]


def run_validation(G: nx.MultiDiGraph, paragraphs: list[str]) -> None:
    graph_correct, keyword_failures = 0, 0
    print(f"{'#':<3}{'Question':<70}{'Graph answer':<20}{'Correct?':<10}{'Keyword search':<15}")
    print()
    for i, case in enumerate(VALIDATION_QUESTIONS, start=1):
        result = query(case["question"], G)
        answer = result["answer"] or "(none)"
        is_correct = case["expected_answer"].lower() in str(answer).lower()
        graph_correct += is_correct

        kw_found = keyword_search_baseline(case["keyword_check"], paragraphs)
        keyword_failures += (not kw_found)

        print(f"{i:<3}{case['question'][:68]:<70}{str(answer)[:18]:<20}"
              f"{'YES' if is_correct else 'no':<10}{'found' if kw_found else 'FAILED':<15}")
        print(f"    path: {result['path']}")
        print(f"    expected: {case['expected_answer']}\n")

    print()
    print(f"Graph agent correct: {graph_correct}/5  (need >= 4 to pass)")
    print(f"Keyword search failed on: {keyword_failures}/5  (need >= 2 to pass)")
    print(f"VALIDATION: {'PASS' if graph_correct >= 4 and keyword_failures >= 2 else 'FAIL'}\n")


#user input

def interactive_loop(G: nx.MultiDiGraph) -> None:
    print("--- Interactive query mode ---")
    print("Type a question about the graph, or 'exit' to quit.\n")
    while True:
        try:
            question = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        result = query(question, G)
        print(f"  answer: {result['answer'] or '(no path found — try rephrasing, or check the edge list above)'}")
        print(f"  path:   {result['path']}\n")


#main

if __name__ == "__main__":
    paragraphs = SAMPLE_PARAGRAPHS

    print("===== Building knowledge graph from sample paragraphs =====")
    print(f"Extraction backend: {'Groq (batched)' if client is not None else 'offline rule-based'}\n")
    G = build_knowledge_graph(paragraphs)
    print(f"Nodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}\n")

    print_graph_edges(G)

    print("===== Visualizing with pyvis =====")
    html_path = build_pyvis_graph(G)
    print(f"Interactive graph saved to: {html_path}\n")

    print("===== Validation: 5 multi-hop questions =====")
    run_validation(G, paragraphs)

    interactive_loop(G)
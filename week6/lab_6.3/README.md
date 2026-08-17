# Lab 6.3 — GraphRAG: Vector + Graph Hybrid Retrieval

A working GraphRAG pipeline that runs **vector retrieval** (ChromaDB, top-5) and
**graph traversal** (NetworkX, neighbourhood expansion) in parallel, merges and
deduplicates the results, and feeds them to an LLM for answer generation. It
also includes an automatic **query router** that picks vector-only,
graph-only, or hybrid retrieval based on the question type.


## Results

| Category | Vector-only | Graph-only | Hybrid | Expected Winner |
|---|---|---|---|---|
| factual | 100% | 100% | 100% | vector |
| relational | 100% | 100% | 100% | graph |
| complex | 20% | 60% | **100%** | hybrid |

**Router accuracy: 15/15** (needs ≥ 12/15 — passes)

Full per-question breakdown: [`results.md`](results.md) (regenerate any time
with `python3 evaluate.py`).

### Why factual/relational show 100% across the board

The knowledge graph in this lab is small and fully populated from the same
facts as the documents, so for a simple *1-hop* question, graph traversal can
almost always find the right fact too — it's not just vector's territory in
this dataset. The real difference between the three modes at that level
isn't correctness, it's **how much noise comes with the answer**:

| Category | Vector-only | Graph-only | Hybrid |
|---|---|---|---|
| factual | 5.0 items | 9.2 items | 16.2 items |
| relational | 5.0 items | 11.8 items | 21.0 items |
| complex | 5.0 items | 11.4 items | 23.0 items |

Vector-only always returns a tight, fixed top-5. Graph-only pulls in
everything within its hop radius — noticeably more context for the same
answer. This is the actual argument for the **router**: sending simple
questions to vector-only isn't just about correctness, it avoids paying for
(and having the LLM sift through) unnecessary extra context.

Where the three modes genuinely diverge on *correctness* is the **complex**
category — questions that need a relationship AND a specific fact chained
together (2–3 hops). There, vector-only and graph-only alone measurably fail
(20% / 60%), while hybrid — which uses a deeper graph traversal *and* vector
search together — gets all 5.

## Architecture

```
questions.py       15 test questions (5 factual, 5 relational, 5 complex),
                    each with an expected route + strict grading criteria

domain_data.py      The knowledge base: the same facts represented twice —
                     as plain-text chunks (DOCUMENTS) and as graph edges
                     (GRAPH_EDGES) — describing a small fictional tech
                     ecosystem (companies, founders, products, cities)

vector_store.py      ChromaDB wrapper. Embeds DOCUMENTS with a TF-IDF vector
                      function (offline, no model download) and returns
                      top-k matches for a query.

graph_store.py        NetworkX wrapper. Builds a directed graph from
                       GRAPH_EDGES, finds entities named in a question, and
                       expands outward (breadth-first) to collect connected
                       facts as plain-English sentences.

generator.py            Turns (question, context) into a final answer.
                         Uses the real Claude API if ANTHROPIC_API_KEY is
                         set, otherwise falls back to a deterministic
                         extractive method (see note below).

router.py                 Rule-based classifier: decides vector / graph /
                           hybrid for a given question (see docstring for
                           why rules instead of a trained model here).

pipeline.py                  Wires it all together: run_vector_only(),
                              run_graph_only(), run_hybrid() (parallel
                              vector + graph via ThreadPoolExecutor, then
                              merge_and_deduplicate()).

evaluate.py                    Runs all 15 questions through all 3 modes +
                                the router, scores everything, prints the
                                comparison table, checks the two pass/fail
                                validation criteria, and writes results.md.
```

Run order: `domain_data → {vector_store, graph_store} → generator → router
→ pipeline → evaluate`

## How to run

```bash
pip install chromadb networkx scikit-learn
python3 evaluate.py
```


## How scoring works

Each question has `required_fact_groups`: sets of keywords that must
**co-occur in a single retrieved sentence** for that "hop" of the answer to
count as found. A question is only marked correct if *every* group is
found. This is stricter (and more honest) than checking "is this keyword
anywhere in the whole pile of retrieved text" — that naive check gives false
positives, since the same entity name often appears in several unrelated
sentences in the corpus. See the docstring at the top of `questions.py` for
the full reasoning.

Complex questions require 2–3 groups (one per hop in the reasoning chain),
which is what makes the complex category a genuine test of whether a
retrieval mode can assemble a multi-hop answer, not just recall one fact.

## Design notes worth knowing about

- **Graph hop depth differs by mode on purpose.** `graph-only` traverses 2
  hops (a sensible, noise-controlled default for a standalone graph
  system). `hybrid` traverses 3 hops, because the complex questions in this
  lab genuinely need 3 hops to reach the final fact, and a hybrid system can
  afford deeper traversal since it has a second signal (vector) and a
  dedup/merge step to keep the extra context manageable. This is explained
  in more detail in `pipeline.py`.
- **The router is rule-based, not ML-trained.** With only 15 labeled
  questions there isn't enough data to train a real classifier — three
  keyword-based checks in a fixed priority order (nested-clause phrasing →
  hybrid, relation phrasing → graph, else → vector) is more transparent and
  easier to debug. See `router.py`'s docstring for how to swap in a trained
  model later.
- **The domain is entirely fictional** (made-up companies/people/products)
  to keep the lab self-contained and copyright-safe, while still being
  richly interconnected enough to support genuine multi-hop questions.




`generator.py` will automatically use the real groq API instead of the
extractive fallback (see `_groq_generate()` in that file). Note: the
extractive fallback can only pick sentences that already share words with
the question, so it under-performs on true multi-hop questions where the
final fact shares no words with the original question — a real LLM handles
this correctly, since it can actually reason across the merged context
instead of just pattern-matching. That's why this lab's automated grading
scores **retrieval quality** (is the right fact present in context) rather
than final-answer text — that metric doesn't depend on which generator is
plugged in.

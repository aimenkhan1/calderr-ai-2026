# Memory Inspector — One Agent, Four Memory Types

A `MemoryAgent` that integrates four distinct memory types, plus a Streamlit
dashboard that shows all four side by side.

## The four memory types

| Type | File | Answers | Behavior |
|---|---|---|---|
| **Episodic** | `episodic.py` | "What happened, and when?" | Raw timestamped log. Importance decays over time; old low-importance entries get forgotten; when the log grows past a threshold, the oldest entries get compressed into a summary block instead of deleted outright. |
| **Semantic** | `semantic.py` | "What stable facts do I know?" | Key → value profile (name, role, preferences...). New statements of the same fact *overwrite* the old value — a profile, not a decaying log. |
| **Procedural** | `procedural.py` | "How should I behave?" | Behavioral corrections learned from feedback (e.g. "always include type hints"). Tracks how often each rule actually gets applied. |
| **Graph** | `knowledge_graph.py` | "How do things relate?" | NetworkX directed graph of entities connected by typed relationships (e.g. `Alex --works_on--> Project Falcon`). |

## What makes it "one agent," not four separate demos

`agent.py`'s `MemoryAgent` class ties them together two ways:

1. **`observe(text)`** — a single entry point. Every observation is always
   logged to episodic memory (the raw log of everything that happens). On
   top of that, simple rule-based pattern matching checks whether the text
   *also* looks like a fact, a correction, or a relationship — and if so,
   distills it into the matching store(s) too. One sentence can land in
   multiple stores at once.
2. **`get_context_for_query(query)`** — pulls relevant material from *all
   four* stores for a given query in one call, so a real response-generation
   step has everything it needs (recent related events, known facts,
   applicable behavior rules, and connected entities) without querying each
   store separately.

The extraction logic in `observe()` is regex-based on purpose (zero external
dependencies, fully offline/deterministic). See the note at the bottom of
`agent.py` for how to swap in LLM-based extraction without touching anything
else in the agent.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens pre-loaded with a scripted demo scenario (`seed_data.py`)
— a developer working with an AI coding assistant across a few sessions —
so all four panels have meaningful content immediately.

### Using the UI

- **Four panels, side by side**: episodic log (top-left), semantic profile
  (top-right), knowledge graph (bottom-left), procedural corrections
  (bottom-right).
- **Sidebar "Observe" box**: type any sentence and watch which panel(s)
  light up. Try:
  - `"My name is Jordan."` → semantic
  - `"Jordan works on Project Comet."` → graph
  - `"Always write commit messages in present tense."` → procedural
  - `"Fixed the login bug today."` → episodic only (nothing else to extract)
- **"Decay pass" button**: manually runs importance decay + forgetting on
  the episodic log.
- **"Reset demo" button**: reloads the original scripted scenario.
- **Unified Context Preview** (bottom of page): type a query and see what
  the agent would pull from all four stores to answer it — this is
  `get_context_for_query()` made visible.

## File overview

```
episodic.py         Episode/MemoryBlock dataclasses + EpisodicMemory store
semantic.py          SemanticFact dataclass + SemanticMemory (profile) store
procedural.py         Correction dataclass + ProceduralMemory store
knowledge_graph.py     KnowledgeGraphMemory (NetworkX wrapper + DOT export)
agent.py                MemoryAgent: observe() + get_context_for_query()
seed_data.py              Scripted demo scenario for a populated first launch
app.py                      Streamlit dashboard (the four side-by-side panels)
```

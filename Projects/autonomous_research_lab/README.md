# 🔬 Autonomous AI Research Lab

**Project 5-P-A — Production** · Week 5 Multi-Agent Systems

A fully autonomous research system: give it a question, and a
**dynamically-assembled team of 3–5 specialist agents** — chosen at
runtime based on the question's domain — researches it end-to-end
(hypothesis → parallel evidence gathering → critique → synthesis → peer
review) and publishes a structured, cited research report, with zero
human intervention after you hit run.

---

## 📸 What It Does

Give it a research question, and the system will:
- Classify the question into a research domain
- **Dynamically assemble** 3–5 specialist agent personas tailored to *this specific question* — not a fixed template team
- Generate a falsifiable hypothesis before any evidence is gathered
- Dispatch all specialists **in true parallel** using LangGraph's `Send` API — the exact number of parallel agents is decided at runtime, not hardcoded
- Have each specialist retrieve real evidence via **RAG** (TF-IDF over a seeded document store) and optionally call a **tool** to fetch a full source document
- Run a **Critic Agent** that reviews every finding and can directly rewrite a finding's confidence score, with a documented reason, when it finds a weak link
- Synthesize a cited report via a **Synthesis Agent**
- Run an independent **Peer Review Agent** — blind to the raw findings, checking only the finished report — for internal contradictions and unsupported claims
- Publish the final structured report via a **FastAPI** endpoint and a live-progress **Streamlit** UI
- Save every report as Markdown

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Backend | Groq (llama-3.3-70b-versatile) |
| Orchestration | LangGraph (dynamic fan-out via `Send`, conditional edges) |
| RAG | scikit-learn TF-IDF retrieval over a seeded local document store (no external embedding API — fully offline after install) |
| Tool Calling | Groq function-calling (`fetch_full_document` tool) |
| Typed Messages | Pydantic |
| API | FastAPI (auto-generated OpenAPI docs) |
| UI | Streamlit (phase-by-phase live progress via `app.stream()`) |
| Deployment | Docker Compose (API + UI as separate services) |
| Environment | python-dotenv |
| Language | Python 3.11+ |

---

## 📂 Project Structure
autonomous_research_lab/

│

├── models.py                    # Typed Pydantic schemas for every phase

├── llm_client.py                 # Groq wrapper — structured JSON + tool-calling

├── graph.py                      # LangGraph orchestration (dynamic Send fan-out)

├── api.py                        # FastAPI REST API

├── streamlit_app.py              # Streamlit UI with live phase progress

├── main.py                       # CLI entry point

├── requirements.txt

├── .env.example

├── docker-compose.yml            # One command starts API + UI

├── BLOG_POST_DRAFT.md            # Ready-to-publish blog post (edit + post to dev.to/Hashnode)

├── architecture_diagram.mmd      # Mermaid diagram source (auto-generated from the live graph)

├── README.md                     # This file

├── docker/

│   ├── Dockerfile.api

│   └── Dockerfile.streamlit

├── agents/

│   ├── domain_classifier.py      # Classifies question into a research domain

│   ├── agent_assembler.py        # Dynamically designs 3–5 specialist personas

│   ├── hypothesis_generator.py   # Phase 1: proposes falsifiable hypothesis

│   ├── evidence_agent.py         # Phase 2: RAG + tool-calling, instantiated dynamically

│   ├── critic_agent.py           # Phase 3: challenges findings, can rewrite confidence

│   ├── synthesis_agent.py        # Phase 4: writes the cited report

│   ├── peer_review_agent.py      # Phase 5: independent second-pass check

│   └── report_publisher.py       # Assembles final report + Markdown rendering

├── rag/

│   └── document_store.py         # TF-IDF retrieval + tool-callable full-document fetch

├── data/seed_corpus/              # Seeded source documents, 5 domains × 3 docs each

│   ├── ai_safety/

│   ├── biotechnology/

│   ├── climate_tech/

│   ├── fintech/

│   └── quantum_computing/

└── sample_reports/                # 5 sample reports, one per domain (see note below)

---

## 🏗 Architecture

```
Research Question Input
        │
        ▼
Domain Classifier ─── constrained to domains the seed corpus covers
        │
        ▼
Dynamic Agent Assembler ─── designs 3-5 specialist personas FOR THIS QUESTION
        │
        ▼
Hypothesis Generator ─── falsifiable hypothesis, before evidence is seen
        │
        ▼
   ┌────┴─────────────────────────────┐
   │   DYNAMIC PARALLEL FAN-OUT        │   ← LangGraph Send() — count decided
   │   (via LangGraph Send API)        │      at runtime by the Assembler
   ▼        ▼        ▼        ▼        ▼
Evidence  Evidence  Evidence  Evidence  Evidence
Agent 1   Agent 2   Agent 3   Agent 4   Agent 5     (3-5, not fixed)
   │        │        │        │        │
   │   each: RAG retrieval (TF-IDF) + optional tool call
   │        │        │        │        │
   └────┬───┴────┬───┴────┬───┴────┬───┘
        ▼ (fan-in — all must complete)
Critic Agent ─── can directly rewrite a finding's confidence, with a reason
        │
        ▼
Synthesis Agent ─── writes cited report
        │
        ▼
Peer Review Agent ─── independent second pass, blind to raw findings
        │
        ▼
Report Publisher ─── final structured ResearchReport
        │
        ├──► FastAPI REST endpoint (JSON + Markdown)
        └──► Streamlit UI (live phase-by-phase progress)
```

A live-generated Mermaid version of this exact graph is in
`architecture_diagram.mmd` — regenerate anytime with `python graph.py`
(also writes `graph.png` if you have internet access).

---
## Complete Flow of one user input example


USER
  │
  │ enters question
  ▼
streamlit_app.py
  │
  │ calls run_research(question)
  ▼
graph.py
  │
  ├── 1. classify_domain
  │       │
  │       ▼
  │   domain_classifier.py
  │       │
  │       └── structured_completion()
  │               │
  │               ▼
  │           llm_client.py
  │               │
  │               ▼
  │           Groq LLM
  │               │
  │               ▼
  │           JSON response
  │               │
  │               ▼
  │           Pydantic validation
  │               │
  │               ▼
  │           DomainClassification
  │
  ├── 2. assemble_team
  │       │
  │       ▼
  │   agent_assembler.py
  │       │
  │       └── structured_completion()
  │               │
  │               ▼
  │           Groq LLM
  │               │
  │               ▼
  │           AssemblyPlan
  │
  ├── 3. generate_hypothesis
  │       │
  │       ▼
  │   hypothesis_generator.py
  │       │
  │       └── structured_completion()
  │               │
  │               ▼
  │           Groq LLM
  │               │
  │               ▼
  │           Hypothesis
  │
  ├── 4. DYNAMIC FAN-OUT
  │
  │       ┌───────────────┐
  │       │               │
  │       ▼               ▼
  │   Evidence 1      Evidence 2
  │       │               │
  │       ▼               ▼
  │   Evidence 3      Evidence 4
  │       │               │
  │       └───────┬───────┘
  │               │
  │               ▼
  │          All findings
  │
  ├── 5. critique
  │       │
  │       ▼
  │   critic_agent.py
  │       │
  │       └── LLM
  │
  ├── 6. synthesize
  │       │
  │       ▼
  │   synthesis_agent.py
  │       │
  │       └── LLM
  │
  ├── 7. peer_review
  │       │
  │       ▼
  │   peer_review_agent.py
  │       │
  │       └── LLM
  │
  └── 8. publish
          │
          ▼
      report_publisher.py
          │
          ▼
      ResearchReport
          │
          ├── Streamlit UI
          └── API / JSON / Markdown


## 📰 Report Sections

| Section | Source |
|---------|--------|
| Hypothesis | Hypothesis Generator (before evidence is seen) |
| Dynamically Assembled Team | Agent Assembler |
| Evidence Findings | Evidence Agents (RAG + tool calls) |
| Critic Report | Critic Agent (confidence rewrites, hypothesis alignment) |
| Synthesized Sections | Synthesis Agent (cited to finding IDs) |
| Peer Review | Peer Review Agent (contradictions, unsupported claims) |

---

## 🚀 Setup & Run

### Option A — Docker Compose (recommended, one command)

```bash
cp .env.example .env
# edit .env and add your GROQ_API_KEY

docker compose up --build
```

- FastAPI: http://localhost:8000/docs
- Streamlit: http://localhost:8501

### Option B — Local Python

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your GROQ_API_KEY

# CLI
python main.py "Will fault-tolerant quantum computing be capable of breaking RSA encryption before 2035?"

# FastAPI
uvicorn api:app --reload

# Streamlit
streamlit run streamlit_app.py
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

---

## ⌨️ API Usage

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "Will direct air capture reach cost parity with nature-based carbon removal by 2030?"}'
```

Full interactive docs (OpenAPI/Swagger) at `/docs` once running.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/domains` | GET | Lists supported research domains |
| `/research` | POST | Runs full pipeline, returns structured JSON report |
| `/research/markdown` | POST | Same pipeline, returns rendered Markdown |

---

## 📊 Observability (LangSmith)

To export traces for every run:

```bash
# in .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=autonomous-research-lab
```

Once set, every LangGraph node execution (including each dynamically
spawned evidence agent) is automatically traced — no code changes needed,
since LangGraph reads these environment variables natively.

---

## 📝 Sample Reports — Important Note

`sample_reports/` contains **one generated report per domain** (5 total),
produced by running the actual pipeline end-to-end with **mocked LLM
responses** rather than a live Groq API key (this environment couldn't
reach the Groq API while building this). Every report is explicitly
labeled with a notice at the top.

**These prove the pipeline architecture works correctly across all 5
domains** — dynamic assembly, parallel fan-out, critic revision, peer
review all fire correctly in every run — but the *content* is
illustrative, not genuine research output.

**Before submitting this as a portfolio piece**, regenerate all 5 with a
real `GROQ_API_KEY`:

```bash
python main.py "Can scalable oversight techniques keep pace with frontier model capability growth?"
python main.py "Will in vivo CRISPR delivery overcome its current tissue-targeting limitations within the next decade?"
python main.py "Can direct air capture reach cost parity with nature-based carbon removal by 2030?"
python main.py "Will embedded finance partnerships face structurally tighter regulation following recent BaaS enforcement actions?"
python main.py "Will fault-tolerant quantum computing be capable of breaking RSA encryption before 2035?"
```

Each run saves `research_report.md` — rename and move each into
`sample_reports/` to replace the mocked versions.



---

## ✅ Verified Behavior (automated tests during build)

Since this environment has no network access to the Groq API, every phase
was verified with mocked LLM responses substituted at the exact call
boundary, so the real orchestration, data flow, and error handling are
exercised authentically:

- ✅ Dynamic assembly correctly produces a runtime-determined number of specialists (tested with 3)
- ✅ `Send`-based fan-out spawns exactly that many parallel `evidence_agent` calls — not hardcoded
- ✅ Critic Agent genuinely rewrites a finding's confidence (tested: 0.75 → 0.55) with a recorded reason
- ✅ One evidence agent failing does **not** crash the run — the pipeline degrades gracefully and still publishes a report from the survivors (tested: 2 of 3 succeeding)
- ✅ RAG retrieval correctly surfaces the most relevant seed document for a given sub-question (tested across 2 domains)
- ✅ FastAPI routes register and validate correctly (`/health`, `/domains`, `/research`, `/research/markdown`)
- ✅ Full pipeline runs successfully across all 5 supported domains (see `sample_reports/`)


---

## 👩‍💻 Built By

Aiman Nadeem Khan

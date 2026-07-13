#  Hybrid Search Engine

A CLI search tool that combines BM25 (keyword) and semantic (embedding) search over a news corpus, with cross-encoder re-ranking, and a rigorous comparison study across all four retrieval methods.

---

## What It Does

Ask any query in plain English and the engine will:

-  **Load** — reads a news dataset (CSV, TXT, or JSON)
-  **Split** — chunks articles into token-sized passages
-  **Index (BM25)** — builds a keyword-based index with `rank_bm25`
-  **Index (Semantic)** — embeds chunks into ChromaDB with sentence-transformers
-  **Fuse** — combines both indexes via LangChain's `EnsembleRetriever`
-  **Re-rank** — narrows results with a `BAAI/bge-reranker-base` cross-encoder
-  **Evaluate** — runs 30 test queries across BM25 / Semantic / Hybrid / Hybrid+Rerank and reports accuracy

---

## Tech Stack

| Layer | Technology |
|---|---|
| Keyword Retrieval | `rank_bm25` via `BM25Retriever` |
| Semantic Retrieval | `sentence-transformers` (`all-MiniLM-L6-v2`) + ChromaDB |
| Retrieval Fusion | LangChain `EnsembleRetriever` |
| Re-ranking | `BAAI/bge-reranker-base` (`CrossEncoder`) |
| Document Loading | `CSVLoader` / `TextLoader` / custom JSON loader |
| Chunking | `RecursiveCharacterTextSplitter` (token-based) |
| Environment | `python-dotenv` |

---

## Project Structure

```
hybrid_search_engine/
│
├── main.py
│   └── Single-file CLI tool containing all sections below.
│
│       sec1 — Embeddings
│         RealEmbeddings         - sentence-transformers, genuine semantic search
│         HashingTrickEmbeddings - dependency-free fallback (keyword-overlap only)
│         get_embeddings()       - tries real embeddings, falls back automatically
│
│       sec2 — Load & Split
│         load_and_split()       - loads .csv / .txt / .json news datasets,
│                                   splits into token-sized chunks
│
│       sec3 — Hybrid Retriever
│         build_hybrid_retriever() - builds BM25 index + Chroma vector index,
│                                     fuses them via EnsembleRetriever
│
│       sec4 — Re-ranking
│         CrossEncoderReranker    - real cross-encoder (BAAI/bge-reranker-base)
│         LexicalOverlapReranker  - dependency-free fallback
│         get_reranker()          - tries real reranker, falls back automatically
│
│       sec5 — Search
│         retrieve_with_reranking() - hybrid retrieve, then rerank to top-N
│
│       sec6 — Evaluation
│         hit_at_k()            - checks if expected keyword appears in results
│         run_comparison()      - runs all 4 methods across 30 queries,
│                                  saves comparison_results.json
│
├── news_corpus.csv / .txt / .json
│   └── Your news dataset (any of the 3 supported formats)
│
├── qa_pairs.json
│   └── 30 test queries, each with an expected_keyword for grading
│
├── comparison_results.json
│   └── Generated after each run — hits/accuracy per method
│
└── README.md
    └── This file.
```

---

## Architecture

```
News Dataset (CSV/TXT/JSON)
    ↓
load_and_split()         reads + chunks articles → Documents
    ↓
get_embeddings()         real semantic embeddings (or fallback)
    ↓
build_hybrid_retriever() BM25 Index + Vector Index → EnsembleRetriever
    ↓
get_reranker()           real cross-encoder (or fallback)
    ↓
Query Input               plain English search query
    ↓
retrieve_with_reranking() hybrid retrieve → rerank → top-N results
    ↓
Results Display           title + snippet per result, printed to CLI
    ↓
run_comparison()          BM25 vs Semantic vs Hybrid vs Hybrid+Rerank
                           across 30 queries → comparison_results.json
```

---

## Setup and Run

### 1. Install dependencies
```bash
pip install langchain langchain-community langchain-classic langchain-chroma
pip install rank_bm25 sentence-transformers python-dotenv pypdf tiktoken
```

### 2. Add your news dataset
Any of these formats work:
- **CSV** — one column should contain article text
- **TXT** — plain text file
- **JSON** — list of objects with a `"text"` or `"content"` field

### 3. Prepare `qa_pairs.json`
30 queries, each shaped like:
```json
{"question": "What happened in the election?", "expected_keyword": "election"}
```

### 4. Run the tool
```bash
python main.py
```
You'll be prompted for:
- Path to your news dataset
- Path to your `qa_pairs.json`
- A search query to demo live

The script then runs the full 30-query comparison automatically and saves results.

---

## Built By
Aiman Nadeem Khan
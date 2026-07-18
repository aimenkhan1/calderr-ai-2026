# 📊 RAG Bench — RAG Evaluation Benchmark

A comprehensive, reusable evaluation framework that tests multiple RAG
configurations (chunk sizes, embedding models, retrieval strategies) in
parallel, runs real statistical significance testing between them, and
publishes the results as a live HTML report via GitHub Pages.

---

##  What It Does

Point it at a document and a set of test questions, and RAG Bench will:

-  **Configure** — Read any number of RAG configurations from a plain YAML file
-  **Build** — Construct a full pipeline (chunk → embed → index) for each one
-  **Run in Parallel** — Evaluate all configurations concurrently, one process per config
-  **Score** — Measure retrieval accuracy (`hit_at_k`) plus RAGAS's faithfulness,
  answer relevancy, and context precision
-  **Analyze** — Run a paired t-test between the best config and every other,
  so you know which differences are *real* and which are noise
-  **Publish** — Generate a static HTML report with charts and tables, auto-deployed
  to GitHub Pages on every push

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Config Format | YAML (`pyyaml`) |
| Retrieval | LangChain + ChromaDB + `sentence-transformers` |
| Parallel Execution | `concurrent.futures.ProcessPoolExecutor` |
| Evaluation Metrics | RAGAS (faithfulness, answer relevancy, context precision) + custom `hit_at_k` |
| Statistical Testing | `scipy.stats` (paired t-test, 95% confidence intervals) |
| Report Rendering | Static HTML + Chart.js (no server required) |
| CI/CD | GitHub Actions → GitHub Pages |
| LLM Backend | Groq (`llama-3.1-8b-instant`) |
| Language | Python 3.11+ |

---

## 📂 Project Structure

```
rag_eval_benchmark/
│
├── config.yaml                     # Define what to test - no code changes needed
├── qa_pairs.json                   # Test questions with expected_keyword / ground_truth
├── run_benchmark.py                # Main entry point - orchestrates the full pipeline
├── requirements.txt
│
├── benchmark/
│   ├── pipeline_builder.py         # Turns one config into a working RAG retriever
│   ├── metrics.py                  # hit_at_k (custom) + RAGAS scoring
│   ├── evaluation_runner.py        # Parallel execution across all configs
│   ├── statistical_analyzer.py     # Paired t-test, confidence intervals, leaderboard
│   └── report_generator.py         # Renders the HTML report
│
├── .github/workflows/
│   └── benchmark.yml               # CI: runs benchmark + deploys to GitHub Pages
│
├── docs/
│   └── index.html                  # Generated report (published via GitHub Pages)
│
├── benchmark_results.json          # Generated - raw statistical analysis output
└── README.md                       # This file
```

---

## 🏗 Architecture

```
config.yaml
      ↓
Pipeline Builder        → builds one full RAG pipeline per configuration
      ↓
Parallel Evaluation Runner → runs all configs concurrently (ProcessPoolExecutor)
      ↓
RAGAS + Custom Metrics   → scores each config: faithfulness, relevancy,
                           context precision, hit_at_k
      ↓
Statistical Analyzer     → mean, 95% CI, and paired t-test vs the best config
                           (is the winner's lead real, or noise?)
      ↓
HTML Report Generator    → static webpage: bar chart, leaderboard, findings
      ↓
GitHub Pages             → published, shareable URL via GitHub Actions CI
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your API key (optional — enables RAGAS metrics)
```
GROQ_API_KEY=your_groq_api_key_here
```
Without it, the benchmark still runs fully using the `hit_at_k` custom metric —
RAGAS scoring is skipped gracefully.

### 3. Edit `config.yaml`
Add, remove, or change configurations — no Python code changes needed:
```yaml
corpus: "your_document.pdf"
qa_pairs: "qa_pairs.json"
max_workers: 3
configurations:
  - name: "my_config"
    chunk_size: 512
    k: 5
    embedding_model: "all-MiniLM-L6-v2"
```

### 4. Run the benchmark
```bash
python run_benchmark.py
```

Outputs:
- `benchmark_results.json` — full statistical analysis
- `docs/index.html` — the publishable report

### 5. Publish to GitHub Pages
Push to `main` — the included GitHub Actions workflow runs the benchmark
and deploys `docs/index.html` automatically. Enable Pages in your repo
settings (Source: **GitHub Actions**) once, and every push republishes it.

---

## ⌨️ Features

| Feature | Description |
|---------|-------------|
| 🔁 Reusable | Test a totally new document or config set by editing YAML only |
| ⚡ Real Parallelism | Each configuration runs in its own process, not just structured sequentially |
| 📏 Dual Metrics | Fast custom `hit_at_k` always runs; RAGAS runs if an LLM key is present |
| 📊 Real Statistics | Paired t-test + 95% confidence intervals, not just "which mean is bigger" |
| 🌐 Publishable | One command produces a shareable HTML report, no server needed |
| 🔄 CI/CD Built In | GitHub Actions workflow re-runs and republishes on every push |
| 🛡️ Graceful Fallbacks | Missing API key or no internet? Still runs, just skips what it can't do |

---

## 📊 Sample Report Output

```
Best configuration: large_chunks_k10 (95% accuracy)

Leaderboard:
  large_chunks_k10    95%  CI=[85%, 100%]
  medium_chunks_k10   90%  CI=[77%, 100%]
  large_chunks_k3     80%  CI=[62%, 98%]
  small_chunks_k10    80%  CI=[62%, 98%]
  medium_chunks_k3    75%  CI=[56%, 94%]
  small_chunks_k3     60%  CI=[38%, 82%]

Significance vs best (paired t-test):
  medium_chunks_k10  p=0.317  not significant - could be noise
  large_chunks_k3    p=0.083  not significant - could be noise
  small_chunks_k3    p=0.012  SIGNIFICANT - real difference
```

---

## 📝 Notes

- The paired t-test is used (not an independent-samples test) because every
  configuration is evaluated on the *same* questions in the *same* order —
  that pairing is real information a plain average discards
- Each configuration runs in its own OS process, avoiding shared-state
  issues between separate Chroma collections and embedding model instances
- Without a `GROQ_API_KEY`, RAGAS metrics are skipped automatically and the
  benchmark still completes using `hit_at_k` alone — useful for fast CI runs
- The framework is corpus-agnostic — swap `corpus:` in `config.yaml` to
  benchmark against any PDF without touching code

---

## 👩‍💻 Built By

Aiman Nadeem khan
"""
Parallel Evaluation Runner — builds and scores every configuration from
config.yaml concurrently, using one process per configuration.

Each config gets its own process because each builds its own Chroma
collection and (potentially) its own embedding model in memory - keeping
them in separate processes avoids state collisions and lets configs
actually run at the same time on multi-core machines.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from benchmark.pipeline_builder import build_pipeline, get_embeddings
from benchmark.metrics import score_pipeline_custom, score_pipeline_ragas
from langchain_groq import ChatGroq


#llm model for scoring func
def _get_llm():

    try:
        if not os.getenv("GROQ_API_KEY"):
            return None
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    except Exception:
        return None


#this function runs a single configuration, builds the pipeline, and scores it
def run_single_config(config, corpus_path, qa_pairs):

    start = time.time()
    pipeline = build_pipeline(config, corpus_path)

    custom_scores = score_pipeline_custom(pipeline, qa_pairs)

    llm = _get_llm()
    embeddings = get_embeddings(config.get("embedding_model", "all-MiniLM-L6-v2"))
    ragas_scores = score_pipeline_ragas(pipeline, qa_pairs, llm=llm, embeddings=embeddings)

    elapsed = time.time() - start
    return {
        "name": config["name"],
        "config": config,
        "num_chunks": pipeline["num_chunks"],
        "hit_at_k_scores": custom_scores,
        "ragas_scores": ragas_scores,
        "elapsed_seconds": round(elapsed, 1),
    }


#this function runs all configurations in parallel, collects results, and prints progress
def run_benchmark(configurations, corpus_path, qa_pairs, max_workers=3):
    print(f"Running {len(configurations)} configurations across up to "f"{max_workers} parallel workers...\n")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_config, cfg, corpus_path, qa_pairs): cfg["name"]
            for cfg in configurations
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                accuracy = sum(result["hit_at_k_scores"]) / len(result["hit_at_k_scores"])
                print(f"  [done] {name:<20} hit_at_k={accuracy:.0%}  "
                      f"({result['elapsed_seconds']}s, {result['num_chunks']} chunks)")
                results.append(result)
            except Exception as e:
                print(f"  [FAILED] {name}: {e}")

    return results
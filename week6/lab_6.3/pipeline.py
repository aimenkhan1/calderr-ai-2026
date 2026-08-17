"""
pipeline.py

Ties vector_store.py, graph_store.py and generator.py together into three
retrieval modes that can each be run end-to-end on a question:

  - run_vector_only(question)  -> ChromaDB top-5 only
  - run_graph_only(question)   -> NetworkX neighbourhood expansion only
  - run_hybrid(question)       -> BOTH, run concurrently, then merged +
                                   deduplicated before generation


"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

from domain_data import DOCUMENTS, GRAPH_EDGES
from vector_store import VectorRetriever
from graph_store import GraphRetriever
from generator import generate

GRAPH_ONLY_HOPS = 2
HYBRID_GRAPH_HOPS = 3
VECTOR_TOP_K = 5




vector_retriever = VectorRetriever(DOCUMENTS)
graph_retriever = GraphRetriever(GRAPH_EDGES)



def merge_and_deduplicate(*context_lists: List[dict]) -> List[dict]:

    best_by_text: dict = {}
    for context_list in context_lists:
        for item in context_list:
            key = " ".join(item["text"].lower().split())  # normalize whitespace/case
            if key not in best_by_text or item["score"] > best_by_text[key]["score"]:
                best_by_text[key] = item

    merged = list(best_by_text.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged




def run_vector_only(question: str) -> dict:
    context = vector_retriever.retrieve(question, top_k=VECTOR_TOP_K)
    answer = generate(question, context)
    return {"mode": "vector", "context": context, "answer": answer}


def run_graph_only(question: str) -> dict:
    context = graph_retriever.retrieve(question, max_hops=GRAPH_ONLY_HOPS)
    answer = generate(question, context)
    return {"mode": "graph", "context": context, "answer": answer}


def run_hybrid(question: str) -> dict:
    # Run vector retrieval and graph traversal AT THE SAME TIME (two threads),
    # rather than one after another.
    with ThreadPoolExecutor(max_workers=2) as pool:
        vector_future = pool.submit(vector_retriever.retrieve, question, VECTOR_TOP_K)
        graph_future = pool.submit(graph_retriever.retrieve, question, HYBRID_GRAPH_HOPS)
        vector_context = vector_future.result()
        graph_context = graph_future.result()

    merged_context = merge_and_deduplicate(vector_context, graph_context)
    answer = generate(question, merged_context)
    return {
        "mode": "hybrid",
        "context": merged_context,
        "vector_context": vector_context,
        "graph_context": graph_context,
        "answer": answer,
    }


if __name__ == "__main__":
    from questions import QUESTIONS

    q = next(item for item in QUESTIONS if item["id"] == "C1")["question"]
    print("Question:", q)
    result = run_hybrid(q)
    print("\nVector found:", len(result["vector_context"]), "items")
    print("Graph found: ", len(result["graph_context"]), "items")
    print("Merged/deduped:", len(result["context"]), "items")
    print("\nAnswer:", result["answer"])

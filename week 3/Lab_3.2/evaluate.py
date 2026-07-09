"""
RAG Evaluation
Measures retrieval accuracy across:
    - chunk_size: 256 / 512 / 1024 tokens
    - k (retrieved chunks): 3 / 10

Metric: "hit@k" - for each of the 20 Q&A pairs, did the expected
keyword/answer appear in ANY of the top-k retrieved chunks? This is
a simple, honest proxy for retrieval accuracy that doesn't require
an LLM call - it tests whether the retriever surfaces the right
evidence at all, which is the necessary first step before generation
can possibly be correct.

Run:
    python evaluate_rag.py
"""
import json
import time

from rag_pipeline import (
    load_documents,
    split_documents,
    get_embeddings,
    build_vectorstore,
    get_retriever,
)

CHUNK_SIZES = [256, 512, 1024]
K_VALUES = [3, 10]


def load_qa_pairs(path="C:\\Users\\HP\\calderr-ai-2026\\week 3\\Lab_3.2\\Q\\A_pairs.json"):
    with open(path) as f:
        return json.load(f)


#this function checks if the expected keyword is present in any of the retrieved documents for each question, and counts the number of hits.
def hit_at_k(retriever, qa_pairs):
    hits = 0
    per_question = []
    for pair in qa_pairs:
        retrieved_docs = retriever.invoke(pair["question"])
        combined_text = " ".join(doc.page_content.lower() for doc in retrieved_docs)
        hit = pair["expected_keyword"].lower() in combined_text
        hits += int(hit)
        per_question.append({
            "question": pair["question"],
            "expected_keyword": pair["expected_keyword"],
            "hit": hit,
        })
    return hits, len(qa_pairs), per_question


# this function runs the evaluation across all combinations of chunk_size and k, measuring the number of hits and calculating accuracy for each configuration. It also records the time taken to build the vectorstore and to perform the queries.
def run_evaluation():
    qa_pairs = load_qa_pairs()
    print(f"Loaded {len(qa_pairs)} Q&A pairs for evaluation.\n")

    pdf_path = input("Enter the PDF path: ").strip().strip('"')
    docs = load_documents(pdf_path)
    embeddings = get_embeddings()

    results = []  # list of dicts: chunk_size, k, hits, total, accuracy, time_sec

    for chunk_size in CHUNK_SIZES:
        print(f"\nChunk size: {chunk_size} tokens\n")
        chunks = split_documents(docs, chunk_size=chunk_size)

        t0 = time.time()
        vectorstore = build_vectorstore(
            chunks, embeddings, collection_name=f"eval_chunk_{chunk_size}"
        )
        build_time = time.time() - t0

        for k in K_VALUES:
            retriever = get_retriever(vectorstore, k=k)
            t0 = time.time()
            hits, total, per_question = hit_at_k(retriever, qa_pairs)
            query_time = time.time() - t0
            accuracy = hits / total

            print(f"  k={k:<3} -> {hits}/{total} correct ({accuracy:.0%}) "f"[build={build_time:.2f}s, {total} queries={query_time:.2f}s]")

            results.append({
                "chunk_size": chunk_size,
                "k": k,
                "num_chunks": len(chunks),
                "hits": hits,
                "total": total,
                "accuracy": accuracy,
                "build_time_sec": round(build_time, 2),
                "query_time_sec": round(query_time, 2),
            })

    return results


def print_summary_table(results):
    print(f"\nSUMMARY\n")
    print(f"{'Chunk Size':<12}{'k':<6}{'# Chunks':<10}{'Hits':<8}{'Accuracy':<10}")
    print()
    for r in results:
        print(f"{r['chunk_size']:<12}{r['k']:<6}{r['num_chunks']:<10}"f"{r['hits']}/{r['total']:<6}{r['accuracy']:.0%}")

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\nBest configuration: chunk_size={best['chunk_size']}, k={best['k']} " f"-> {best['accuracy']:.0%} accuracy")


if __name__ == "__main__":
    results = run_evaluation()
    print_summary_table(results)

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full results to evaluation_results.json")
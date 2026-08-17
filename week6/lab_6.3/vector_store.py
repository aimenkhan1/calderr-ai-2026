"""
vector_store.py

Wraps ChromaDB to do the "vector retrieval" half of the pipeline.


"""

from __future__ import annotations

from typing import List

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddingFunction(EmbeddingFunction):

    def __init__(self, corpus_texts: List[str]):

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectorizer.fit(corpus_texts)

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self.vectorizer.transform(input)
        return vectors.toarray().tolist()


class VectorRetriever:

    def __init__(self, documents: List[dict], collection_name: str = "graphrag_lab"):
        texts = [d["text"] for d in documents]
        self._embedder = TfidfEmbeddingFunction(texts)

        # In-memory ChromaDB client -- nothing is written to disk.
        self._client = chromadb.EphemeralClient()
        # Fresh collection each run (avoids stale-data issues across re-runs).
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self._client.create_collection(
            name=collection_name, embedding_function=self._embedder
        )

        self.collection.add(
            ids=[d["id"] for d in documents],
            documents=texts,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:

        results = self.collection.query(query_texts=[query], n_results=top_k)

        ids = results["ids"][0]
        texts = results["documents"][0]
        distances = results["distances"][0]  # ChromaDB returns distance, not similarity

        out = []
        for doc_id, text, dist in zip(ids, texts, distances):
            # Chroma's default distance for this setup is squared L2 on the TF-IDF
            # vectors; convert to a similarity-like score in [0, 1] for readability.
            similarity = 1.0 / (1.0 + dist)
            out.append({"id": doc_id, "text": text, "score": round(similarity, 4)})
        return out


if __name__ == "__main__":
    from domain_data import DOCUMENTS

    retriever = VectorRetriever(DOCUMENTS)
    for q in ["In what year was NimbusCloud founded?", "Who co-founded QuantumLeap Robotics?"]:
        print(f"\nQuery: {q}")
        for hit in retriever.retrieve(q, top_k=5):
            print(f"  [{hit['score']}] {hit['id']}: {hit['text']}")

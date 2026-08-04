"""
Lightweight RAG document store using TF-IDF similarity.

Deliberately avoids embedding-model downloads (sentence-transformers, etc.)
so the whole pipeline runs offline after `pip install` — no HuggingFace
Hub access required. TF-IDF is a legitimate, if simpler, retrieval method:
this is still genuine retrieval-augmented generation, just with a lexical
retriever instead of a dense one. Swapping in a dense embedding model later
only requires changing this file — nothing downstream needs to change.
"""

import os
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "seed_corpus")


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class DocumentStore:
    def __init__(self, corpus_root: str = CORPUS_ROOT):
        self.corpus_root = corpus_root
        self._domain_docs: dict[str, list[tuple[str, str]]] = {}  # domain -> [(filename, text)]
        self._domain_vectorizers: dict[str, TfidfVectorizer] = {}
        self._domain_matrices = {}
        self._load()

    def _load(self):
        if not os.path.isdir(self.corpus_root):
            raise RuntimeError(f"Seed corpus not found at {self.corpus_root}")

        for domain in os.listdir(self.corpus_root):
            domain_path = os.path.join(self.corpus_root, domain)
            if not os.path.isdir(domain_path):
                continue
            docs = []
            for filename in sorted(os.listdir(domain_path)):
                if not filename.endswith(".txt"):
                    continue
                with open(os.path.join(domain_path, filename)) as f:
                    docs.append((filename, f.read()))
            if not docs:
                continue

            self._domain_docs[domain] = docs
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform([text for _, text in docs])
            self._domain_vectorizers[domain] = vectorizer
            self._domain_matrices[domain] = matrix

    def available_domains(self) -> List[str]:
        return list(self._domain_docs.keys())

    def retrieve(self, domain: str, query: str, top_k: int = 2) -> List[RetrievedChunk]:
        if domain not in self._domain_docs:
            return []

        vectorizer = self._domain_vectorizers[domain]
        matrix = self._domain_matrices[domain]
        docs = self._domain_docs[domain]

        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, matrix).flatten()

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        results = []
        for (filename, text), score in ranked[:top_k]:
            results.append(RetrievedChunk(text=text, source=filename, score=float(score)))
        return results

    def fetch_full_document(self, domain: str, filename: str) -> str | None:
        for fname, text in self._domain_docs.get(domain, []):
            if fname == filename:
                return text
        return None

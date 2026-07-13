"""
Hybrid Search Engine - BM25 + Semantic Search + Cross-Encoder Re-ranking
"""
import hashlib
import json
import os
import re

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
load_dotenv()


#embedding(semantic) and hashing-trick (keyword) embeddings

class RealEmbeddings(Embeddings):
    """Genuine semantic embeddings via sentence-transformers."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class HashingTrickEmbeddings(Embeddings):

    def __init__(self, dim=384):
        self.dim = dim

    def _embed_one(self, text):
        import numpy as np
        vec = np.zeros(self.dim)
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            sign = 1 if int(hashlib.md5((word + "_s").encode()).hexdigest(), 16) % 2 == 0 else -1
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else vec).tolist()

    def embed_documents(self, texts):
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text):
        return self._embed_one(text)


def get_embeddings():
    try:
        emb = RealEmbeddings()
        print("[embeddings] Using real sentence-transformers embeddings (semantic).")
        return emb
    except Exception:
        print("[embeddings] Could not download embedding model - falling back to "
              "keyword-hash embeddings (NOT semantic). Works fine on a machine "
              "with internet access, automatically.")
        return HashingTrickEmbeddings()


#load and split

def load_and_split(news_path, chunk_size=512, chunk_overlap=None):
    if chunk_overlap is None:
        chunk_overlap = int(chunk_size * 0.2)

    extension = os.path.splitext(news_path)[1].lower()

    if extension == ".csv":
        loader = CSVLoader(news_path)
        docs = loader.load()

    elif extension == ".txt":
        loader = TextLoader(news_path)
        docs = loader.load()

    elif extension == ".json":
        with open(news_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        for article in data:
            if isinstance(article, dict):
                text = article.get("text", article.get("content", ""))

                metadata = {
                    "title": article.get("title", "No Title"),
                    "category": article.get("category", ""),
                    "date": article.get("date", ""),
                }
            else:
                text = str(article)
                metadata = {"title": "No Title"}
            docs.append(Document(page_content=text, metadata=metadata))

    else:
        raise ValueError("Unsupported dataset format. Use CSV, TXT, or JSON.")

    print(f"[load] Loaded {len(docs)} news articles from {news_path}")

    # Token-based chunking - falls back to word-count approximation if
    # tiktoken can't download its encoding file (e.g. offline).
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        splitter._length_function("test")
    except Exception:
        def approx_token_length(text):
            return int(len(text.split()) / 0.75)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=approx_token_length,
        )

    chunks = splitter.split_documents(docs)
    print(f"[split] Split into {len(chunks)} chunks (chunk_size={chunk_size} tokens)")
    return chunks


#this function builds a hybrid retriever that combines BM25 and semantic search. It takes in the document chunks, embeddings, and optional parameters for the number of results to retrieve (k) and the weights for BM25 and semantic search. It returns the hybrid retriever along with the individual BM25 and semantic retrievers.

def build_hybrid_retriever(chunks, embeddings, k=5, bm25_weight=0.5, semantic_weight=0.5):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="news_hybrid_search",
    )
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[bm25_weight, semantic_weight],
    )
    print(f"[hybrid] Built hybrid retriever (BM25={bm25_weight}, Semantic={semantic_weight})")
    return hybrid_retriever, bm25_retriever, semantic_retriever


#this section deals with re-ranking the retrieved documents using a cross-encoder model. It defines two classes: CrossEncoderReranker, which uses a real cross-encoder model for re-ranking, and LexicalOverlapReranker, which provides a fallback method based on word overlap scoring. The get_reranker function attempts to instantiate the CrossEncoderReranker and falls back to the LexicalOverlapReranker if the model cannot be downloaded. The retrieve_with_reranking function retrieves documents using the hybrid retriever and then reranks them using the specified reranker.

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_n=5):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
        return [doc for doc, _ in ranked[:top_n]]


class LexicalOverlapReranker:

    def _score(self, query, text):
        q_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        t_words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        return len(q_words & t_words) / len(q_words) if q_words else 0.0

    def rerank(self, query, docs, top_n=5):
        ranked = sorted(docs, key=lambda d: -self._score(query, d.page_content))
        return ranked[:top_n]


def get_reranker():
    try:
        reranker = CrossEncoderReranker()
        print("[reranker] Using real cross-encoder (BAAI/bge-reranker-base).")
        return reranker
    except Exception:
        print("[reranker] Cross-encoder unavailable - using lexical overlap fallback.")
        return LexicalOverlapReranker()


#this function is used to retrieve documents using the hybrid retriever and then rerank them using the reranker. It takes a query, a hybrid retriever, a reranker, and an optional top_n parameter (defaulting to 5) to specify how many top documents to return after reranking.

def retrieve_with_reranking(query, hybrid_retriever, reranker, top_n=5):
    docs = hybrid_retriever.invoke(query)
    return reranker.rerank(query, docs, top_n=top_n)


#eval

def hit_at_k(docs, expected_keyword):
    combined = " ".join(doc.page_content.lower() for doc in docs)
    return expected_keyword.lower() in combined


def run_comparison(qa_pairs, bm25_retriever, semantic_retriever, hybrid_retriever, reranker):
    bm25_hits = semantic_hits = hybrid_hits = reranked_hits = 0
    total = len(qa_pairs)

    print(f"\nEvaluating {total} queries across 4 methods...\n")
    for pair in qa_pairs:
        question, keyword = pair["question"], pair["expected_keyword"]

        bm25_docs = bm25_retriever.invoke(question)
        bm25_hits += int(hit_at_k(bm25_docs, keyword))

        semantic_docs = semantic_retriever.invoke(question)
        semantic_hits += int(hit_at_k(semantic_docs, keyword))

        hybrid_docs = hybrid_retriever.invoke(question)
        hybrid_hits += int(hit_at_k(hybrid_docs, keyword))

        reranked_docs = reranker.rerank(question, hybrid_docs, top_n=5)
        reranked_hits += int(hit_at_k(reranked_docs, keyword))

    print()
    print("SEARCH COMPARISON")
    print()
    print(f"{'Method':<35}{'Hits':<10}{'Accuracy'}")
    print()
    print(f"{'BM25':<35}{bm25_hits}/{total:<8}{bm25_hits/total:.0%}")
    print(f"{'Semantic Search':<35}{semantic_hits}/{total:<8}{semantic_hits/total:.0%}")
    print(f"{'Hybrid Search':<35}{hybrid_hits}/{total:<8}{hybrid_hits/total:.0%}")
    print(f"{'Hybrid + Re-ranking':<35}{reranked_hits}/{total:<8}{reranked_hits/total:.0%}")

    print(f"\nHybrid improvement over BM25: {(hybrid_hits - bm25_hits) / total * 100:+.0f} percentage points")
    print(f"Hybrid improvement over Semantic: {(hybrid_hits - semantic_hits) / total * 100:+.0f} percentage points")
    print(f"Re-ranking improvement over Hybrid: {(reranked_hits - hybrid_hits) / total * 100:+.0f} percentage points")

    results = {
        "bm25": {"hits": bm25_hits, "total": total, "accuracy": bm25_hits / total},
        "semantic": {"hits": semantic_hits, "total": total, "accuracy": semantic_hits / total},
        "hybrid": {"hits": hybrid_hits, "total": total, "accuracy": hybrid_hits / total},
        "hybrid_reranked": {"hits": reranked_hits, "total": total, "accuracy": reranked_hits / total},
    }
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved comparison_results.json")
    return results


#main

if __name__ == "__main__":
    dataset_path = ""
    while dataset_path == "" or not os.path.exists(dataset_path):
        dataset_path = input("Enter path to news dataset (.csv/.txt/.json): ").strip().strip('"')
        if not os.path.exists(dataset_path):
            print("Dataset not found. Try again.")

    qa_path = ""
    while qa_path == "" or not os.path.exists(qa_path):
        qa_path = input("Enter path to qa_pairs.json: ").strip().strip('"')
        if not os.path.exists(qa_path):
            print("File not found. Try again.")

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    # --- Building the search engine ---
    chunks = load_and_split(dataset_path, chunk_size=512)
    embeddings = get_embeddings()
    hybrid, bm25_only, semantic_only = build_hybrid_retriever(chunks, embeddings, k=5)
    reranker = get_reranker()

    # --- CLI demo ---
    print()
    print("Hybrid Search CLI")
    print()

    query = input("\nEnter a search query: ").strip()
    if query:
        results = retrieve_with_reranking(query, hybrid, reranker, top_n=5)
        print(f"\nTop {len(results)} results for: '{query}'\n")
        for i, doc in enumerate(results, 1):
            title = doc.metadata.get("title", "No Title")
            print(f"{i}. {title}")
            print(f"   {doc.page_content[:250]}...")
            print()

    # --- Full 30-query comparison study ---
    run_comparison(qa_pairs, bm25_only, semantic_only, hybrid, reranker)
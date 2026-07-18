"""
Pipeline Builder — turns one configuration dict into a fully-built,
queryable RAG pipeline (chunker -> embedder -> retriever).

Now varies THREE dimensions per the project spec, not just chunk size:
  1. chunk_size - how documents are split
  2. embedding_model- which sentence-transformers model embeds chunks
  3. retrieval_strategy- semantic / bm25 / hybrid / hybrid_rerank
"""
import hashlib
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


#embeds

class RealEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
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


def get_embeddings(model_name="all-MiniLM-L6-v2"):
    try:
        emb = RealEmbeddings(model_name)
        emb.embed_query("test")
        return emb
    except Exception:
        return HashingTrickEmbeddings()


#rerankers-its used to rerank the retrieved documents based on their relevance to the query

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.model.predict([("test", "test")])  # force early failure if offline

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
        return CrossEncoderReranker()
    except Exception:
        return LexicalOverlapReranker()


#this class wraps a base retriever and a reranker behind the same .invoke(query) interface as a normal LangChain retriever, so the rest of the codebase (metrics.py, evaluation_runner.py) never needs to know reranking happened - it just calls .invoke() either way.

class RerankedRetriever:
    def __init__(self, base_retriever, reranker, top_n):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.top_n = top_n

    def invoke(self, query):
        docs = self.base_retriever.invoke(query)
        return self.reranker.rerank(query, docs, top_n=self.top_n)


#load and split

def load_and_split(corpus_path, chunk_size, chunk_overlap=None):
    if chunk_overlap is None:
        chunk_overlap = int(chunk_size * 0.2)

    loader = PyPDFLoader(corpus_path)
    docs = loader.load()

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

    return splitter.split_documents(docs)


#main

def build_pipeline(config, corpus_path):
    
    chunk_size = config["chunk_size"]
    k = config["k"]
    embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
    strategy = config.get("retrieval_strategy", "semantic")

    chunks = load_and_split(corpus_path, chunk_size)


    semantic_retriever = None
    bm25_retriever = None

    if strategy in ("semantic", "hybrid", "hybrid_rerank"):
        embeddings = get_embeddings(embedding_model)
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings,
            collection_name=f"benchmark_{config['name']}",
        )
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    if strategy in ("bm25", "hybrid", "hybrid_rerank"):
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = k

    # Assembling the final retriever based on strategy
    if strategy == "semantic":
        retriever = semantic_retriever

    elif strategy == "bm25":
        retriever = bm25_retriever

    elif strategy == "hybrid":
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5],
        )

    elif strategy == "hybrid_rerank":
        hybrid = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5],
        )
        reranker = get_reranker()
        retriever = RerankedRetriever(hybrid, reranker, top_n=k)

    else:
        raise ValueError(f"Unknown retrieval_strategy: {strategy!r}")

    return {
        "name": config["name"],
        "config": config,
        "retriever": retriever,
        "num_chunks": len(chunks),
    }
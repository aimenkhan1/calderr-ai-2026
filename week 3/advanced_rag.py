"""

ADVANCED RAG — Hybrid Search + Cross-Encoder Reranking + Multi-Query Retrieval


Flow:
    1. Load a PDF and split it into chunks
    2. Build a HYBRID retriever: BM25 (keyword) + semantic (embeddings),
       combined via LangChain's EnsembleRetriever
    3. Add CROSS-ENCODER RERANKING (BAAI/bge-reranker-base) as a second pass
       over retrieved chunks, for higher precision than embeddings alone
    4. Implement MULTI-QUERY RETRIEVAL: generate 3 phrasings of the user's
       question, retrieve for each, DEDUPLICATE overlapping results, then
       rerank the combined pool down to a final top-N
    5. MEASURE IMPROVEMENT: run all of the above against 20 Q&A pairs and
       compare hit-rate against naive (single-query, semantic-only) RAG



"""
import hashlib
import json
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()



#sec1-semantic retrieval side of hybrid search

class RealEmbeddings(Embeddings):

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


# This is a fallback embedding class that uses a simple hashing trick to create embeddings based on word occurrences. It is not semantically meaningful but can be used when the real embeddings model cannot be loaded (e.g., due to lack of internet access).
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


#sec2 - load and split pdf into chunks

def load_and_split(pdf_path, chunk_size=512, chunk_overlap=None):
    if chunk_overlap is None:
        chunk_overlap = int(chunk_size * 0.2)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[load] Loaded {len(docs)} pages from {pdf_path}")

    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        splitter._length_function("test")  # force early failure if offline
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


#sec3 - build hybrid retriever (BM25 + semantic) with LangChain's EnsembleRetriever

#
def build_hybrid_retriever(chunks, embeddings, k=5, bm25_weight=0.5, semantic_weight=0.5):
 
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="advanced_rag",
    )
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[bm25_weight, semantic_weight],
    )
    print(f"[hybrid] Built hybrid retriever (BM25 weight={bm25_weight}, "
          f"semantic weight={semantic_weight})")
    return hybrid_retriever, bm25_retriever, semantic_retriever


#sec4 - reranker (cross-encoder)

class CrossEncoderReranker:

    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_n=5):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: -x[1])
        return [doc for doc, _ in scored[:top_n]]

#fallback reranker that uses simple lexical overlap instead of a cross-encoder model. It counts the number of shared words between the query and each document, and ranks documents based on this overlap. This is not semantically meaningful but can be used when the real reranker cannot be loaded (e.g., due to lack of internet access).
class LexicalOverlapReranker:

    def _score(self, query, text):
        q_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        t_words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        return len(q_words & t_words) / len(q_words) if q_words else 0.0

    def rerank(self, query, docs, top_n=5):
        scored = sorted(docs, key=lambda d: -self._score(query, d.page_content))
        return scored[:top_n]


def get_reranker():
    try:
        reranker = CrossEncoderReranker()
        print("[reranker] Using real cross-encoder (BAAI/bge-reranker-base).")
        return reranker
    except Exception:
        print("[reranker] Could not download cross-encoder model - falling back to "
              "word-overlap reranking (NOT a real cross-encoder). Works fine on a "
              "machine with internet access, automatically.")
        return LexicalOverlapReranker()


#sec5 - multi-query retrieval (generate variations, retrieve each, dedupe, rerank)

MULTI_QUERY_PROMPT = """Generate exactly 3 different ways to ask the following question.
Each variation should use different words/phrasing but ask for the same information.
Return ONLY the 3 variations, one per line, no numbering, no extra text.

Original question: {question}"""


def get_llm():
    try:
        if not os.getenv("GROQ_API_KEY"):
            return None
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    except Exception:
        return None


def generate_query_variations(question, llm=None):
    if llm is not None:
        try:
            response = llm.invoke(MULTI_QUERY_PROMPT.format(question=question))
            variations = [line.strip() for line in response.content.split("\n") if line.strip()]
            if len(variations) >= 3:
                return variations[:3]
        except Exception as e:
            print(f"[multi-query] LLM generation failed ({e}), using rule-based fallback.")

    # Dependency-free fallback so dedup/rerank/evaluation still run without an LLM call.
    return [
        question,
        f"Can you explain: {question.rstrip('?')}?",
        f"I want to know {question[0].lower()}{question[1:].rstrip('?')}.",
    ]


#the function multi_query_retrieve takes a question, generates variations of it, retrieves documents for each variation using a hybrid retriever, deduplicates the results, and reranks them using a reranker. It returns the final top N documents along with the generated variations.
def multi_query_retrieve(question, hybrid_retriever, reranker, llm=None,final_top_n=5):
    variations = generate_query_variations(question, llm)

    all_docs = []
    seen = set()
    for variant in variations:
        for doc in hybrid_retriever.invoke(variant):
            key = doc.page_content[:200]   # dedupe by content prefix
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    reranked = reranker.rerank(question, all_docs, top_n=final_top_n)
    return reranked, variations


#sec6 - evaluation- this section contains functions to evaluate the retrieval accuracy of different methods (naive RAG, hybrid, multi-query + hybrid + rerank) against a set of Q&A pairs. It measures hits and calculates accuracy for each method.

def hit_at_k(docs, expected_keyword):
    combined = " ".join(doc.page_content.lower() for doc in docs)
    return expected_keyword.lower() in combined


def run_comparison(qa_pairs, hybrid_retriever, semantic_retriever, reranker, llm):
    naive_hits = hybrid_hits = multiquery_hits = 0
    total = len(qa_pairs)

    print(f"\nEvaluating {total} questions across 3 methods...\n")
    for pair in qa_pairs:
        question, keyword = pair["question"], pair["expected_keyword"]

        naive_hits += int(hit_at_k(semantic_retriever.invoke(question), keyword))
        hybrid_hits += int(hit_at_k(hybrid_retriever.invoke(question), keyword))
        mq_docs, _ = multi_query_retrieve(question, hybrid_retriever, reranker, llm)
        multiquery_hits += int(hit_at_k(mq_docs, keyword))

    print()
    print("RESULTS — Retrieval Accuracy Comparison")
    print()
    print(f"{'Method':<45}{'Hits':<10}{'Accuracy'}")
    print()
    print(f"{'A) Naive RAG (semantic-only, single query)':<45}{naive_hits}/{total:<8}{naive_hits/total:.0%}")
    print(f"{'B) Hybrid (BM25+semantic, single query)':<45}{hybrid_hits}/{total:<8}{hybrid_hits/total:.0%}")
    print(f"{'C) Multi-query + Hybrid + Rerank':<45}{multiquery_hits}/{total:<8}{multiquery_hits/total:.0%}")

    improvement = (multiquery_hits - naive_hits) / total * 100
    print(f"\nImprovement (C vs A): {improvement:+.0f} percentage points")

    results = {
        "naive_rag": {"hits": naive_hits, "total": total, "accuracy": naive_hits / total},
        "hybrid_only": {"hits": hybrid_hits, "total": total, "accuracy": hybrid_hits / total},
        "multiquery_hybrid_rerank": {"hits": multiquery_hits, "total": total, "accuracy": multiquery_hits / total},
        "improvement_pp": improvement,
    }
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to comparison_results.json")
    return results


#main

if __name__ == "__main__":
    # --- Get inputs interactively ---
    pdf_path = ""
    while pdf_path == "" or not os.path.exists(pdf_path):
        pdf_path = input("Enter the path: ").strip().strip('"')
        if not os.path.exists(pdf_path):
            print("That file path does not exist. Try again.")

    qa_path = ""
    while qa_path == "" or not os.path.exists(qa_path):
        qa_path = input("Enter the path to your qa_pairs.json file: ").strip().strip('"')
        if not os.path.exists(qa_path):
            print("That file path does not exist. Try again.")

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    # --- Build the full pipeline ---
    chunks = load_and_split(pdf_path, chunk_size=512) # Split the PDF into 512-sized chunks. 512 is chosen as a balance between 256 (too little context) and 1024 (too much context)
    embeddings = get_embeddings()
    hybrid, bm25_only, semantic_only = build_hybrid_retriever(chunks, embeddings, k=5)
    reranker = get_reranker()
    llm = get_llm()
    if llm is None:
        print("[multi-query] GROQ_API_KEY not set - using rule-based query variations.")

    # --- Live demo query ---
    demo_question = qa_pairs[0]["question"]
    print(f"\nDEMO QUERY: '{demo_question}'\n")

    reranked_docs, variations = multi_query_retrieve(demo_question, hybrid, reranker, llm)
    print(f"\nGenerated query variations:")
    for i, v in enumerate(variations, 1):
        print(f"  {i}. {v}")

    print(f"\nTop {len(reranked_docs)} chunks after multi-query + hybrid + rerank:")
    for i, doc in enumerate(reranked_docs, 1):
        print(f"  {i}. {doc.page_content[:150]}...")

    # --- Full evaluation vs naive RAG ---
    run_comparison(qa_pairs, hybrid, semantic_only, reranker, llm)
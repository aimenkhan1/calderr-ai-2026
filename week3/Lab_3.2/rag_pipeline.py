"""
Naive RAG Pipeline (LangChain)
Load -> Split -> Embed -> Store -> Retrieve -> Generate

"""
import hashlib
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()




def get_pdf_path():
    while True:
        pdf_path = input("Enter the full path to your PDF: ").strip().strip('"')

        if os.path.isfile(pdf_path):
            return pdf_path

        print("\n Invalid file path.")
        print("Please enter a valid PDF file.\n")


#embeddings 
class RealEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist() #normalize_embeddings=True ensures the embeddings are unit vectors


#backup if fails to load model 
#this is a fallback embedding class that uses a simple hashing trick to create embeddings based on word occurrences. It is not semantically meaningful but can be used when the real embeddings model cannot be loaded (e.g., due to lack of internet access).
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
        print("Using RealEmbeddings (all-MiniLM-L6-v2) - genuine semantic search.")
        return emb
    except Exception as e:
        print(f"Could not load sentence-transformers model ({type(e).__name__}).")
        print("Falling back to HashingTrickEmbeddings - keyword-overlap only, NOT semantic.")
        print("On your own machine with internet access this will use real embeddings.\n")
        return HashingTrickEmbeddings()


#load
def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {pdf_path}")
    return docs


#split -chunk based on token count (512 tokens with 20% overlap)
def _approx_token_length(text):
    """Fallback token counter (~0.75 tokens per word, a common rule of thumb)
    used only if tiktoken can't download its encoding file (e.g. offline)."""
    return int(len(text.split()) / 0.75)


def split_documents(docs, chunk_size=512, chunk_overlap=None):
    if chunk_overlap is None:
        chunk_overlap = int(chunk_size * 0.2)  # 20% overlap means that each chunk will share 20% of its content with the previous chunk, which helps maintain context across chunks.

    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        splitter._length_function("test")
        #if the tiktoken encoder is available, this will succeed and we can use it for accurate token counting. If not, it will raise an exception and we will fall back to the approximate word-based token count.
    except Exception:
        print("tiktoken encoding unavailable (offline) - using approximate word-based token count.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=_approx_token_length,
        )

    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (chunk_size={chunk_size} tokens, overlap={chunk_overlap})")
    return chunks


#embeds and stores
def build_vectorstore(chunks, embeddings, collection_name="rag_paper", persist_directory=None):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,  # None = in-memory
    )
    print(f"Stored {len(chunks)} chunks in Chroma collection '{collection_name}'")
    return vectorstore


#retrieves
def get_retriever(vectorstore, k=5):
    return vectorstore.as_retriever(search_kwargs={"k": k})


#generates answer
RAG_PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question, retriever, llm=None):
    retrieved_docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    if llm is None:
        # No LLM configured - just show what would be sent to the generator.
        return {
            "answer": "[No LLM configured - set GROQ_API_KEY and pass an llm to generate_answer()]",
            "context_used": context,
            "sources": retrieved_docs,
        }

    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "context_used": context,
        "sources": retrieved_docs,
    }


def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set - generation step will be skipped.")
        return None
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


#main
if __name__ == "__main__":
    pdf_path = get_pdf_path()
    docs = load_documents(pdf_path)
    chunks = split_documents(docs, chunk_size=512)
    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    retriever = get_retriever(vectorstore, k=5)
    llm = get_llm()

    question = "What retriever architecture does RAG use to find relevant passages?"
    result = generate_answer(question, retriever, llm)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nTop retrieved chunk preview:\n{result['sources'][0].page_content[:300]}...")
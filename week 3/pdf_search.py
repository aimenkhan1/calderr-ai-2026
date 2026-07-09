"""
Document Ingestion Pipeline

    Built a document ingestion pipeline for PDF pages.
    Store in ChromaDB with metadata (source, page, date).
    Query with filters.

Pipeline (followed step by step):
    1. User provides a PDF file path
    2. Extract text from the PDF, one entry per page
    3. Store each page in ChromaDB (PersistentClient - data is saved
       to disk and survives between runs) along with metadata:
           source = the PDF filename
           page   = the page number
           date   = the date it was ingested
    4. User asks a question
    5. User chooses: search with filters, or search everything (no filters)
    6. Return the matching results

Run: python ingest_pdfs.py

flow:
Enter PDF path → PdfReader loads PDF → extract_pages() extracts text per page → build metadata (source, page, date) → ChromaDB PersistentClient → collection.upsert() → embedding model converts text to vectors → stored in ChromaDB (vector + text + metadata) → User types a question → choose "all" or "filter" → build where_clause / where_document_clause (if filtering) → question gets embedded → collection.query() → filters narrow the search (if any) → cosine similarity ranks results → top matching pages returned → print_results() displays them → loop back for next question

the first time you run this, ChromaDB downloads a small
embedding model (all-MiniLM-L6-v2) automatically.
"""

import os
from datetime import date
from pypdf import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb


#this function extracts the text from each page of the PDF and returns a list of dictionaries containing the text and page number

def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:  # skip empty/blank pages
            pages.append({"text": text, "page": page_number})

    return pages


#this function ingests the PDF pages into the ChromaDB collection, storing the text and metadata for each page

def ingest_pdf(pdf_path, collection):

    filename = os.path.basename(pdf_path)
    pages = extract_pages(pdf_path)
    today = date.today().isoformat()

    ids = []
    documents = []
    metadatas = []

    for page_info in pages:
        page_num = page_info["page"]

        ids.append(f"{filename}_page_{page_num}")
        documents.append(page_info["text"])
        metadatas.append({
            "source": filename,
            "page": page_num,
            "date": today
        })

    # upsert instead of add - safe to re-run without duplicate errors
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Ingested {len(pages)} pages from {filename}")
    return len(pages)


#this function queries the ChromaDB collection for documents matching the user's question, with optional metadata and text filters

def query_documents(collection, question, where_clause, where_document_clause=None):

    results = collection.query(
        query_texts=[question],
        n_results=3,
        where=where_clause,
        where_document=where_document_clause
    )
    return results

#this function builds a metadata filter for the ChromaDB query based on the user's input, allowing filtering by source file, page range, exact page, and ingestion date
def build_filter(source=None, exclude_source=None, min_page=None, max_page=None,
                  exact_page=None, ingest_date=None):
    """Builds the ChromaDB 'where' filter (metadata filter) from whatever the user chose."""

    filters = []

    if source:
        filters.append({"source": source})

    if exclude_source:
        filters.append({"source": {"$ne": exclude_source}})

    if exact_page is not None:
        # exact page overrides a range - only makes sense to use one or the other
        filters.append({"page": {"$eq": exact_page}})
    else:
        if min_page is not None:
            filters.append({"page": {"$gte": min_page}})
        if max_page is not None:
            filters.append({"page": {"$lte": max_page}})

    if ingest_date:
        filters.append({"date": ingest_date})

    if len(filters) == 0:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        return {"$and": filters}


#this function builds a text filter for the ChromaDB query based on the user's input, allowing filtering by specific words or phrases in the document text
def build_text_filter(contains_text):

    if not contains_text:
        return None
    return {"$contains": contains_text}


def print_results(results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        print("No matching results found.")
        return

    for doc, meta in zip(documents, metadatas):
        print(f"\n[{meta['source']} - page {meta['page']} - ingested {meta['date']}]")
        print(f"  {doc[:200]}...")


#main-cli

if __name__ == "__main__":

    print()
    print("DOCUMENT INGESTION PIPELINE")
    print()

    # Step 1 - Ask for the PDF path
    pdf_path = ""
    while pdf_path == "" or not os.path.exists(pdf_path):
        pdf_path = input("\nEnter the path to your PDF file: ").strip()
        if not os.path.exists(pdf_path):
            print("That file path does not exist. Try again.")

    # PersistentClient - data is saved to disk in ./chroma_storage and will still be there the next time you run this script
    client = chromadb.PersistentClient(path="./chroma_storage")
    collection = client.get_or_create_collection(name="documents", embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"))

    # Step 2 & 3 Ingest the PDF (extract text, store with metadata)
    print("\nIngesting your PDF...")
    ingest_pdf(pdf_path, collection)
    print(f"Total pages now stored in the collection: {collection.count()}")

    filename = os.path.basename(pdf_path)

    # Step 4-6 - Ask questions in a loop
    print("\nYou can now ask questions about your document.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye.")
            break

        if question == "":
            continue

        choice = input("Search with filters, or search everything? (filter/all): ").strip().lower()

        if choice == "filter":

            print("\n  (Press Enter to skip any filter you don't want to use)\n")

            source_input = input(f"  Only search in this file? (e.g. {filename}): ").strip()
            source = source_input if source_input else None

            exclude_input = input("  Exclude a specific file from the search?: ").strip()
            exclude_source = exclude_input if exclude_input else None

            exact_input = input("  Only this exact page number?: ").strip()
            exact_page = int(exact_input) if exact_input != "" else None

            min_page = None
            max_page = None
            if exact_page is None:
                min_input = input("  From page: ").strip()
                max_input = input("  To page: ").strip()
                min_page = int(min_input) if min_input != "" else None
                max_page = int(max_input) if max_input != "" else None

            date_input = input("  Only pages ingested on this date? (YYYY-MM-DD): ").strip()
            ingest_date = date_input if date_input else None

            contains_input = input("  Only pages whose text contains this word/phrase?: ").strip()

            where_clause = build_filter(
                source=source,
                exclude_source=exclude_source,
                min_page=min_page,
                max_page=max_page,
                exact_page=exact_page,
                ingest_date=ingest_date
            )
            where_document_clause = build_text_filter(contains_input)

            print(f"\nSearching with filter: {where_clause}")
            if where_document_clause:
                print(f"Text must contain: {where_document_clause}")

        else:
            where_clause = None
            where_document_clause = None
            print("\nSearching everything (no filters)...")

        results = query_documents(collection, question, where_clause, where_document_clause)
        print_results(results)
        print()
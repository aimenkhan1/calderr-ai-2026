    
"""
Semantic Memory Store — facts about the user, extracted from conversation

architecture:

conversation text -> extract_facts() -> LLM returns JSON -> validated into
UserFacts(Pydantic) -> store_facts() -> for each Fact: sqlite row + chromadb
vector -> return list of stored Fact objects

query -> query_facts() -> chromadb(cosine similarity) -> top 5 fact ids ->
sqlite complete rows -> fetch_by_ids -> return

LLM backend: real Groq if GROQ_API_KEY is set, else an offline rule-based
extractor (regex/keyword patterns) so this can be built and tested without
an API key or internet.
"""

import os
import re
import json
import time
import uuid
import hashlib
import sqlite3
from typing import Optional, Literal

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from groq import Groq
load_dotenv() 


client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"





class OfflineHashEmbedding(EmbeddingFunction):
    def __init__(self, dims: int = 384):
        self.dims = dims

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = [0.0] * self.dims
            for word in text.lower().split():
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims
                vec[bucket] += 1.0
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors




class Fact(BaseModel):
    fact_type: Literal["name", "preference", "goal", "constraint"]
    content: str = Field(..., description="the fact itself, in plain words")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class UserFacts(BaseModel):
    facts: list[Fact] = Field(default_factory=list)


# stored version of a fact — adds the bookkeeping fields (id, user, timestamp)
# that aren't part of what the LLM extracts, but are needed for storage/retrieval
class StoredFact:
    def __init__(self, fact_id, user, timestamp, fact_type, content, confidence):
        self.fact_id = fact_id
        self.user = user
        self.timestamp = timestamp
        self.fact_type = fact_type
        self.content = content
        self.confidence = confidence




EXTRACTION_SYSTEM_PROMPT = """You extract structured facts about a user from a conversation.
Return ONLY valid JSON matching this exact shape, nothing else:

{"facts": [{"fact_type": "name" | "preference" | "goal" | "constraint", "content": "...", "confidence": 0.0-1.0}]}

Rules:
- "name" = the user's own name
- "preference" = something they like/dislike/prefer
- "goal" = something they want to achieve or are working towards
- "constraint" = a limitation: budget, deadline, tech stack requirement, thing they can't do
- Only extract facts actually stated. If nothing fits a category, don't invent one.
- If there are no facts at all, return {"facts": []}
"""


def call_groq_extract(conversation_text: str) -> UserFacts:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": conversation_text,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = response.choices[0].message.content

    # Convert JSON string into a Pydantic object
    return UserFacts.model_validate_json(raw)


#fallback
def stub_extract(conversation_text: str) -> UserFacts:
    facts = []
    text = re.sub(r"\s+", " ", conversation_text).strip()

    name_match = re.search(r"\b(?i:my name is|i'?m|i am)\s+([A-Z][a-zA-Z]+)", text)
    if name_match:
        facts.append(Fact(fact_type="name", content=name_match.group(1), confidence=0.9))


    for m in re.finditer(r"\bi (?:like|love|prefer)\s+([^.,!?\n]+)", text, re.IGNORECASE):
        facts.append(Fact(fact_type="preference", content=m.group(0).strip(), confidence=0.7))


    for m in re.finditer(
        r"\b(?:i want to|my goal is|i'?m trying to|i'?m working on)\s+([^.,!?\n]+)",
        text, re.IGNORECASE,
    ):
        facts.append(Fact(fact_type="goal", content=m.group(0).strip(), confidence=0.7))


    for m in re.finditer(
        r"\b(?:i can'?t|i don'?t have|no budget|deadline is|only have)\s+([^.,!?\n]+)",
        text, re.IGNORECASE,
    ):
        facts.append(Fact(fact_type="constraint", content=m.group(0).strip(), confidence=0.7))

    return UserFacts(facts=facts)


def extract_facts(conversation_text: str) -> UserFacts:

    if "GROQ_API_KEY" in os.environ:
        try:
            return call_groq_extract(conversation_text)
        except (ValidationError, Exception):
            pass   # fall through to the offline extractor below
    return stub_extract(conversation_text)



#store
class SemanticMemoryStore:

    def __init__(self, sqlite_path: str = "semantic_memory.db", chroma_path: str = "semantic_chroma"):
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id    TEXT PRIMARY KEY,
                user       TEXT NOT NULL,
                timestamp  REAL NOT NULL,
                fact_type  TEXT NOT NULL,
                content    TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        """)
        self.conn.commit()

        self.chroma = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma.get_or_create_collection(
            name="user_facts",
            metadata={"hnsw:space": "cosine"},
            embedding_function=OfflineHashEmbedding(),   # no internet/model download needed
        )

    # write 
    def store_facts(self, user: str, facts: UserFacts) -> list[StoredFact]:
        stored = []
        for fact in facts.facts:
            stored.append(self._log(user, fact))
        return stored

    def _log(self, user: str, fact: Fact) -> StoredFact:
        fact_id = str(uuid.uuid4())
        timestamp = time.time()

        self.conn.execute(
            "INSERT INTO facts (fact_id, user, timestamp, fact_type, content, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (fact_id, user, timestamp, fact.fact_type, fact.content, fact.confidence),
        )
        self.conn.commit()

        self.collection.add(
            ids=[fact_id],
            documents=[fact.content],
            metadatas=[{
                "user": user,
                "fact_type": fact.fact_type,
                "timestamp": timestamp,
                "confidence": fact.confidence,
            }],
        )

        return StoredFact(fact_id, user, timestamp, fact.fact_type, fact.content, fact.confidence)

    # read and search
    def query_facts(self, user: str, query: str, top_k: int = 5) -> list[StoredFact]:
        results = self.collection.query(
            query_texts=[query], n_results=top_k, where={"user": user}
        )

        ids = results["ids"][0]

        if not ids:
            return []

        return self._fetch_by_ids(ids)

    # plain sql pull of every known fact about a user, grouped by type
    def get_all_facts(self, user: str) -> list[StoredFact]:
        rows = self.conn.execute(
            "SELECT fact_id, user, timestamp, fact_type, content, confidence "
            "FROM facts WHERE user = ? ORDER BY fact_type, timestamp",
            (user,),
        ).fetchall()
        return [StoredFact(*row) for row in rows]

    # fetching facts by their IDs, preserving the order of relevance from ChromaDB.
    def _fetch_by_ids(self, ids: list[str]) -> list[StoredFact]:
        placeholders = ",".join("?" * len(ids))

        rows = self.conn.execute(
            f"SELECT fact_id, user, timestamp, fact_type, content, confidence "
            f"FROM facts WHERE fact_id IN ({placeholders})",
            ids,
        ).fetchall()

        by_id = {r[0]: StoredFact(*r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]


#main run

if __name__ == "__main__":
    store = SemanticMemoryStore()

    sample_conversation = """
    Hi, I'm Aiman. I'm a computer science student at FAST NUCES Karachi.
    I prefer Python over Java for most of my projects. My goal is to finish
    my FYP on diabetic retinopathy detection before the semester deadline.
    I don't have a GPU on my laptop, so I can't train large models locally.
    """

    print("===== Extracting facts from sample conversation =====")
    facts = extract_facts(sample_conversation)
    for f in facts.facts:
        print(f"[{f.fact_type}] {f.content} (confidence={f.confidence})")

    print("\n===== Storing facts =====")
    stored = store.store_facts("aiman", facts)
    print(f"Stored {len(stored)} facts.")

    print("\n===== Semantic query: 'what programming language' =====")
    for f in store.query_facts("aiman", "what programming language does the user like"):
        print(f"[{f.fact_type}] {f.content}")

    print("\n===== Semantic query: 'any limitations on hardware' =====")
    for f in store.query_facts("aiman", "any limitations on hardware"):
        print(f"[{f.fact_type}] {f.content}")

    print("\n===== All known facts about aiman =====")
    for f in store.get_all_facts("aiman"):
        print(f"[{f.fact_type}] {f.content}")


    while True:
        print("\n========== MENU FOR ANY NEW CONVO ==========")
        print("1. Enter a new conversation")
        print("2. Ask a semantic query")
        print("3. Show all stored facts")
        print("4. Exit")
    
        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            conversation = input("\nEnter conversation:\n")

            facts = extract_facts(conversation)

            print("\nExtracted Facts:")
            for f in facts.facts:
                print(f"[{f.fact_type}] {f.content} (confidence={f.confidence})")

            stored = store.store_facts("aiman", facts)
            print(f"\nStored {len(stored)} new fact(s).")

        elif choice == "2":
            query = input("\nEnter your semantic query: ")

            results = store.query_facts("aiman", query)

            if not results:
                print("No relevant facts found.")
            else:
                print("\nRelevant Facts:")
            for f in results:
                print(f"[{f.fact_type}] {f.content}")

        elif choice == "3":
            print("\n===== All Known Facts =====")
            facts = store.get_all_facts("aiman")

            if not facts:
                print("No facts stored.")
            else:
                for f in facts:
                    print(f"[{f.fact_type}] {f.content}")

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice.")


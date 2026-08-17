"""
Lab 6.1 - Memory-Augmented Chatbot (single file version)

A CLI chatbot backed by two memory stores:
  - SQLite  : the episodic log — raw interaction history with timestamps
  - ChromaDB: semantic index — embedded summaries of past sessions (plus
              the raw turns too), searchable by meaning

architecture:

logging a turn....
user/bot message -> log_turn() -> generate memory id(uuid), timestamp, importance ->sqlite stored, chromadb(id+vector) -> return Memory object

logging a session summary....
end of session -> summarize_session() -> log_summary() -> same path as above but kind="summary" instead of kind="turn"


retrieving....
new user message -> retrieve_relevant() -> chromadb(cosine similarity,
excluding current session) -> top 5 memory ids -> sqlite complete rows ->
build_system_prompt(memories) -> call_groq() or stub_reply() -> print reply ->
log_turn(user) + log_turn(assistant) -> loop

LLM backend: real Groq if GROQ_API_KEY is set in the environment, else an
offline stub that just proves the retrieval is working (no internet/API
key needed to test it).

validation: run this file 3 separate times (3 separate sessions). in the
3rd run,we ask about something only mentioned in the 1st run the bot must
answer correctly using retrieved memory, even though the 1st session's
text is nowhere in the 3rd session's own context.
"""

import os
import uuid
import json
import hashlib
import sqlite3
import time
from typing import Optional
from dotenv import load_dotenv
from groq import Groq
load_dotenv() 

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

#offline fallback embedding function — no real model, just a simple hash-based vectorizer that works without internet or model downloads. 
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
            norm = sum(v * v for v in vec) ** 0.5  # normalize so length doesn't skew similarity
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


#data

class Memory:
    def __init__(self, memory_id, session_id, timestamp, role, kind, content, importance):
        self.memory_id = memory_id
        self.session_id = session_id   
        self.timestamp = timestamp
        self.role = role             
        self.kind = kind            
        self.content = content
        self.importance = importance


def score_importance(content: str) -> float:
    cues = ["remember", "always", "never", "important", "my name is",
            "i prefer", "i like", "i love", "deadline", "note that"]

    text = content.lower()

    score = 0.3

    score += 0.4 * any(cue in text for cue in cues)

    score += min(len(content) / 500, 0.2)

    score += 0.1 * ("?" in content)

    return round(min(score, 1.0), 3) 


#sqlite for raw logs and chromedb for semantic search

class ChatMemoryStore:

    def __init__(self, sqlite_path: str = "chat_memory.db", chroma_path: str = "chat_chroma"):
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                memory_id  TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp  REAL NOT NULL,
                role       TEXT NOT NULL,
                kind       TEXT NOT NULL,
                content    TEXT NOT NULL,
                importance REAL NOT NULL
            )
        """)
        self.conn.commit()

        # persistent client stores the embeddings for long term storage and retrieval.
        self.chroma = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma.get_or_create_collection(
            name="chat_memory",
            metadata={"hnsw:space": "cosine"},
            embedding_function=OfflineHashEmbedding(),  
        )

    # write and log a single chat turn (user message or assistant reply)
    def log_turn(self, session_id: str, role: str, content: str) -> Memory:
        return self._log(session_id, role, "turn", content)

    # write and log a session summary once a session ends
    def log_summary(self, session_id: str, summary_text: str) -> Memory:
        return self._log(session_id, "summary", "summary", summary_text)

    def _log(self, session_id: str, role: str, kind: str, content: str) -> Memory:
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        importance = score_importance(content)

        self.conn.execute(
            "INSERT INTO interactions "
            "(memory_id, session_id, timestamp, role, kind, content, importance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory_id, session_id, timestamp, role, kind, content, importance),
        )
        self.conn.commit()  # saving it to disk right away

        self.collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[{
                "session_id": session_id,
                "role": role,
                "kind": kind,
                "timestamp": timestamp,
                "importance": importance,
            }],
        )

        return Memory(memory_id, session_id, timestamp, role, kind, content, importance)

    # reading and searching — semantic search across ALL past sessions,optionally excluding the session that's currently running
    def retrieve_relevant(self, query: str, top_k: int = 5,
                           exclude_session: Optional[str] = None) -> list[Memory]:
        where = {"session_id": {"$ne": exclude_session}} if exclude_session else None

        results = self.collection.query(
            query_texts=[query], n_results=top_k, where=where
        )

        ids = results["ids"][0]

        if not ids:
            return []

        return self._fetch_by_ids(ids)

    # plain sql pull of one session's raw turns, used to build its summary
    def get_session_turns(self, session_id: str) -> list[Memory]:
        rows = self.conn.execute(
            "SELECT memory_id, session_id, timestamp, role, kind, content, importance "
            "FROM interactions WHERE session_id = ? AND kind = 'turn' "
            "ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
        return [Memory(*row) for row in rows]

    # fetching memories by their IDs, preserving the order of relevance from ChromaDB.
    def _fetch_by_ids(self, ids: list[str]) -> list[Memory]:
        placeholders = ",".join("?" * len(ids))

        rows = self.conn.execute(
            f"SELECT memory_id, session_id, timestamp, role, kind, content, importance "
            f"FROM interactions WHERE memory_id IN ({placeholders})",
            ids,
        ).fetchall()

        by_id = {r[0]: Memory(*r) for r in rows}  # mapping memory_id to Memory object for quick lookup.
        return [by_id[i] for i in ids if i in by_id]



def call_groq(system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


#fallback-if groq not available
def stub_reply(memories: list[Memory], user_message: str) -> str:
    if not memories:
        return "I don't have any relevant memory of that yet."

    STOPWORDS = {"a", "an", "the", "is", "are", "i", "my", "did", "what",
                 "was", "you", "to", "of", "and", "it", "that", "this"}
    q_words = set(user_message.lower().split()) - STOPWORDS

    def overlap(m):
        return len(q_words & (set(m.content.lower().split()) - STOPWORDS))

    best = max(memories, key=overlap)

    if overlap(best) < 1:  # best match still isn't really relevant
        return "I don't have a relevant memory for that."

    return f"(recalling a past session) You mentioned: \"{best.content}\""



def build_system_prompt(memories: list[Memory]) -> str:
    if not memories:
        return "You are a helpful assistant. You have no relevant past memories for this message."

    lines = ["You are a helpful assistant with access to memories from PAST chat sessions.",
             "Use them if they are relevant to the user's new message:\n"]

    for m in memories:
        lines.append(f"- (from a previous {m.role} message) {m.content}")

    lines.append("\nIf a memory answers the user's question, use it directly and confidently.")

    return "\n".join(lines)



def summarize_session(turns: list[Memory]) -> str:
    if not turns:
        return "Empty session — nothing was discussed."

    user_lines = [t.content for t in turns if t.role == "user"]

    return "Session covered: " + " | ".join(user_lines)


#chat-one session

def run_chat(session_id: str = None, input_lines: list[str] = None) -> list[str]:
    store = ChatMemoryStore()
    session_id = session_id or str(uuid.uuid4())
    use_groq = "GROQ_API_KEY" in os.environ

    print(f"\n===== Memory-Augmented Chatbot — session {session_id[:8]} =====")
    print(f"LLM backend: {'Groq' if use_groq else 'offline stub'}")
    print("Type 'exit' to end the session.\n")

    replies = []
    line_iter = iter(input_lines) if input_lines is not None else None

    while True:
        if line_iter is not None:
            try:
                user_message = next(line_iter)
            except StopIteration:
                break
            print(f"You: {user_message}")
        else:
            user_message = input("You: ").strip()

        if user_message.lower() in ("exit", "quit"):
            break

        if not user_message:
            continue

        # reading and searching both stores — chromadb finds the matching
        # ids (semantic), and internally that gets joined back to the
        # sqlite rows for the full record. current session excluded so it
        # doesn't just match itself.
        memories = store.retrieve_relevant(
            query=user_message, top_k=5, exclude_session=session_id
        )

        system_prompt = build_system_prompt(memories)

        if use_groq:
            try:
                reply = call_groq(system_prompt, user_message)
            except Exception as e:
                reply = f"[Groq call failed: {e}]"
        else:
            reply = stub_reply(memories, user_message)

        print(f"Bot: {reply}\n")
        replies.append(reply)

        # write and log this turn so it becomes a searchable memory for
        # future sessions
        store.log_turn(session_id, "user", user_message)
        store.log_turn(session_id, "assistant", reply)

    # end of session — write a summary as one more memory
    turns = store.get_session_turns(session_id)
    summary = summarize_session(turns)
    store.log_summary(session_id, summary)
    print(f"[session summary saved]: {summary}\n")

    return replies


if __name__ == "__main__":
    run_chat()
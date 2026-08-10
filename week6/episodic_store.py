"""
Episodic Store practise  — Episodic memory system using SQLite for persistent 
structured storage and ChromaDB for semantic similarity search.


architecture:

inserting.... 
user input->episodicstore(log_episode())->generate episode id(uuid),timestamp,importance->
sqlite stored,chromedb(id+vector)->return episode object          

searching... 
user query->search_episode()->chromedb(cosine similalrity)->top 5 ids->sqlite db complete row->
first inserting placeholder ?->fetch_by_ids(reconcstruct epsiode)->return
"""

import sqlite3   
import time       
import uuid    
from typing import Optional       
import chromadb   


class Episode:
    def __init__(self, episode_id, timestamp, user, content, importance):
        self.episode_id = episode_id
        self.timestamp = timestamp
        self.user = user
        self.content = content
        self.importance = importance


def score_importance(content: str) -> float:
    cues = ["remember", "always", "never", "important", "my name is",
            "i prefer", "deadline", "password", "note that"]

    text = content.lower()         

    score = 0.3                 

    score += 0.4 * any(cue in text for cue in cues)

    score += min(len(content) / 500, 0.2)

#class Episode:#Due to SQL injection prevention—the ? placeholders let SQLite safely bind the ChromaDB IDs as parameters instead of directly inserting them into the SQL query.
    score += 0.1 * ("?" in content)

    return round(min(score, 1.0), 3)  #imp score can goes upto 1.0



#this class handles the episodic memory system, using SQLite for structured storage and ChromaDB for semantic search.
class EpisodicStore:

    def __init__(self, sqlite_path: str = "episodes.db",chroma_path: str = "chroma_store"):
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,   
                timestamp  REAL NOT NULL,     
                user       TEXT NOT NULL,    
                content    TEXT NOT NULL,      
                importance REAL NOT NULL      
            )
        """)
        self.conn.commit()  

#persistant client stores the embedding for long term storage and retrieval.
        self.chroma = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"},
        )


#write and log episodes
    def log_episode(self, user: str, content: str,importance: Optional[float] = None) -> Episode:

        episode_id = str(uuid.uuid4())
        timestamp = time.time()
        importance = importance if importance is not None else score_importance(content)


        self.conn.execute(
            "INSERT INTO episodes (episode_id, timestamp, user, content, importance) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, timestamp, user, content, importance),
        )
        self.conn.commit()   # saving it to disk right away


        self.collection.add(
            ids=[episode_id],       
            documents=[content],        
            metadatas=[{"user": user, "timestamp": timestamp, "importance": importance}],
        )


        return Episode(episode_id, timestamp, user, content, importance)


#reading and searching
    def search_episodes(self, query: str, top_k: int = 5,user: Optional[str] = None) -> list[Episode]:

        where = {"user": user} if user else None
        results = self.collection.query(
            query_texts=[query], n_results=top_k, where=where
        )

        ids = results["ids"][0]

        if not ids:         
            return []

        return self._fetch_by_ids(ids)

    def get_recent(self, n: int = 10, user: Optional[str] = None) -> list[Episode]:
        q = "SELECT episode_id, timestamp, user, content, importance FROM episodes"
        params = ()

        if user:
            q += " WHERE user = ?"
            params = (user,)

        q += " ORDER BY timestamp DESC LIMIT ?"

        rows = self.conn.execute(q, (*params, n)).fetchall()

        # Converting each raw SQL row (a tuple) into a Episode object.
        return [Episode(*row) for row in rows]

    def get_important(self, threshold: float = 0.6) -> list[Episode]:
        rows = self.conn.execute(
            "SELECT episode_id, timestamp, user, content, importance "
            "FROM episodes WHERE importance >= ? ORDER BY importance DESC",
            (threshold,),
        ).fetchall()
        return [Episode(*row) for row in rows]


#fetching episodes by their IDs, preserving the order of relevance from ChromaDB.
    def _fetch_by_ids(self, ids: list[str]) -> list[Episode]:


        placeholders = ",".join("?" * len(ids))

        rows = self.conn.execute(
            f"SELECT episode_id, timestamp, user, content, importance "
            f"FROM episodes WHERE episode_id IN ({placeholders})",
            ids,
        ).fetchall()


        by_id = {r[0]: Episode(*r) for r in rows} # Mapping episode_id to Episode object for quick lookup.
        return [by_id[i] for i in ids if i in by_id] 




#main 

if __name__ == "__main__":

    store = EpisodicStore(
        sqlite_path="demo_episodes.db",
        chroma_path="demo_chroma"
    )

    while True:
        print("\n===== Episodic Memory System =====")
        print("1. Add new memory")
        print("2. Search memories")
        print("3. Show recent memories")
        print("4. Show important memories")
        print("5. Exit")

        choice = input("\nEnter your choice: ").strip()

        # 1. Add a new episode
        if choice == "1":
            user = input("Enter user name: ").strip()
            content = input("Enter your memory/message: ").strip()

            episode = store.log_episode(user, content)

            print("\nMemory saved successfully!")
            print("Episode ID:", episode.episode_id)
            print("Importance:", episode.importance)

        # 2. Semantic search
        elif choice == "2":
            query = input("What do you want to search for? ").strip()

            results = store.search_episodes(query, top_k=5)

            print("\n--- Search Results ---")

            if not results:
                print("No matching memories found.")
            else:
                for ep in results:
                    print(
                        f"[{ep.importance:.2f}] "
                        f"{ep.user}: {ep.content}"
                    )

        # 3. Show recent memories
        elif choice == "3":
            n = int(input("How many recent memories? ").strip() or "5")

            print("\n--- Recent Episodes ---")

            for ep in store.get_recent(n):
                print(
                    f"[{ep.importance:.2f}] "
                    f"{ep.user}: {ep.content}"
                )

        # 4. Show important memories
        elif choice == "4":
            threshold = float(
                input("Enter importance threshold (0-1): ").strip() or "0.6"
            )

            print("\n--- Important Episodes ---")

            results = store.get_important(threshold)

            if not results:
                print("No important memories found.")
            else:
                for ep in results:
                    print(
                        f"[{ep.importance:.2f}] "
                        f"{ep.user}: {ep.content}"
                    )

        # 5. Exit
        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


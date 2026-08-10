"""
User Profile Builder — synthesizes a structured profile from episodic memory

architecture:                                                      

log_interaction() -> sqlite episodes table -> every 10 calls ->
synthesize_profile() -> LLM reads last N episodes -> UserProfile(Pydantic) ->
overwrite the stored profile for this user

generate_response() -> get_profile() -> if profile exists, adjust tone/depth
based on it -> else generic reply

LLM backend: real Groq if GROQ_API_KEY is set, else an offline rule-based
synthesizer, so this can be built/tested without an API key or internet.
"""

import os
import re
import json
import time
import uuid
import sqlite3
from typing import Optional, Literal

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from groq import Groq
load_dotenv() 


client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"
INTERACTIONS_PER_SYNTHESIS = 10


#pydantic

class UserProfile(BaseModel):
    name: Optional[str] = None
    expertise_level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    communication_style: Literal["concise", "detailed"] = "detailed"
    interests: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


#episodic log

class Episode:
    def __init__(self, episode_id, user, timestamp, content):
        self.episode_id = episode_id
        self.user = user
        self.timestamp = timestamp
        self.content = content


class EpisodicLog:
    def __init__(self, sqlite_path: str = "profile_episodes.db"):
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                user       TEXT NOT NULL,
                timestamp  REAL NOT NULL,
                content    TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def log(self, user: str, content: str) -> Episode:
        episode_id = str(uuid.uuid4())
        timestamp = time.time()

        self.conn.execute(
            "INSERT INTO episodes (episode_id, user, timestamp, content) VALUES (?, ?, ?, ?)",
            (episode_id, user, timestamp, content),
        )
        self.conn.commit()

        return Episode(episode_id, user, timestamp, content)

    def count_for_user(self, user: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE user = ?", (user,)
        ).fetchone()
        return row[0]

    def get_all_for_user(self, user: str) -> list[Episode]:
        rows = self.conn.execute(
            "SELECT episode_id, user, timestamp, content FROM episodes "
            "WHERE user = ? ORDER BY timestamp ASC",
            (user,),
        ).fetchall()
        return [Episode(*row) for row in rows]


#profilestore-rewritten everytime it resynthesized

class ProfileStore:
    def __init__(self, sqlite_path: str = "profile_episodes.db"):
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                user            TEXT PRIMARY KEY,
                profile_json    TEXT NOT NULL,
                updated_at      REAL NOT NULL,
                interactions_at_update INTEGER NOT NULL
            )
        """)
        self.conn.commit()

    def save(self, user: str, profile: UserProfile, interaction_count: int):
        self.conn.execute(
            "INSERT INTO profiles (user, profile_json, updated_at, interactions_at_update) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user) DO UPDATE SET "
            "profile_json=excluded.profile_json, "
            "updated_at=excluded.updated_at, "
            "interactions_at_update=excluded.interactions_at_update",
            (user, profile.model_dump_json(), time.time(), interaction_count),
        )
        self.conn.commit()

    def get(self, user: str) -> Optional[UserProfile]:
        row = self.conn.execute(
            "SELECT profile_json FROM profiles WHERE user = ?", (user,)
        ).fetchone()
        if not row:
            return None
        return UserProfile.model_validate_json(row[0])



SYNTHESIS_SYSTEM_PROMPT = """You build a user profile from their conversation history.
Return ONLY valid JSON matching this exact shape, nothing else:

{"name": "..." or null,
 "expertise_level": "beginner" | "intermediate" | "advanced",
 "communication_style": "concise" | "detailed",
 "interests": ["..."],
 "goals": ["..."],
 "constraints": ["..."]}

Base expertise_level on the vocabulary and questions asked (basic "what is X"
questions = beginner; technical/optimization questions = advanced).
Base communication_style on how the user themselves writes (short messages =
concise; long explanatory messages = detailed).
"""


def call_groq_synthesize(history_text: str) -> UserProfile:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYNTHESIS_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": history_text,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = response.choices[0].message.content

    # Convert JSON string into a Pydantic UserProfile object
    return UserProfile.model_validate_json(raw)


def stub_synthesize(episodes: list[Episode]) -> UserProfile:
    joined = " ".join(e.content for e in episodes)
    text = re.sub(r"\s+", " ", joined).strip()
    lower = text.lower()

    name = None
    name_match = re.search(r"\b(?i:my name is|i'?m|i am)\s+([A-Z][a-zA-Z]+)", text)
    if name_match:
        name = name_match.group(1)

    beginner_cues = ["what is", "new to", "beginner", "explain simply",
                      "i don't understand", "eli5", "just starting"]
    advanced_cues = ["optimize", "under the hood", "time complexity",
                      "edge case", "benchmark", "architecture", "internals"]
    beginner_hits = sum(lower.count(c) for c in beginner_cues)
    advanced_hits = sum(lower.count(c) for c in advanced_cues)
    if advanced_hits > beginner_hits:
        expertise_level = "advanced"
    elif beginner_hits > 0:
        expertise_level = "beginner"
    else:
        expertise_level = "intermediate"

    avg_len = sum(len(e.content) for e in episodes) / max(len(episodes), 1)
    communication_style = "concise" if avg_len < 60 else "detailed"


    interests = [m.group(0).strip() for m in re.finditer(
        r"\bi (?:like|love|prefer|enjoy)\s+([^.,!?\n]+)", text, re.IGNORECASE)]
    goals = [m.group(0).strip() for m in re.finditer(
        r"\b(?:i want to|my goal is|i'?m trying to|i'?m working on)\s+([^.,!?\n]+)",
        text, re.IGNORECASE)]
    constraints = [m.group(0).strip() for m in re.finditer(
        r"\b(?:i can'?t|i don'?t have|no budget|deadline is|only have)\s+([^.,!?\n]+)",
        text, re.IGNORECASE)]

    return UserProfile(
        name=name,
        expertise_level=expertise_level,
        communication_style=communication_style,
        interests=interests[:5],
        goals=goals[:5],
        constraints=constraints[:5],
    )


def synthesize_profile(episodes: list[Episode]) -> UserProfile:
    if "GROQ_API_KEY" in os.environ:
        try:
            history_text = "\n".join(e.content for e in episodes)
            return call_groq_synthesize(history_text)
        except (ValidationError, Exception):
            pass   # fall through to the offline synthesizer
    return stub_synthesize(episodes)


#agent-logs and synthesizes anf generate on basis of that profile

class ProfileAwareAgent:
    def __init__(self, sqlite_path: str = "profile_episodes.db"):
        self.log = EpisodicLog(sqlite_path)
        self.profiles = ProfileStore(sqlite_path)

    def handle_message(self, user: str, message: str) -> str:
        # 1) log this interaction into episodic memory
        self.log.log(user, message)
        count = self.log.count_for_user(user)

        # 2) auto-update: every 10 interactions, re-synthesize the profile from the FULL episodic history so far
        if count % INTERACTIONS_PER_SYNTHESIS == 0:
            episodes = self.log.get_all_for_user(user)
            profile = synthesize_profile(episodes)
            self.profiles.save(user, profile, count)

        # 3) generate a reply, shaped by whatever profile currently exists (None until the 10th interaction fires the first synthesis)
        profile = self.profiles.get(user)
        return self._generate_response(message, profile)

    def _generate_response(self, message: str, profile: Optional[UserProfile]) -> str:
        if profile is None:
            # no profile yet — generic, one-size-fits-all reply
            return f"[generic reply] Here's an answer about: {message}"

        # profile exists — behaviour actually changes based on it
        prefix = f"Hey {profile.name}, " if profile.name else ""

        if profile.expertise_level == "beginner":
            body = (f"let me explain this in simple terms with an analogy, "
                     f"step by step: {message}")
        elif profile.expertise_level == "advanced":
            body = f"skipping the basics — here's the technical detail on: {message}"
        else:
            body = f"here's a balanced explanation of: {message}"

        if profile.communication_style == "concise":
            body = body.split(":")[0] + ": [short, to-the-point answer]"
        else:
            body += " [with full context and examples]"

        return f"[profile-adapted reply] {prefix}{body}"


#main

if __name__ == "__main__":
    for f in ("profile_episodes.db",):
        if os.path.exists(f):
            os.remove(f)

    agent = ProfileAwareAgent()
    user = "aiman"

    turns = [
        "Hi, I'm Aiman, I'm new to machine learning.",
        "What is a neural network?",
        "What is overfitting?",
        "I don't understand backpropagation, can you explain simply?",
        "What is a loss function?",
        "I'm just starting out with Python too.",
        "What is a for loop?",
        "I like drawing in my free time.",
        "My goal is to build a simple chatbot.",
        "I don't have a powerful laptop for training big models.",
        "How do I make my chatbot better?",
        "What should I learn next?",
    ]

    for i, message in enumerate(turns, start=1):
        reply = agent.handle_message(user, message)
        print(f"--- turn {i} ---")
        print(f"User: {message}")
        print(f"Bot : {reply}\n")

    print("===== Final synthesized profile =====")
    profile = agent.profiles.get(user)
    print(profile.model_dump_json(indent=2))
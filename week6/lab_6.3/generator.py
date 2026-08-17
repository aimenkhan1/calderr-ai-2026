"""
generator.py

Turns (question, merged_context) into a final answer.


"""

from __future__ import annotations

import os
import re
from typing import List


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "of", "to",
    "for", "and", "or", "who", "what", "when", "where", "which", "did", "does",
    "do", "that", "this", "with", "by", "from", "go", "went", "into",
}


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _extractive_generate(question: str, context: List[dict]) -> str:

    if not context:
        return "I don't have enough retrieved information to answer this question."

    q_words = _tokenize(question)
    scored = []
    for item in context:
        c_words = _tokenize(item["text"])
        overlap = len(q_words & c_words)
        scored.append((overlap, item["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_sentences = [text for overlap, text in scored if overlap > 0][:4]

    if not top_sentences:

        top_sentences = [context[0]["text"]]

    return " ".join(top_sentences)


def _groq_generate(question: str, context: List[dict]) -> str:

    from groq import Groq

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    context_block = "\n".join(
        f"- {item['text']}" for item in context
    )

    prompt = (
        "Answer the question using ONLY the facts below. "
        "Be concise (1-2 sentences).\n\n"
        f"Facts:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def generate(question: str, context: List[dict]) -> str:

    if os.environ.get("GROQ_API_KEY"):
        try:
            return _groq_generate(question, context)
        except Exception as e:
            print(f"[generator] Groq call failed ({e}), falling back to extractive mode.")
            return _extractive_generate(question, context)

    return _extractive_generate(question, context)


if __name__ == "__main__":
    demo_context = [
        {"id": "d1", "text": "Aria Solano founded NimbusCloud in 2015.", "score": 0.9},
        {"id": "d2", "text": "NimbusCloud is headquartered in Austin.", "score": 0.6},
    ]
    print(generate("In what year was NimbusCloud founded?", demo_context))

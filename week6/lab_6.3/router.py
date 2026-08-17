"""
router.py

Decides, for a given question, which retrieval mode should handle it:
  - "vector"  -> a single, standalone fact (e.g. "What year was X founded?")
  - "graph"   -> a direct relationship/connection between two known entities
                 (e.g. "Who co-founded X with Y?")
  - "hybrid"  -> a multi-hop question that needs a relationship AND a
                 specific fact chained together (e.g. "What product did the
                 company founded by the person who worked at X create?")

"""

from __future__ import annotations


#for hybrid
NESTED_CLAUSE_MARKERS = [
    "the person who",
    "the company that",
    "the company where",
    "the company founded by",
    "before co-founding",
    "who previously",
    "go on to create",
    "originally founded",
]

#for graph
RELATION_KEYWORDS = [
    "founded by",
    "co-founded",
    "co-founder",
    "worked at",
    "works at",
    "work at",
    "acqui",  # covers "acquire" / "acquired" / "acquisition"
    "partnered",
    "partnership",
    "alongside",
    "both",
    "together with",
    "which two companies",
    "which company did",
    "which person",
    "connection between",
    "relationship between",
]


def route(question: str) -> str:
    q_lower = question.lower()


    if any(marker in q_lower for marker in NESTED_CLAUSE_MARKERS):
        return "hybrid"

    if any(keyword in q_lower for keyword in RELATION_KEYWORDS):
        return "graph"

    return "vector"


if __name__ == "__main__":
    examples = [
        ("In what year was NimbusCloud founded?", "vector"),
        ("Who co-founded QuantumLeap Robotics alongside Marcus Bell?", "graph"),
        (
            "What product did the company founded by the person who previously "
            "worked at Verdant Energy Co go on to create?",
            "hybrid",
        ),
    ]
    for question, expected in examples:
        predicted = route(question)
        status = "OK" if predicted == expected else "MISMATCH"
        print(f"[{status}] expected={expected:7s} predicted={predicted:7s} | {question}")

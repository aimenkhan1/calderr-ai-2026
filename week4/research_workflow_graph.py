"""
Research Workflow Graph
query -> search -> draft -> feedback -> [good enough? respond : redraft]
Real Tavily web search + Groq LLM calls for drafting and quality judging.
Iterates on the draft until quality threshold is met (max 3 iterations).
"""

import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from groq import Groq
from tavily import TavilyClient

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MODEL = "llama-3.1-8b-instant"

MAX_ITERATIONS = 3
QUALITY_THRESHOLD = 0.7


class State(TypedDict):
    query: str
    search_results: Annotated[List[str], operator.add]
    draft: str
    feedback: str
    quality_score: float
    iteration: int
    final_report: str


# ── nodes ────────────────────────────────────────────────────
def search(state: State) -> dict:
    # Real live web search via Tavily
    response = tavily_client.search(
        query=state["query"],
        max_results=3,
        search_depth="basic",
    )

    result_texts = []
    for r in response["results"]:
        result_texts.append(f"- {r['title']}: {r['content'][:200]}...")

    combined = "\n".join(result_texts)
    return {"search_results": [f"[Search pass {state['iteration'] + 1}]\n{combined}"]}


def draft(state: State) -> dict:
    iteration = state["iteration"] + 1
    all_context = "\n\n".join(state["search_results"])
    prior_feedback = state["feedback"] if state["feedback"] else "None yet — this is the first draft."

    prompt = f"""Write a short research report answering this question, using the real search
context below. Incorporate the previous feedback if there is any.

Question: {state['query']}

Search context:
{all_context}

Previous feedback to address:
{prior_feedback}

Write a clear, well-structured 3-4 sentence report."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    new_draft = response.choices[0].message.content.strip()

    return {"draft": new_draft, "iteration": iteration}


def get_feedback(state: State) -> dict:
    prompt = f"""You are a strict research report reviewer. Rate this draft's quality
from 0.0 to 1.0 based on clarity, accuracy, and completeness relative to the question.

Question: {state['query']}

Draft:
{state['draft']}

Respond in EXACTLY this format, nothing else:
SCORE: <number between 0.0 and 1.0>
FEEDBACK: <one sentence of specific improvement advice>"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()

    score = 0.5
    feedback = "Could not parse feedback."
    for line in text.splitlines():
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return {"quality_score": score, "feedback": feedback}


def route_after_feedback(state: State) -> str:
    if state["quality_score"] >= QUALITY_THRESHOLD:
        return "good_enough"
    if state["iteration"] >= MAX_ITERATIONS:
        return "max_reached"
    return "needs_work"


def finalize_report(state: State) -> dict:
    return {"final_report": state["draft"]}


def finalize_best_effort(state: State) -> dict:
    return {"final_report": f"[Best effort after {state['iteration']} iterations] {state['draft']}"}


#builder and graph compilation
builder = StateGraph(State)

builder.add_node("search", search)
builder.add_node("draft", draft)
builder.add_node("get_feedback", get_feedback)
builder.add_node("finalize_report", finalize_report)
builder.add_node("finalize_best_effort", finalize_best_effort)

builder.add_edge(START, "search")
builder.add_edge("search", "draft")
builder.add_edge("draft", "get_feedback")

builder.add_conditional_edges(
    "get_feedback",
    route_after_feedback,
    {
        "good_enough": "finalize_report",
        "needs_work": "search",
        "max_reached": "finalize_best_effort",
    },
)

builder.add_edge("finalize_report", END)
builder.add_edge("finalize_best_effort", END)

graph = builder.compile()


#main
if __name__ == "__main__":
    query = input("Enter a research question: ").strip()

    result = graph.invoke({
        "query": query,
        "search_results": [],
        "draft": "",
        "feedback": "",
        "quality_score": 0.0,
        "iteration": 0,
        "final_report": "",
    })

    print(f"\nIterations taken: {result['iteration']}")
    print(f"Final quality score: {result['quality_score']}")
    print(f"Last feedback: {result['feedback']}")
    print(f"\nFinal Report\n{result['final_report']}")

    print("\nAll Search Passes")
    for s in result["search_results"]:
        print(s)
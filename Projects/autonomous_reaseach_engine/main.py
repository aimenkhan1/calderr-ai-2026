"""
Autonomous Research Workflow Engine
plan research -> parallel research (subgraph, 3 threads) -> synthesize -> gap analysis
-> [complete: generate report | gaps: plan additional research (loop)] -> publish

Real Tavily search + real Groq LLM calls throughout. Each of the 3 parallel research
threads runs its own compiled LangGraph subgraph (search -> summarize), executed
concurrently via ThreadPoolExecutor
"""

import os
from concurrent.futures import ThreadPoolExecutor
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

MAX_ITERATIONS = 2   


# SUBGRAPH: one research thread (search -> summarize),Compiled once, invoked 3x concurrently inside parallel_research()

class ThreadState(TypedDict):
    sub_question: str
    raw_results: str
    summary: str


def thread_search(state: ThreadState) -> dict:
    response = tavily_client.search(
        query=state["sub_question"], max_results=3, search_depth="basic"
    )
    combined = "\n".join(
        f"- {r['title']}: {r['content'][:200]}" for r in response["results"]
    )
    return {"raw_results": combined}


def thread_summarize(state: ThreadState) -> dict:
    prompt = f"""Summarize these search results in 2-3 sentences, focused specifically
on answering this question: {state['sub_question']}

Search results:
{state['raw_results']}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {"summary": response.choices[0].message.content.strip()}


thread_builder = StateGraph(ThreadState)
thread_builder.add_node("search", thread_search)
thread_builder.add_node("summarize", thread_summarize)
thread_builder.add_edge(START, "search")
thread_builder.add_edge("search", "summarize")
thread_builder.add_edge("summarize", END)
research_thread_graph = thread_builder.compile()


def run_research_thread(sub_question: str) -> str:
    result = research_thread_graph.invoke({
        "sub_question": sub_question, "raw_results": "", "summary": ""
    })
    return f"[{sub_question}]\n{result['summary']}"



# MAIN GRAPH: plan -> parallel research -> synthesize -> gap analysis -> loop/report

class State(TypedDict):
    topic: str
    research_plan: List[str]
    research_findings: Annotated[List[str], operator.add]   # accumulates across loop iterations
    synthesis: str
    gaps: List[str]
    gap_analysis_complete: bool
    iteration: int
    final_report: str


def plan_research(state: State) -> dict:
    iteration = state["iteration"] + 1

    if state.get("gaps"):
        gap_text = "\n".join(state["gaps"])
        prompt = f"""The following gaps were identified in research on: {state['topic']}
Gaps to address:
{gap_text}

List exactly 3 specific, distinct search queries that would help fill these gaps.
Reply with ONLY the 3 queries, one per line, no numbering."""
    else:
        prompt = f"""Break down this research topic into exactly 3 distinct, specific angles
to investigate in parallel (e.g. background/context, current data or statistics, expert
opinion or implications) rather than 3 near-duplicate questions.

Topic: {state['topic']}

Reply with ONLY the 3 questions, one per line, no numbering."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    lines = [l.lstrip("-•0123456789. ").strip()
             for l in response.choices[0].message.content.strip().splitlines() if l.strip()]
    plan = lines[:3] if len(lines) >= 3 else (lines + [state["topic"]] * (3 - len(lines)))

    return {"research_plan": plan, "iteration": iteration, "gaps": []}


def parallel_research(state: State) -> dict:
    with ThreadPoolExecutor(max_workers=3) as executor:
        findings = list(executor.map(run_research_thread, state["research_plan"]))
    return {"research_findings": findings}


def synthesize(state: State) -> dict:
    all_findings = "\n\n".join(state["research_findings"])
    prompt = f"""Synthesize the following research findings into a coherent, well-organized
draft answering the topic. Integrate all relevant points and avoid repetition.

Topic: {state['topic']}

Findings:
{all_findings}

Write 4-6 sentences."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return {"synthesis": response.choices[0].message.content.strip()}


def gap_analysis(state: State) -> dict:
    prompt = f"""Review this research synthesis for completeness regarding the topic below.

Topic: {state['topic']}

Synthesis:
{state['synthesis']}

If it comprehensively covers the topic, respond with EXACTLY: COMPLETE
Otherwise, respond with the word GAPS on the first line, followed by up to 3 specific
missing sub-questions, one per line, nothing else."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()

    if text.upper().startswith("COMPLETE"):
        return {"gap_analysis_complete": True, "gaps": []}

    lines = [l.strip() for l in text.splitlines()[1:] if l.strip()]
    return {"gap_analysis_complete": False, "gaps": lines[:3]}


def route_after_gap_analysis(state: State) -> str:
    if state["gap_analysis_complete"] or state["iteration"] >= MAX_ITERATIONS:
        return "complete"
    return "gaps"


def generate_report(state: State) -> dict:
    prompt = f"""Write a polished final research report on the topic below, based on the
synthesis. Include a short title, then organize the body into clear short paragraphs.

Topic: {state['topic']}

Synthesis:
{state['synthesis']}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return {"final_report": response.choices[0].message.content.strip()}


def publish(state: State) -> dict:
    footer = (f"\n\n---\n[Generated after {state['iteration']} research round(s), "
              f"{len(state['research_findings'])} findings gathered across parallel threads.]")
    return {"final_report": state["final_report"] + footer}


#graph building
builder = StateGraph(State)

builder.add_node("plan_research", plan_research)
builder.add_node("parallel_research", parallel_research)
builder.add_node("synthesize", synthesize)
builder.add_node("gap_analysis", gap_analysis)
builder.add_node("generate_report", generate_report)
builder.add_node("publish", publish)

builder.add_edge(START, "plan_research")
builder.add_edge("plan_research", "parallel_research")
builder.add_edge("parallel_research", "synthesize")
builder.add_edge("synthesize", "gap_analysis")

builder.add_conditional_edges(
    "gap_analysis",
    route_after_gap_analysis,
    {"complete": "generate_report", "gaps": "plan_research"},   
)

builder.add_edge("generate_report", "publish")
builder.add_edge("publish", END)

graph = builder.compile()


#CLI
if __name__ == "__main__":
    topic = input("Enter a research topic: ").strip()

    result = graph.invoke({
        "topic": topic,
        "research_plan": [],
        "research_findings": [],
        "synthesis": "",
        "gaps": [],
        "gap_analysis_complete": False,
        "iteration": 0,
        "final_report": "",
    })

    print(f"\nResearch rounds: {result['iteration']}")
    print(f"Total findings gathered: {len(result['research_findings'])}")
    print(f"\nFinal Report\n{result['final_report']}")

    print("\nAll Research Findings")
    for f in result["research_findings"]:
        print(f, "\n")
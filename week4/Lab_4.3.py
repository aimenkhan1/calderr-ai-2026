"""
Lab 4.3 — Content Moderation Graph
Posts flow through: auto-approve, borderline (human review), final decision.
State persists correctly across the interrupt, even if resumed later.
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
import random


#if score 0.0 so its safe and nearer to 1.0 its v toxic 
class State(TypedDict):
    post_text: str
    toxicity_score: float
    category: str              #it can autoapprove autoreject or somewhere boderline that will be human review 
    human_decision: str
    final_status: str


AUTO_APPROVE_THRESHOLD = 0.3  
AUTO_REJECT_THRESHOLD = 0.8    



def score_post(state: State) -> dict:
    score = round(random.uniform(0.0, 1.0), 2)
    return {"toxicity_score": score}


def categorize(state: State) -> dict:
    score = state["toxicity_score"]
    if score < AUTO_APPROVE_THRESHOLD:
        category = "auto_approve"
    elif score > AUTO_REJECT_THRESHOLD:
        category = "auto_reject"
    else:
        category = "borderline"
    return {"category": category}


def route_by_category(state: State) -> str:
    return state["category"]


def auto_approve(state: State) -> dict:
    return {"final_status": f" Auto-approved (score: {state['toxicity_score']})"}


def auto_reject(state: State) -> dict:
    return {"final_status": f" Auto-rejected (score: {state['toxicity_score']})"}


def human_review(state: State) -> dict:

    decision = interrupt({
        "post_text": state["post_text"],
        "toxicity_score": state["toxicity_score"],
        "instruction": "This post is borderline. Reply 'approve' or 'reject'.",
    })
    return {"human_decision": decision}


def route_after_human(state: State) -> str:
    return "approved" if state["human_decision"] == "approve" else "rejected"


def finalize_approved(state: State) -> dict:
    return {"final_status": f" Approved by human review (score: {state['toxicity_score']})"}


def finalize_rejected(state: State) -> dict:
    return {"final_status": f" Rejected by human review (score: {state['toxicity_score']})"}


builder = StateGraph(State)

builder.add_node("score_post", score_post)
builder.add_node("categorize", categorize)
builder.add_node("auto_approve", auto_approve)
builder.add_node("auto_reject", auto_reject)
builder.add_node("human_review", human_review)
builder.add_node("finalize_approved", finalize_approved)
builder.add_node("finalize_rejected", finalize_rejected)

builder.add_edge(START, "score_post")
builder.add_edge("score_post", "categorize")

builder.add_conditional_edges(
    "categorize",
    route_by_category,
    {
        "auto_approve": "auto_approve",
        "auto_reject": "auto_reject",
        "borderline": "human_review",
    },
)

builder.add_conditional_edges(
    "human_review",
    route_after_human,
    {"approved": "finalize_approved", "rejected": "finalize_rejected"},
)

builder.add_edge("auto_approve", END)
builder.add_edge("auto_reject", END)
builder.add_edge("finalize_approved", END)
builder.add_edge("finalize_rejected", END)

with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    post_text = input("Enter a post to moderate: ").strip()

    # Unique thread_id per post 
    config = {"configurable": {"thread_id": f"post-{hash(post_text) % 10000}"}}

    initial_state = {
        "post_text": post_text,
        "toxicity_score": 0.0,
        "category": "",
        "human_decision": "",
        "final_status": "",
    }

    result = graph.invoke(initial_state, config=config)

    if "__interrupt__" in result:
        # It paused — needs a human
        info = result["__interrupt__"][0].value
        print("\n--- NEEDS HUMAN REVIEW ---")
        print("Post:", info["post_text"])
        print("Toxicity score:", info["toxicity_score"])
        print(info["instruction"])

        decision = input("\nYour decision (approve/reject): ").strip().lower()
        result = graph.invoke(Command(resume=decision), config=config)

    print("\n--- FINAL STATUS ---")
    print(result["final_status"])
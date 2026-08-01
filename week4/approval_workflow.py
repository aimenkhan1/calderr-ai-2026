"""
Human-in-the-Loop: Simple Approval Workflow
Agent proposes an action -> human reviews -> approves/rejects -> agent adjusts
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    task: str
    proposed_action: str
    human_decision: str
    final_result: str


def propose_action(state: State) -> dict:
    action = f"Delete all records matching: '{state['task']}'"
    return {"proposed_action": action}


def human_review(state: State) -> dict:
    decision = interrupt({
        "question": "Approve this action?",
        "proposed_action": state["proposed_action"],
    })
    return {"human_decision": decision}


def route_after_review(state: State) -> str:
    return "approved" if state["human_decision"] == "approve" else "rejected"


def execute_action(state: State) -> dict:
    return {"final_result": f" Executed: {state['proposed_action']}"}


def cancel_action(state: State) -> dict:
    return {"final_result": f" Cancelled: {state['proposed_action']}"}


builder = StateGraph(State)
builder.add_node("propose_action", propose_action)
builder.add_node("human_review", human_review)
builder.add_node("execute_action", execute_action)
builder.add_node("cancel_action", cancel_action)

builder.add_edge(START, "propose_action")
builder.add_edge("propose_action", "human_review")
builder.add_conditional_edges(
    "human_review",
    route_after_review,
    {"approved": "execute_action", "rejected": "cancel_action"},
)
builder.add_edge("execute_action", END)
builder.add_edge("cancel_action", END)

graph = builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    task = input("Enter a task: ").strip()

    config = {"configurable": {"thread_id": "demo-thread-1"}}


    result = graph.invoke({"task": task, "proposed_action": "", "human_decision": "", "final_result": ""}, config=config)

    print("\n--- PAUSED FOR HUMAN REVIEW ---")
    print(result["__interrupt__"][0].value["question"])
    print("Proposed action:", result["__interrupt__"][0].value["proposed_action"])

    decision = input("\nType 'approve' or 'reject': ").strip().lower()

    final = graph.invoke(Command(resume=decision), config=config)

    print("\n--- FINAL RESULT ---")
    print(final["final_result"])
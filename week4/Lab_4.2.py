"""
Lab 4.2 — Self-Correcting Agent Loop
Generate -> Validate -> [pass: respond | fail: regenerate], max 3 attempts.
Logs how many iterations each input required.
"""

from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END
import random


#state
class State(TypedDict):
    prompt: str              
    draft: str                
    attempts: int              
    is_valid: bool              
    log: List[str]               
    final_response: str           


MAX_ATTEMPTS = 3

#node functions for the graph
def generate(state: State) -> dict:
    attempt_num = state["attempts"] + 1

    draft = f"Draft #{attempt_num} for: {state['prompt']}"

    log_entry = f"[Attempt {attempt_num}] Generated: {draft}"

    return {
        "draft": draft,
        "attempts": attempt_num,
        "log": state["log"] + [log_entry],   
    }


def validate(state: State) -> dict:
    is_valid = random.random() > 0.5

    result_text = "PASSED" if is_valid else "FAILED"
    log_entry = f"[Attempt {state['attempts']}] Validation: {result_text}"

    return {
        "is_valid": is_valid,
        "log": state["log"] + [log_entry],
    }


def route_after_validate(state: State) -> str:
    if state["is_valid"]:
        return "pass"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "give_up"
    return "fail"


def respond(state: State) -> dict:
    log_entry = f"Accepted after {state['attempts']} attempt(s)."
    return {
        "final_response": state["draft"],
        "log": state["log"] + [log_entry],
    }


def give_up(state: State) -> dict:
    log_entry = f"Gave up after {state['attempts']} attempts — max retries reached."
    return {
        "final_response": f"[FAILED] Best attempt: {state['draft']}",
        "log": state["log"] + [log_entry],
    }


#graph construction
builder = StateGraph(State)

builder.add_node("generate", generate)
builder.add_node("validate", validate)
builder.add_node("respond", respond)
builder.add_node("give_up", give_up)

builder.add_edge(START, "generate")
builder.add_edge("generate", "validate")

builder.add_conditional_edges(
    "validate",
    route_after_validate,
    {
        "pass": "respond",
        "fail": "generate",     
        "give_up": "give_up",
    },
)

builder.add_edge("respond", END)
builder.add_edge("give_up", END)

graph = builder.compile()


#main
if __name__ == "__main__":
    prompt = input("Enter a prompt: ").strip()

    result = graph.invoke({
        "prompt": prompt,
        "draft": "",
        "attempts": 0,
        "is_valid": False,
        "log": [],
        "final_response": "",
    })
    print()
    print("Iteration Log")
    for line in result["log"]:
        print(line)

    print()
    print("Final Result")
    print(result["final_response"])
    print(f"Total iterations required: {result['attempts']}")
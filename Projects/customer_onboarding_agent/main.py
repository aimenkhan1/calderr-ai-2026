"""
Customer Onboarding Agent
collect info -> validate -> [standard: auto approve | large: human review] -> create account -> notify -> schedule

Large accounts (by seats requested or monthly value) pause for human approval
via LangGraph's interrupt() mechanism. State is checkpointed to SQLite, so a
paused onboarding can be resumed at any time.
"""
q
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
load_dotenv(dotenv_path="../../.env")

LARGE_ACCOUNT_SEATS_THRESHOLD = 50
LARGE_ACCOUNT_VALUE_THRESHOLD = 5000.0   # monthly value in USD

#state
class State(TypedDict):
    applicant_name: str
    applicant_email: str
    company_name: str
    seats_requested: int
    monthly_value: float

    is_valid: bool
    validation_notes: str

    account_type: str          #standard or large
    human_decision: str        # approve" or "reject if large 

    account_id: str
    notification_sent: bool
    followup_date: str

    status: str                  
    log: Annotated[List[str], operator.add] 


#nodes
def collect_info(state: State) -> dict:
    entry = f"[{datetime.now().isoformat(timespec='seconds')}] Collected info for {state['applicant_name']} ({state['company_name']})"
    return {"log": [entry]}


def validate(state: State) -> dict:
    notes = []
    is_valid = True

    if "@" not in state["applicant_email"] or "." not in state["applicant_email"].split("@")[-1]:
        is_valid = False
        notes.append("Invalid email format.")

    if not state["applicant_name"].strip():
        is_valid = False
        notes.append("Missing applicant name.")

    if not state["company_name"].strip():
        is_valid = False
        notes.append("Missing company name.")

    if state["seats_requested"] <= 0:
        is_valid = False
        notes.append("Seats requested must be greater than 0.")

    validation_notes = " ".join(notes) if notes else "All fields valid."
    entry = f"Validation: {'PASSED' if is_valid else 'FAILED'} - {validation_notes}"

    return {"is_valid": is_valid, "validation_notes": validation_notes, "log": [entry]}


def route_after_validate(state: State) -> str:
    return "valid" if state["is_valid"] else "invalid"


def reject_invalid(state: State) -> dict:
    return {
        "status": f"Rejected at validation: {state['validation_notes']}",
        "log": [f"Onboarding rejected - failed validation."],
    }


def categorize_account(state: State) -> dict:
    is_large = (
        state["seats_requested"] > LARGE_ACCOUNT_SEATS_THRESHOLD
        or state["monthly_value"] > LARGE_ACCOUNT_VALUE_THRESHOLD
    )
    account_type = "large" if is_large else "standard"
    entry = f"Categorized as '{account_type}' account (seats={state['seats_requested']}, value=${state['monthly_value']})"
    return {"account_type": account_type, "log": [entry]}


def route_by_account_type(state: State) -> str:
    return state["account_type"]


def auto_approve(state: State) -> dict:
    entry = "Standard account - auto-approved, no human review needed."
    return {"human_decision": "approve", "log": [entry]}


#state is checkpoint to sqlite at this exact point .. and its paused 
def human_review(state: State) -> dict:
    decision = interrupt({
        "instruction": "Large account requires manual approval. Reply 'approve' or 'reject'.",
        "applicant_name": state["applicant_name"],
        "company_name": state["company_name"],
        "seats_requested": state["seats_requested"],
        "monthly_value": state["monthly_value"],
    })
    entry = f"Human review decision received: {decision}"
    return {"human_decision": decision, "log": [entry]}


def route_after_human(state: State) -> str:
    return "approved" if state["human_decision"] == "approve" else "rejected"


def reject_account(state: State) -> dict:
    entry = "Account rejected by human reviewer."
    return {"status": "Rejected during human review.", "log": [entry]}


def create_account(state: State) -> dict:
    account_id = f"ACC-{uuid.uuid4().hex[:8].upper()}"
    entry = f"Account created: {account_id}"
    return {"account_id": account_id, "log": [entry]}


def notify(state: State) -> dict:
    entry = f"Welcome email sent to {state['applicant_email']}."
    return {"notification_sent": True, "log": [entry]}


#schedule followup for now placeholder we can use calender api call
def schedule_followup(state: State) -> dict:
    followup_date = "7 days from account creation"
    entry = f"Follow-up scheduled: {followup_date}"
    return {
        "followup_date": followup_date,
        "status": f"Onboarding complete. Account: {state['account_id']}",
        "log": [entry],
    }


#graph built
builder = StateGraph(State)

builder.add_node("collect_info", collect_info)
builder.add_node("validate", validate)
builder.add_node("reject_invalid", reject_invalid)
builder.add_node("categorize_account", categorize_account)
builder.add_node("auto_approve", auto_approve)
builder.add_node("human_review", human_review)
builder.add_node("reject_account", reject_account)
builder.add_node("create_account", create_account)
builder.add_node("notify", notify)
builder.add_node("schedule_followup", schedule_followup)

builder.add_edge(START, "collect_info")
builder.add_edge("collect_info", "validate")

builder.add_conditional_edges(
    "validate",
    route_after_validate,
    {"valid": "categorize_account", "invalid": "reject_invalid"},
)

builder.add_conditional_edges(
    "categorize_account",
    route_by_account_type,
    {"standard": "auto_approve", "large": "human_review"},
)

builder.add_conditional_edges(
    "auto_approve",
    route_after_human,
    {"approved": "create_account", "rejected": "reject_account"},
)

builder.add_conditional_edges(
    "human_review",
    route_after_human,
    {"approved": "create_account", "rejected": "reject_account"},
)

builder.add_edge("create_account", "notify")
builder.add_edge("notify", "schedule_followup")
builder.add_edge("schedule_followup", END)
builder.add_edge("reject_invalid", END)
builder.add_edge("reject_account", END)


def build_graph(checkpointer):
    return builder.compile(checkpointer=checkpointer)


#CLI
def run_new_application(graph):
    print("\nNew Applicant")
    name = input("Applicant name: ").strip()
    email = input("Applicant email: ").strip()
    company = input("Company name: ").strip()
    seats = int(input("Seats requested: ").strip() or "0")
    value = float(input("Monthly value ($): ").strip() or "0")

    thread_id = f"onboarding-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "applicant_name": name,
        "applicant_email": email,
        "company_name": company,
        "seats_requested": seats,
        "monthly_value": value,
        "is_valid": False,
        "validation_notes": "",
        "account_type": "",
        "human_decision": "",
        "account_id": "",
        "notification_sent": False,
        "followup_date": "",
        "status": "",
        "log": [],
    }

    result = graph.invoke(initial_state, config=config)

    if "__interrupt__" in result:
        info = result["__interrupt__"][0].value
        print(f"\n PAUSED - saved under thread_id: '{thread_id}'")
        print("You can close this program and resume later with this thread_id.")
        print(f"\n{info['instruction']}")
        print(f"Applicant: {info['applicant_name']} @ {info['company_name']}")
        print(f"Seats: {info['seats_requested']}, Monthly value: ${info['monthly_value']}")
    else:
        print_result(result)


def run_resume(graph):
    thread_id = input("Enter thread_id to resume: ").strip()
    config = {"configurable": {"thread_id": thread_id}}
    decision = input("Decision ('approve' or 'reject'): ").strip().lower()

    result = graph.invoke(Command(resume=decision), config=config)
    print_result(result)


def print_result(result):
    print("\nResult")
    print("Status:", result.get("status"))
    print("\nAudit Log")
    for entry in result.get("log", []):
        print(" -", entry)


if __name__ == "__main__":
    with SqliteSaver.from_conn_string("onboarding_checkpoints.sqlite") as checkpointer:
        graph = build_graph(checkpointer)

        mode = input("Type 'new' for a new applicant, or 'resume' to continue a paused one: ").strip().lower()
        if mode == "new":
            run_new_application(graph)
        elif mode == "resume":
            run_resume(graph)
        else:
            print("Unknown mode. Use 'new' or 'resume'.")
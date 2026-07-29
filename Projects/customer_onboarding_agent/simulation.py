"""
Demo: Batch Onboarding Simulation
Runs every applicant in sample_applicants.json through the onboarding graph.
Standard accounts auto-approve. Large accounts pause for human review --
this script simulates the human decision automatically so the whole batch
can run end-to-end without manual input, demonstrating both paths.
"""

import json
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from main import build_graph

def simulate_human_decision(applicant: dict) -> str:

    value_per_seat = applicant["monthly_value"] / max(applicant["seats_requested"], 1)
    if value_per_seat > 200:
        return "reject"
    return "approve"


def run_one(graph, applicant: dict, index: int):
    thread_id = f"demo-{index}-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "applicant_name": applicant["applicant_name"],
        "applicant_email": applicant["applicant_email"],
        "company_name": applicant["company_name"],
        "seats_requested": applicant["seats_requested"],
        "monthly_value": applicant["monthly_value"],
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
        decision = simulate_human_decision(applicant)
        print(f"  -> paused for human review, simulated decision: '{decision}'")
        result = graph.invoke(Command(resume=decision), config=config)

    return result


def main():
    with open("sample_applicants.json") as f:
        applicants = json.load(f)

    checkpointer = MemorySaver()  
    graph = build_graph(checkpointer)

    print(f"Running {len(applicants)} applicants through the onboarding graph...\n")

    results = []
    for i, applicant in enumerate(applicants, start=1):
        print(f"[{i}] {applicant['applicant_name']} @ {applicant['company_name']} "
              f"(seats={applicant['seats_requested']}, value=${applicant['monthly_value']})")
        result = run_one(graph, applicant, i)
        print(f"  -> {result['status']}\n")
        results.append({
            "applicant": applicant["applicant_name"],
            "account_type": result.get("account_type"),
            "status": result.get("status"),
        })

    print()
    print("SUMMARY")
    print()
    for r in results:
        print(f"{r['applicant']:20s} | {r['account_type'] or 'n/a':10s} | {r['status']}")


if __name__ == "__main__":
    main()
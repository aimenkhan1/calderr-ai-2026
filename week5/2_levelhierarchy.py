"""
2-LEVEL HIERARCHY
==============================
Build: Executive Supervisor -> [Research Lead, Engineering Lead] -> 4 Worker Agents

Each level should only ever see what it strictly needs - nothing more.
  - A Worker should NEVER see the Executive's original goal, or what
    the OTHER worker on its team is doing.
  - A Team Lead should NEVER pass raw Worker chatter up to the
    Executive - it must first condense its team's results into ONE
    summary. The Executive should never see 4 individual worker
    outputs, only 2 team-level summaries.
This mirrors a real company org chart: the CEO doesn't read every
engineer's Slack messages - they get a summary from the Engineering
Lead. An individual engineer doesn't see the CEO's full strategy doc -
they get a specific ticket assigned to them.

Hierarchy built here:
    Executive Supervisor
      |-- Research Lead
      |     |-- Worker: Researcher1
      |     |-- Worker: Researcher2
      |-- Engineering Lead
            |-- Worker: Engineer1
            |-- Worker: Engineer2
"""

from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field
import random


#typed messages-one per boundary

#executive->team lead
class Directive(BaseModel):

    sender: str          
    recipient: str      
    team: str
    objective: str       

#team lead->worker
class WorkerAssignment(BaseModel):

    sender: str          
    recipient: str    
    task_id: str
    instructions: str     

#worker->lead
class WorkerResult(BaseModel):
    sender: str
    recipient: str      
    task_id: str
    output: str
    confidence: float = Field(ge=0.0, le=1.0)

#lead->executive-report
class TeamSummary(BaseModel):

    sender: str         
    recipient: str    
    team: str
    summary: str           
    worker_count: int
    average_confidence: float = Field(ge=0.0, le=1.0)


#final report made by executive with two combined
class ExecutiveReport(BaseModel):
    goal: str
    team_summaries: List[TeamSummary]
    overall_confidence: float = Field(ge=0.0, le=1.0)


#worker 

class Worker:
    def __init__(self, name: str):
        self.name = name
        self.inbox: List[BaseModel] = []

    def receive_assignment(self, assignment: WorkerAssignment) -> WorkerResult:
        self.inbox.append(assignment)  
        output = f"[{self.name}] Completed: {assignment.instructions}"
        confidence = round(random.uniform(0.65, 0.95), 2)

        return WorkerResult(
            sender=self.name,
            recipient=assignment.sender,  
            task_id=assignment.task_id,
            output=output,
            confidence=confidence,
        )


#team lead

class TeamLead:
    def __init__(self, name: str, worker_a: Worker, worker_b: Worker):
        self.name = name
        self.workers = [worker_a, worker_b]
        self.inbox: List[BaseModel] = []

    def receive_directive(self, directive: Directive) -> TeamSummary:
        self.inbox.append(directive)   

        worker_results: List[WorkerResult] = []
        for i, worker in enumerate(self.workers, start=1):
            assignment = WorkerAssignment(
                sender=self.name,
                recipient=worker.name,
                task_id=f"{directive.team}-{i}",
                instructions=f"Handle part {i} of: {directive.objective}",
            )
            result = worker.receive_assignment(assignment)
            worker_results.append(result)


        avg_confidence = round(
            sum(r.confidence for r in worker_results) / len(worker_results), 2
        )
        combined_summary = (
            f"{self.name} completed {len(worker_results)} sub-tasks for "
            f"'{directive.objective}' with average confidence {avg_confidence}."
        )

        return TeamSummary(
            sender=self.name,
            recipient="Executive",
            team=directive.team,
            summary=combined_summary,
            worker_count=len(worker_results),
            average_confidence=avg_confidence,
        )


#executive-top hierarchy

class ExecutiveSupervisor:
    def __init__(self, research_lead: TeamLead, engineering_lead: TeamLead):
        self.leads = [research_lead, engineering_lead]
        self.inbox: List[BaseModel] = []

    def run(self, goal: str) -> ExecutiveReport:
        team_summaries: List[TeamSummary] = []

        for lead in self.leads:
            directive = Directive(
                sender="Executive",
                recipient=lead.name,
                team=lead.name,
                objective=f"{goal} (from the {lead.name}'s perspective)",
            )
            summary = lead.receive_directive(directive)
            self.inbox.append(summary)   # log what the Executive actually received
            team_summaries.append(summary)

        overall_confidence = round(
            sum(s.average_confidence for s in team_summaries) / len(team_summaries), 2
        )

        return ExecutiveReport(
            goal=goal,
            team_summaries=team_summaries,
            overall_confidence=overall_confidence,
        )


#main

def build_hierarchy() -> ExecutiveSupervisor:
    researcher1 = Worker("Researcher1")
    researcher2 = Worker("Researcher2")
    engineer1 = Worker("Engineer1")
    engineer2 = Worker("Engineer2")

    research_lead = TeamLead("ResearchLead", researcher1, researcher2)
    engineering_lead = TeamLead("EngineeringLead", engineer1, engineer2)

    executive = ExecutiveSupervisor(research_lead, engineering_lead)
    return executive, research_lead, engineering_lead, [researcher1, researcher2, engineer1, engineer2]


def run_demo():
    executive, research_lead, engineering_lead, workers = build_hierarchy()

    print("--- Running hierarchy for goal: 'Launch new product line' ---\n")
    report = executive.run("Launch new product line")

    print("EXECUTIVE REPORT")
    print("=" * 60)
    print(f"Goal: {report.goal}")
    for summary in report.team_summaries:
        print(f"\n  Team: {summary.team}")
        print(f"  Summary: {summary.summary}")
        print(f"  Workers involved: {summary.worker_count}, avg confidence: {summary.average_confidence}")
    print(f"\nOverall confidence: {report.overall_confidence}")

    return executive, research_lead, engineering_lead, workers


#

def verify_no_context_leaked(executive: ExecutiveSupervisor, research_lead: TeamLead,
                               engineering_lead: TeamLead, workers: list):
    print("\n" + "=" * 60)
    print("CONTEXT-LEAK VERIFICATION")
    print("=" * 60)

    # CHECK 1: The Executive should ONLY have ever received TeamSummary
    # objects - never a raw WorkerResult or WorkerAssignment.
    exec_message_types = {type(m).__name__ for m in executive.inbox}
    assert exec_message_types == {"TeamSummary"}, (
        f"LEAK DETECTED: Executive received unexpected message types: {exec_message_types}"
    )
    print(f"✓ Executive only ever received: {exec_message_types} "
          f"(never saw raw WorkerResult data)")

    # CHECK 2: Each Team Lead should have only received Directives
    # meant for ITSELF - never the other team's Directive.
    for lead in [research_lead, engineering_lead]:
        recipients = {m.recipient for m in lead.inbox}
        assert recipients == {lead.name}, (
            f"LEAK DETECTED: {lead.name} received a message meant for someone else: {recipients}"
        )
    print(f"✓ Each Team Lead only received Directives addressed to itself")

    # CHECK 3: Each Worker should have only received ITS OWN
    # WorkerAssignment - never a sibling worker's assignment, and
    # never the original Directive or the Executive's goal text.
    for worker in workers:
        recipients = {m.recipient for m in worker.inbox}
        message_types = {type(m).__name__ for m in worker.inbox}
        assert recipients == {worker.name}, (
            f"LEAK DETECTED: {worker.name} received a message meant for someone else: {recipients}"
        )
        assert message_types == {"WorkerAssignment"}, (
            f"LEAK DETECTED: {worker.name} received an unexpected message type: {message_types}"
        )
    print(f"✓ Each Worker only received its own WorkerAssignment "
          f"(never the Executive's goal, never a sibling's task)")

    print("\nAll checks passed - context is properly scoped at every level.")


if __name__ == "__main__":
    executive, research_lead, engineering_lead, workers = run_demo()
    verify_no_context_leaked(executive, research_lead, engineering_lead, workers)
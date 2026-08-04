"""
LAB 5.2 - SUPERVISOR WITH FAILURE RECOVERY
============================================
Requirement:
  "Build a Supervisor Agent that delegates to 3 specialist agents.
   Inject failures into 2 specialists (random timeout, low-confidence
   response). The supervisor must: detect the failure type, log its
   reasoning for re-routing, try an alternative agent, and - if all
   alternatives fail - produce a gracefully degraded response. The
   system must never crash."

The 3 specialists, and their injected failure modes:
  - SpecialistA : RELIABLE. Always works. This is the "control" agent.
  - SpecialistB : TIMEOUT failure. ~50% of the time it simulates
                  taking too long and the call is abandoned.
  - SpecialistC : LOW-CONFIDENCE failure. It always responds
                  (never times out), but its confidence score is
                  usually below the quality bar we require.

The core idea this lab is testing: a Supervisor pattern is NOT just
"call an agent and hope." It's "call an agent, actively check if
the result is good enough, and if not, KNOW WHY, and try someone
else" - all without ever crashing the whole program.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import random
import time


# ============================================================
# PART 1 - TYPED MESSAGES
# ============================================================

class TaskRequest(BaseModel):
    task_id: str
    description: str


class SpecialistResponse(BaseModel):

    agent_name: str
    task_id: str
    output: str
    confidence: float = Field(ge=0.0, le=1.0)


class FailureType(str, Enum):
    TIMEOUT = "timeout"
    LOW_CONFIDENCE = "low_confidence"
    NONE = "none"   # used for successful attempts in the log


class DecisionLogEntry(BaseModel):
  
    task_id: str
    agent_tried: str
    failure_type: FailureType
    reasoning: str
    confidence: Optional[float] = None


class FinalResult(BaseModel):

    task_id: str
    output: str
    confidence: float = Field(ge=0.0, le=1.0)
    was_degraded: bool          # True if NO specialist actually succeeded
    agents_tried: List[str]
    decision_log: List[DecisionLogEntry]




CONFIDENCE_THRESHOLD = 0.6  #below this ot good enough


class SpecialistA:   #reliable
    name = "SpecialistA"

    def process(self, request: TaskRequest) -> SpecialistResponse:
        time.sleep(0.05)
        return SpecialistResponse(
            agent_name=self.name,
            task_id=request.task_id,
            output=f"[{self.name}] Solid answer for: {request.description}",
            confidence=round(random.uniform(0.75, 0.95), 2),
        )


class SpecialistB:   #timeout 

    name = "SpecialistB"

    def process(self, request: TaskRequest) -> SpecialistResponse:
        if random.random() < 0.5:
            # simulate a hung call that never comes back in time
            raise TimeoutError(f"{self.name} did not respond in time.")
        time.sleep(0.05)
        return SpecialistResponse(
            agent_name=self.name,
            task_id=request.task_id,
            output=f"[{self.name}] Answer for: {request.description}",
            confidence=round(random.uniform(0.7, 0.9), 2),
        )


class SpecialistC:   #low confidence

    name = "SpecialistC"

    def process(self, request: TaskRequest) -> SpecialistResponse:
        time.sleep(0.05)
        return SpecialistResponse(
            agent_name=self.name,
            task_id=request.task_id,
            output=f"[{self.name}] Uncertain answer for: {request.description}",
            confidence=round(random.uniform(0.25, 0.55), 2),   # almost always below threshold
        )



class SupervisorAgent:
    name = "Supervisor"

    def __init__(self, specialists: List):
        self.specialists = specialists   # the fallback order to try, in sequence

    def handle_task(self, request: TaskRequest) -> FinalResult:
        decision_log: List[DecisionLogEntry] = []
        agents_tried: List[str] = []

        for specialist in self.specialists:
            agents_tried.append(specialist.name)

            # --- Attempt 1: try to catch a TIMEOUT ---
            try:
                response = specialist.process(request)
            except TimeoutError as e:
                reasoning = (
                    f"{specialist.name} timed out ({e}). "
                    f"Rerouting to the next available specialist."
                )
                print(f"  [Supervisor] {reasoning}")
                decision_log.append(DecisionLogEntry(
                    task_id=request.task_id,
                    agent_tried=specialist.name,
                    failure_type=FailureType.TIMEOUT,
                    reasoning=reasoning,
                ))
                continue   # try the next specialist in the list

            # --- Attempt 2: check for LOW CONFIDENCE ---
            if response.confidence < CONFIDENCE_THRESHOLD:
                reasoning = (
                    f"{specialist.name} responded but confidence "
                    f"{response.confidence} is below the {CONFIDENCE_THRESHOLD} threshold. "
                    f"Rerouting to the next available specialist."
                )
                print(f"  [Supervisor] {reasoning}")
                decision_log.append(DecisionLogEntry(
                    task_id=request.task_id,
                    agent_tried=specialist.name,
                    failure_type=FailureType.LOW_CONFIDENCE,
                    reasoning=reasoning,
                    confidence=response.confidence,
                ))
                continue

            # --- SUCCESS: good enough response found ---
            reasoning = (
                f"{specialist.name} succeeded with confidence {response.confidence} "
                f"(meets the {CONFIDENCE_THRESHOLD} threshold)."
            )
            print(f"  [Supervisor] {reasoning}")
            decision_log.append(DecisionLogEntry(
                task_id=request.task_id,
                agent_tried=specialist.name,
                failure_type=FailureType.NONE,
                reasoning=reasoning,
                confidence=response.confidence,
            ))

            return FinalResult(
                task_id=request.task_id,
                output=response.output,
                confidence=response.confidence,
                was_degraded=False,
                agents_tried=agents_tried,
                decision_log=decision_log,
            )

        # --- EVERY specialist failed: graceful degradation, NOT a crash ---
        reasoning = (
            f"All {len(self.specialists)} specialists failed or returned "
            f"low-confidence results. Producing a degraded fallback response "
            f"instead of raising an error."
        )
        print(f"  [Supervisor] {reasoning}")

        return FinalResult(
            task_id=request.task_id,
            output=(
                f"Unable to produce a high-confidence answer for "
                f"'{request.description}'. All specialists were unavailable "
                f"or uncertain - this is a best-effort placeholder response."
            ),
            confidence=0.0,
            was_degraded=True,
            agents_tried=agents_tried,
            decision_log=decision_log,
        )



def run_demo():
    # Build all 3 specialists once
    a, b, c = SpecialistA(), SpecialistB(), SpecialistC()


    task_specialist_orders = {
        "T-1": [b, c, a],   # starts with the timeout-prone specialist
        "T-2": [c, a, b],   # starts with the low-confidence specialist
        "T-3": [a, b, c],   # starts with the reliable specialist
    }

    tasks = [
        TaskRequest(task_id="T-1", description="Summarize Q3 sales trends"),
        TaskRequest(task_id="T-2", description="Classify this support ticket"),
        TaskRequest(task_id="T-3", description="Estimate delivery time"),
    ]

    all_results: List[FinalResult] = []

    for task in tasks:
        print(f"\n--- Task {task.task_id}: {task.description} ---")
        supervisor = SupervisorAgent(specialists=task_specialist_orders[task.task_id])
        result = supervisor.handle_task(task)
        all_results.append(result)

        status = "DEGRADED (all specialists failed)" if result.was_degraded else "SUCCESS"
        print(f"  [Supervisor] FINAL RESULT ({status}): {result.output} "
              f"(confidence {result.confidence})")

    print()
    print("SUMMARY ACROSS ALL TASKS")
    print()
    for r in all_results:
        print(f"{r.task_id}: tried {r.agents_tried} -> "
              f"{'DEGRADED' if r.was_degraded else 'success'} "
              f"(confidence {r.confidence})")

    return all_results


def force_full_failure_demo():

    print("\n--- Forced full-failure scenario (all 3 specialists guaranteed to fail) ---")

    class AlwaysTimesOut:
        name = "AlwaysTimesOut"
        def process(self, request):
            raise TimeoutError("Simulated guaranteed timeout.")

    class AlwaysLowConfidence:
        name = "AlwaysLowConfidence"
        def process(self, request):
            return SpecialistResponse(
                agent_name=self.name, task_id=request.task_id,
                output="Low quality guess.", confidence=0.1,
            )

    class AlsoAlwaysLowConfidence:
        name = "AlsoAlwaysLowConfidence"
        def process(self, request):
            return SpecialistResponse(
                agent_name=self.name, task_id=request.task_id,
                output="Another low quality guess.", confidence=0.2,
            )

    supervisor = SupervisorAgent(specialists=[AlwaysTimesOut(), AlwaysLowConfidence(), AlsoAlwaysLowConfidence()])
    request = TaskRequest(task_id="T-FORCED", description="A task nobody can handle well")

    result = supervisor.handle_task(request)

    print(f"\n  Program did NOT crash. Final result was_degraded={result.was_degraded}")
    print(f"  Fallback message: {result.output}")


if __name__ == "__main__":
    run_demo()
    force_full_failure_demo()
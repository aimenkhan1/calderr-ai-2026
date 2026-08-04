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




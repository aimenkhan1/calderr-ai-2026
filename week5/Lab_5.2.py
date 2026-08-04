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




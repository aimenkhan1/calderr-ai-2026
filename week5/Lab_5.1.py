"""
LAB 5.1 - TYPED MESSAGE BUS
============================
Requirement:
  "Build a message-passing backbone for multi-agent systems. Define 4
   typed Pydantic message schemas (TaskRequest, TaskResult, ErrorReport,
   Handoff). Implement a simple in-memory message bus. Build 3 agents
   that communicate exclusively through typed messages - no raw
   strings. Verify that a malformed message raises a validation error
   before any agent receives it."

The big idea: agents never call each other's functions directly and
never pass plain strings/dicts around. Every single thing that moves
between agents is ONE of the 4 typed message classes below. The
"bus" is just a small in-memory post office: agents drop typed
messages into it, and it delivers them to whoever the message says
it's addressed to.

The 3 agents built here:
  - Coordinator : hands out work and makes the final call
  - Worker      : actually does the task (can succeed or fail)
  - Validator   : double-checks a Worker's result before it's final

Flow for one task:
  Coordinator --TaskRequest--> Worker
  Worker      --TaskResult or ErrorReport--> Coordinator
  Coordinator --Handoff--> Validator   (only if Worker succeeded)
  Validator   --TaskResult--> Coordinator   (final, validated result)
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ValidationError
import random


# ============================================================
# PART 1 - THE 4 TYPED MESSAGE SCHEMAS
#    These are the ONLY things allowed to travel on the bus.
#    Every one of them has a `sender` and `recipient` so the
#    bus knows who sent it and who should get it.
# ============================================================

class TaskRequest(BaseModel):
    """
    Sent when one agent wants another agent to do something.
    Example: Coordinator asks Worker to process a task.
    """
    sender: str
    recipient: str
    task_id: str
    description: str
    priority: int = Field(ge=1, le=5)   # 1 = low, 5 = urgent. Must be 1-5, nothing else.


class TaskResult(BaseModel):
    """
    Sent back when a task was completed successfully.
    """
    sender: str
    recipient: str
    task_id: str
    output: str
    confidence: float = Field(ge=0.0, le=1.0)   # must be between 0 and 1


class ErrorType(str, Enum):
    """A fixed list of error categories - like a dropdown, not free text."""
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    TOOL_FAILURE = "tool_failure"
    UNKNOWN = "unknown"


class ErrorReport(BaseModel):
    """
    Sent instead of a TaskResult when something went wrong.
    This is how an agent says "I couldn't do it" in a structured way,
    instead of just crashing or returning a vague error string.
    """
    sender: str
    recipient: str
    task_id: str
    error_type: ErrorType
    error_message: str
    is_recoverable: bool   # can this be retried, or is it a dead end?


class Handoff(BaseModel):
    """
    Sent when one agent passes ongoing work to another agent -
    NOT a brand new task, but "here, you take it from here."
    Example: Coordinator hands a completed-but-unverified result
    to the Validator to get double-checked.
    """
    sender: str
    recipient: str
    task_id: str
    reason: str
    payload: dict   # whatever context the next agent needs to continue


# ============================================================
# PART 2 - THE MESSAGE BUS
#    A simple in-memory "post office". Agents register with it
#    under a name. Sending a message means: look up the
#    recipient by name, and call their `.receive()` method.
# ============================================================

# The bus will only ever accept these 4 types - anything else is refused.
ALLOWED_MESSAGE_TYPES = (TaskRequest, TaskResult, ErrorReport, Handoff)


class MessageBus:
    def __init__(self):
        self._agents: dict[str, "Agent"] = {}   # name -> agent object
        self.message_log: list[BaseModel] = []   # keeps every message ever sent, for the audit trail

    def register(self, agent: "Agent") -> None:
        """An agent joins the bus under its own name."""
        self._agents[agent.name] = agent
        agent.bus = self   # so the agent can call self.bus.send(...) later

    def send(self, message: BaseModel) -> None:
        """
        THE CORE RULE OF THIS LAB:
        only one of the 4 typed messages may ever be sent.
        Anything else (a raw string, a dict, a random object) is
        rejected right here, before it ever reaches an agent.
        """
        if not isinstance(message, ALLOWED_MESSAGE_TYPES):
            raise TypeError(
                f"MessageBus only accepts typed messages "
                f"({[t.__name__ for t in ALLOWED_MESSAGE_TYPES]}), "
                f"got {type(message).__name__} instead."
            )

        recipient = self._agents.get(message.recipient)
        if recipient is None:
            raise ValueError(f"No agent named '{message.recipient}' is registered on the bus.")

        self.message_log.append(message)
        print(f"  [bus] {message.sender} -> {message.recipient}  ({type(message).__name__})")
        recipient.receive(message)


# ============================================================
# PART 3 - THE 3 AGENTS
#    Every agent only talks through bus.send(...) and only ever
#    receives one of the 4 typed messages in .receive().
# ============================================================

class Agent:
    """Small shared base so every agent has a name + a bus reference."""
    def __init__(self, name: str):
        self.name = name
        self.bus: Optional[MessageBus] = None

    def receive(self, message: BaseModel) -> None:
        raise NotImplementedError


class Coordinator(Agent):
    """
    Hands out work, and makes the final call once a Worker's
    result comes back validated (or reports failure if it doesn't).
    """
    def __init__(self):
        super().__init__("Coordinator")
        self.final_results: dict[str, BaseModel] = {}

    def start_task(self, task_id: str, description: str, priority: int = 3) -> None:
        request = TaskRequest(
            sender=self.name,
            recipient="Worker",
            task_id=task_id,
            description=description,
            priority=priority,
        )
        self.bus.send(request)

    def receive(self, message: BaseModel) -> None:
        if isinstance(message, TaskResult):
            # Worker succeeded -> hand off to Validator for a final check
            handoff = Handoff(
                sender=self.name,
                recipient="Validator",
                task_id=message.task_id,
                reason="Worker completed the task - please verify before closing.",
                payload={"output": message.output, "confidence": message.confidence},
            )
            self.bus.send(handoff)

        elif isinstance(message, ErrorReport):
            # Worker failed -> log it as the final outcome, no validator needed
            print(f"  [Coordinator] Task {message.task_id} failed: "
                  f"{message.error_type.value} - {message.error_message} "
                  f"(recoverable: {message.is_recoverable})")
            self.final_results[message.task_id] = message

        elif isinstance(message, TaskResult) is False and hasattr(message, "task_id"):
            # (Validator's final TaskResult also lands here - handled below)
            pass

    def receive_final(self, message: TaskResult) -> None:
        """Called when the Validator sends back the final, checked result."""
        print(f"  [Coordinator] Task {message.task_id} CONFIRMED: "
              f"'{message.output}' (confidence {message.confidence})")
        self.final_results[message.task_id] = message


class Worker(Agent):
    """
    Does the actual task. Randomly fails sometimes, on purpose, to
    prove the ErrorReport path works and nothing crashes.
    """
    def __init__(self):
        super().__init__("Worker")

    def receive(self, message: BaseModel) -> None:
        if isinstance(message, TaskRequest):
            if random.random() < 0.25:   # 25% pretend-failure rate
                error = ErrorReport(
                    sender=self.name,
                    recipient=message.sender,
                    task_id=message.task_id,
                    error_type=random.choice(list(ErrorType)),
                    error_message="Simulated failure while processing the task.",
                    is_recoverable=random.choice([True, False]),
                )
                self.bus.send(error)
            else:
                result = TaskResult(
                    sender=self.name,
                    recipient=message.sender,
                    task_id=message.task_id,
                    output=f"Completed: {message.description}",
                    confidence=round(random.uniform(0.6, 0.95), 2),
                )
                self.bus.send(result)


class Validator(Agent):
    """
    Double-checks a Worker's result before the Coordinator treats
    it as final. Always approves in this simple lab, but the
    structure is here for you to add real validation logic later.
    """
    def __init__(self):
        super().__init__("Validator")

    def receive(self, message: BaseModel) -> None:
        if isinstance(message, Handoff):
            # pretend to verify the payload, then send the FINAL TaskResult back
            final = TaskResult(
                sender=self.name,
                recipient="Coordinator",
                task_id=message.task_id,
                output=message.payload["output"],
                confidence=message.payload["confidence"],
            )
            # NOTE: this goes through the normal bus.send(), but the
            # Coordinator needs a way to tell "first result from Worker"
            # apart from "final result from Validator" - see receive_final
            # wiring below in run_demo().
            self.bus.send(final)


# ============================================================
# PART 4 - WIRING IT ALL TOGETHER
# ============================================================

def run_demo():
    bus = MessageBus()
    coordinator = Coordinator()
    worker = Worker()
    validator = Validator()

    bus.register(coordinator)
    bus.register(worker)
    bus.register(validator)

    # Small patch: route messages FROM Validator to Coordinator's
    # receive_final() instead of its normal receive(), so we can
    # tell "raw Worker result" apart from "validated final result".
    original_receive = coordinator.receive
    def coordinator_receive(message):
        if isinstance(message, TaskResult) and message.sender == "Validator":
            coordinator.receive_final(message)
        else:
            original_receive(message)
    coordinator.receive = coordinator_receive

    print("--- Running 3 tasks through Coordinator -> Worker -> Validator ---\n")
    for i in range(1, 4):
        print(f"Task {i}:")
        coordinator.start_task(task_id=f"T-{i}", description=f"Sample task #{i}", priority=3)
        print()

    print(f"--- Message bus handled {len(bus.message_log)} typed messages total ---")


def demonstrate_validation_error():
    """
    Proves the typing is actually enforced. We deliberately build a
    broken message and show Pydantic refusing it BEFORE it could ever
    reach bus.send() or any agent.
    """
    print("\n--- Validation check #1: TaskRequest with bad priority ---")
    try:
        TaskRequest(
            sender="Coordinator",
            recipient="Worker",
            task_id="T-BAD",
            description="This should fail",
            priority=9,   # invalid - must be between 1 and 5
        )
    except ValidationError as e:
        print("Correctly rejected before reaching the bus:")
        print(e)

    print("\n--- Validation check #2: TaskResult with bad confidence ---")
    try:
        TaskResult(
            sender="Worker",
            recipient="Coordinator",
            task_id="T-BAD",
            output="Some output",
            confidence=1.7,   # invalid - must be between 0.0 and 1.0
        )
    except ValidationError as e:
        print("Correctly rejected before reaching the bus:")
        print(e)

    print("\n--- Validation check #3: bus refuses a non-typed message ---")
    bus = MessageBus()
    coordinator = Coordinator()
    bus.register(coordinator)
    try:
        bus.send({"sender": "X", "recipient": "Coordinator", "task_id": "T-1"})  # a raw dict, not a typed message
    except TypeError as e:
        print("Correctly rejected by the bus itself:")
        print(e)


if __name__ == "__main__":
    run_demo()
    demonstrate_validation_error()
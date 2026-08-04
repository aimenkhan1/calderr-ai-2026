"""
LAB 5.1 - TYPED MESSAGE BUS
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



class TaskRequest(BaseModel):

    sender: str
    recipient: str
    task_id: str
    description: str
    priority: int = Field(ge=1, le=5)   # 1 = low, 5 = urgent. Must be 1-5,


class TaskResult(BaseModel):

    sender: str
    recipient: str
    task_id: str
    output: str
    confidence: float = Field(ge=0.0, le=1.0)   # must be between 0 and 1


class ErrorType(str, Enum):
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    TOOL_FAILURE = "tool_failure"
    UNKNOWN = "unknown"


class ErrorReport(BaseModel):

    sender: str
    recipient: str
    task_id: str
    error_type: ErrorType
    error_message: str
    is_recoverable: bool   # can this be retried, or is it a dead end?


class Handoff(BaseModel):

    sender: str
    recipient: str
    task_id: str
    reason: str
    payload: dict   # whatever context the next agent needs to continue



ALLOWED_MESSAGE_TYPES = (TaskRequest, TaskResult, ErrorReport, Handoff)


class MessageBus:
    def __init__(self):
        self._agents: dict[str, "Agent"] = {}   
        self.message_log: list[BaseModel] = [] 

    def register(self, agent: "Agent") -> None:
        """An agent joins the bus under its own name."""
        self._agents[agent.name] = agent
        agent.bus = self   # so the agent can call self.bus.send(...) later

    def send(self, message: BaseModel) -> None:

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

#

class Agent:
    def __init__(self, name: str):
        self.name = name
        self.bus: Optional[MessageBus] = None

    def receive(self, message: BaseModel) -> None:
        raise NotImplementedError


class Coordinator(Agent):

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
            # Validator's final TaskResult also lands here 
            pass

    def receive_final(self, message: TaskResult) -> None:
        print(f"  [Coordinator] Task {message.task_id} CONFIRMED: "
              f"'{message.output}' (confidence {message.confidence})")
        self.final_results[message.task_id] = message


class Worker(Agent):

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
    
            self.bus.send(final)




def run_demo():
    bus = MessageBus()
    coordinator = Coordinator()
    worker = Worker()
    validator = Validator()

    bus.register(coordinator)
    bus.register(worker)
    bus.register(validator)


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
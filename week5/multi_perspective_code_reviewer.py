"""
Multi-Perspective Code Reviewer 
Three specialist agents (Security, Performance, Maintainability) independently
review the same code. A Consensus Agent synthesizes their typed opinions into
one final verdict, explicitly surfacing conflicts between agents.
"""

import json
import os
import sys
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, List, Optional, TypedDict
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()  
#schemas for msgs passing


class Severity(int, Enum):
    TRIVIAL = 1
    MINOR = 2
    MODERATE = 3
    MAJOR = 4
    CRITICAL = 5


class Verdict(str, Enum):
    APPROVE = "approve"
    APPROVE_WITH_COMMENTS = "approve_with_comments"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


class ReviewFinding(BaseModel):
    title: str
    description: str
    severity: Severity
    line_reference: Optional[str] = None
    suggestion: Optional[str] = None


class ReviewOpinion(BaseModel):
    agent_name: str
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    findings: List[ReviewFinding] = Field(default_factory=list)
    reasoning: str


class ConflictAnnotation(BaseModel):
    topic: str
    agents_involved: List[str]
    description: str
    resolution_note: str


class ConsensusVerdict(BaseModel):
    final_verdict: Verdict
    weighted_confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    per_agent_opinions: List[ReviewOpinion]
    conflicts: List[ConflictAnnotation] = Field(default_factory=list)
    top_priority_actions: List[str] = Field(default_factory=list)


class ErrorReport(BaseModel):
    agent_name: str
    error_type: str
    message: str
    recoverable: bool = True


class _SynthesisOutput(BaseModel):
    summary: str
    conflicts: List[ConflictAnnotation] = Field(default_factory=list)
    top_priority_actions: List[str] = Field(default_factory=list)




DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_client = None


def get_client():
    global _client
    if _client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. `export GROQ_API_KEY=...` first.")
        _client = Groq(api_key=api_key)
    return _client


def structured_completion(system_prompt: str, user_prompt: str, schema, max_retries: int = 2):
    client = get_client()
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    full_system = (
        f"{system_prompt}\n\n"
        "Respond with a single valid JSON object matching this schema and nothing else "
        f"(no markdown fences, no preamble):\n{schema_hint}"
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            data = json.loads(response.choices[0].message.content)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError, Exception) as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {last_error}")





class BaseReviewAgent(ABC):
    name: str = "BaseAgent"

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    def review(self, code: str, filename: str) -> ReviewOpinion:
        user_prompt = (
            f"Review the following code file (`{filename}`) strictly from your assigned "
            f"specialty. Do not comment on concerns outside your domain.\n\n```python\n{code}\n```"
        )
        opinion = structured_completion(self.system_prompt, user_prompt, ReviewOpinion)
        opinion.agent_name = self.name
        return opinion


class SecurityAgent(BaseReviewAgent):
    name = "SecurityAgent"
    system_prompt = (
        "You are a senior application security engineer. Your ONLY concern is security: "
        "injection risks, auth/authz flaws, hardcoded secrets, weak input validation, unsafe "
        "deserialization, insecure crypto, sensitive data exposure. Do NOT comment on "
        "performance or style. Assign severity by real-world exploitability. Cite exact "
        "lines for every finding."
    )


class PerformanceAgent(BaseReviewAgent):
    name = "PerformanceAgent"
    system_prompt = (
        "You are a senior performance engineer. Your ONLY concern is runtime efficiency: "
        "algorithmic complexity, N+1 patterns, blocking I/O, memory waste, missing caching, "
        "poor data-structure choices, scalability under load. Do NOT comment on security or "
        "style. Quantify impact where possible. Cite exact lines for every finding."
    )


class MaintainabilityAgent(BaseReviewAgent):
    name = "MaintainabilityAgent"
    system_prompt = (
        "You are a senior engineer focused purely on long-term maintainability: naming "
        "clarity, function size, coupling, missing docs, inconsistent style, swallowed "
        "errors, testability. Do NOT comment on security or raw performance. Every finding "
        "needs a concrete suggestion. Cite exact lines for every finding."
    )




_VERDICT_SEVERITY_ORDER = [Verdict.REJECT, Verdict.REQUEST_CHANGES,
                            Verdict.APPROVE_WITH_COMMENTS, Verdict.APPROVE]


class ConsensusAgent:
    name = "ConsensusAgent"

    def _weighted_verdict(self, opinions: List[ReviewOpinion]) -> tuple:
        total = sum(o.confidence for o in opinions) or 1e-6
        scores: dict = {}
        for o in opinions:
            scores[o.verdict] = scores.get(o.verdict, 0.0) + o.confidence
        best_verdict, best_score = max(scores.items(), key=lambda kv: kv[1])
        weighted_share = round(best_score / total, 3)

        for o in opinions:
            if o.agent_name == "SecurityAgent" and any(f.severity.value >= 5 for f in o.findings):
                if _VERDICT_SEVERITY_ORDER.index(Verdict.REQUEST_CHANGES) < \
                   _VERDICT_SEVERITY_ORDER.index(best_verdict):
                    return Verdict.REQUEST_CHANGES, max(weighted_share, o.confidence)

        return best_verdict, weighted_share

    def synthesize(self, opinions: List[ReviewOpinion], filename: str) -> ConsensusVerdict:
        final_verdict, weighted_confidence = self._weighted_verdict(opinions)

        opinions_blob = "\n\n".join(
            f"### {o.agent_name} (verdict={o.verdict.value}, confidence={o.confidence})\n"
            f"Reasoning: {o.reasoning}\nFindings:\n" +
            "\n".join(f"  - [{f.severity.name}] {f.title}: {f.description}"
                      + (f" (suggestion: {f.suggestion})" if f.suggestion else "")
                      for f in o.findings)
            for o in opinions
        )

        system_prompt = (
            f"You are the Consensus Agent for a review of `{filename}`. You have independent "
            "reports from SecurityAgent, PerformanceAgent, and MaintainabilityAgent below.\n"
            "1. Write a concise executive summary of overall code health.\n"
            "2. Identify CONFLICTS between agents (e.g. a performance fix that reintroduces "
            "a security risk). Don't invent conflicts if agents agree.\n"
            "3. Produce a ranked list of top-priority actions, most severe first.\n"
            f"The final verdict is already decided as '{final_verdict.value}' "
            f"(weighted confidence {weighted_confidence}) — explain it, don't override it. "
            "Don't fabricate findings not present in the reports."
        )

        synthesis = structured_completion(
            system_prompt, f"Specialist reports:\n\n{opinions_blob}", _SynthesisOutput
        )

        return ConsensusVerdict(
            final_verdict=final_verdict,
            weighted_confidence=weighted_confidence,
            summary=synthesis.summary,
            per_agent_opinions=opinions,
            conflicts=synthesis.conflicts,
            top_priority_actions=synthesis.top_priority_actions,
        )




import operator  
from langgraph.graph import StateGraph, START, END  


class ReviewState(TypedDict):
    code: str
    filename: str
    opinions: Annotated[List[ReviewOpinion], operator.add]
    errors: Annotated[List[ErrorReport], operator.add]
    consensus: Optional[ConsensusVerdict]


_security, _performance, _maintainability, _consensus = (
    SecurityAgent(), PerformanceAgent(), MaintainabilityAgent(), ConsensusAgent()
)


def _specialist_node(agent):
    def node(state: ReviewState) -> dict:
        try:
            return {"opinions": [agent.review(state["code"], state["filename"])]}
        except Exception as e:  # noqa: BLE001 — never crash the graph on one agent's failure
            return {"errors": [ErrorReport(agent_name=agent.name, error_type=type(e).__name__,
                                            message=str(e))]}
    return node


def _consensus_node(state: ReviewState) -> dict:
    if not state["opinions"]:
        raise RuntimeError(f"All specialists failed: {state.get('errors')}")
    return {"consensus": _consensus.synthesize(state["opinions"], state["filename"])}


def build_graph():
    graph = StateGraph(ReviewState)
    graph.add_node("security", _specialist_node(_security))
    graph.add_node("performance", _specialist_node(_performance))
    graph.add_node("maintainability", _specialist_node(_maintainability))
    graph.add_node("consensus", _consensus_node)

    for node in ("security", "performance", "maintainability"):
        graph.add_edge(START, node)
        graph.add_edge(node, "consensus")
    graph.add_edge("consensus", END)

    return graph.compile()


def run_review(code: str, filename: str = "submission.py") -> ReviewState:
    app = build_graph()
    return app.invoke({"code": code, "filename": filename, "opinions": [], "errors": [], "consensus": None})



_ICON = {Severity.TRIVIAL: "⚪", Severity.MINOR: "🟡", Severity.MODERATE: "🟠",
         Severity.MAJOR: "🔴", Severity.CRITICAL: "🔥"}


def print_report(result: ReviewState, elapsed_s: float) -> None:
    print("====================================")
    print("MULTI-PERSPECTIVE CODE REVIEW REPORT")
    print("====================================")

    if result["errors"]:
        print("\n  AGENT FAILURES (degraded gracefully):")
        for err in result["errors"]:
            print(f"   - {err.agent_name}: {err.error_type} — {err.message}")

    print(f"\n{len(result['opinions'])} specialist(s) reported in {elapsed_s:.1f}s\n")

    for op in result["opinions"]:
        print(f"── {op.agent_name} " + "─" * max(1, 40 - len(op.agent_name)))
        print(f"   Verdict: {op.verdict.value}  |  Confidence: {op.confidence}")
        print(f"   Reasoning: {op.reasoning}")
        for f in op.findings:
            loc = f" ({f.line_reference})" if f.line_reference else ""
            print(f"   {_ICON.get(f.severity, '•')} [{f.severity.name}] {f.title}{loc}")
            print(f"       {f.description}")
            if f.suggestion:
                print(f"       → Suggestion: {f.suggestion}")
        print()

    c = result["consensus"]
    print("=================")
    print("CONSENSUS VERDICT")
    print("=================")
    print(f"Final verdict: {c.final_verdict.value.upper()}")
    print(f"Weighted confidence: {c.weighted_confidence}")
    print(f"\nSummary:\n{c.summary}")

    if c.conflicts:
        print("\n CONFLICTS BETWEEN AGENTS:")
        for conf in c.conflicts:
            print(f"   Topic: {conf.topic}  |  Agents: {', '.join(conf.agents_involved)}")
            print(f"   {conf.description}")
            print(f"   Resolution: {conf.resolution_note}\n")

    if c.top_priority_actions:
        print(" TOP PRIORITY ACTIONS:")
        for i, action in enumerate(c.top_priority_actions, 1):
            print(f"   {i}. {action}")
    print("==========================")


def main():

    filepath = input("Enter the path of the Python file: ")
    with open(filepath) as f:
        code = f.read()

    start = time.time()
    result = run_review(code, filename=filepath)
    print_report(result, time.time() - start)


if __name__ == "__main__":
    main()
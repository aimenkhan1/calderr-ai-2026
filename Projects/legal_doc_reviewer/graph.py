"""
LangGraph wiring for the Multi-Agent Legal Document Reviewer.

"""

import operator
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from agents.risk_agent import RiskAgent
from agents.compliance_agent import ComplianceAgent
from agents.liability_agent import LiabilityAgent
from agents.obligations_agent import ObligationsAgent
from agents.debate_facilitator import DebateFacilitator
from agents.judge_agent import JudgeAgent
from models import AgentReview, ClauseFinding, DebateChallenge, ErrorReport, JudgeVerdict


class ReviewState(TypedDict):
    contract_text: str
    reviews: Annotated[List[AgentReview], operator.add]
    challenges: List[DebateChallenge]
    final_findings: List[ClauseFinding]
    errors: Annotated[List[ErrorReport], operator.add]
    verdict: Optional[JudgeVerdict]


_risk, _compliance, _liability, _obligations = (
    RiskAgent(), ComplianceAgent(), LiabilityAgent(), ObligationsAgent()
)
_facilitator = DebateFacilitator()
_judge = JudgeAgent()

_AGENT_REGISTRY = {a.name: a for a in (_risk, _compliance, _liability, _obligations)}


def _specialist_node(agent):
    def node(state: ReviewState) -> dict:
        try:
            return {"reviews": [agent.review(state["contract_text"])]}
        except Exception as e: 
            return {"errors": [ErrorReport(agent_name=agent.name, error_type=type(e).__name__,
                                            message=str(e))]}
    return node


def _debate_node(state: ReviewState) -> dict:
    reviews = state["reviews"]
    if not reviews:
        return {"challenges": []}
    challenges = _facilitator.run_challenge_round(reviews)
    return {"challenges": challenges}


def _response_node(state: ReviewState) -> dict:
    reviews = state["reviews"]
    challenges = state["challenges"]

    findings_by_id = {f.finding_id: f for r in reviews for f in r.findings}

    resolved_challenges = []
    for challenge in challenges:
        target_agent = _AGENT_REGISTRY.get(challenge.target_agent)
        target_finding = findings_by_id.get(challenge.target_finding_id)
        if target_agent is None or target_finding is None:
            challenge.resolution = "skipped (unresolvable reference)"
            resolved_challenges.append(challenge)
            continue

        try:
            response = target_agent.respond_to_challenge(
                target_finding.title, target_finding.description, challenge.challenging_rationale
            )
            if response.stands_by_original:
                challenge.resolution = "upheld"
            else:
                challenge.resolution = "revised"
                target_finding.revised = True
                target_finding.revision_note = response.response_reasoning
                if response.updated_severity is not None:
                    target_finding.severity = response.updated_severity
                if response.updated_description:
                    target_finding.description = response.updated_description
        except Exception as e:  # noqa: BLE001
            challenge.resolution = f"skipped (agent error: {e})"

        resolved_challenges.append(challenge)

    final_findings = list(findings_by_id.values())
    return {"challenges": resolved_challenges, "final_findings": final_findings}


def _judge_node(state: ReviewState) -> dict:
    if not state["reviews"]:
        raise RuntimeError(f"All specialist agents failed; no reviews available. "
                            f"Errors: {state.get('errors')}")
    verdict = _judge.render_verdict(state["reviews"], state["challenges"], state["final_findings"])
    return {"verdict": verdict}


def build_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("risk", _specialist_node(_risk))
    graph.add_node("compliance", _specialist_node(_compliance))
    graph.add_node("liability", _specialist_node(_liability))
    graph.add_node("obligations", _specialist_node(_obligations))
    graph.add_node("debate", _debate_node)
    graph.add_node("response", _response_node)
    graph.add_node("judge", _judge_node)

    for specialist in ("risk", "compliance", "liability", "obligations"):
        graph.add_edge(START, specialist)
        graph.add_edge(specialist, "debate")

    graph.add_edge("debate", "response")
    graph.add_edge("response", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


def run_review(contract_text: str) -> ReviewState:
    app = build_graph()
    return app.invoke({
        "contract_text": contract_text, "reviews": [], "challenges": [],
        "final_findings": [], "errors": [], "verdict": None,
    })

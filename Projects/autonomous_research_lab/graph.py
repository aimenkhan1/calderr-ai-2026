"""
LangGraph wiring for the Autonomous AI Research Lab.

Graph shape:

  START ─► classify_domain ─► assemble_team ─► generate_hypothesis
                                                        │
                                          (dynamic fan-out via Send —
                                           3 to 5 evidence_agent calls,
                                           the exact count decided at
                                           runtime by the Assembler)
                                                        │
                                                        ▼
                                    evidence_agent ×N (parallel)
                                                        │
                                              (fan-in, all must finish)
                                                        ▼
                                                     critique
                                                        │
                                                        ▼
                                                   synthesize
                                                        │
                                                        ▼
                                                  peer_review
                                                        │
                                                        ▼
                                                    publish ─► END

The dynamic fan-out (via `Send`) is the key structural difference from
every prior Week 5 project: the number of parallel evidence_agent
invocations is NOT fixed at graph-build time — it's decided at runtime
by the Agent Assembler, based on how many distinct angles THIS
particular question actually has (3 to 5).
"""

import operator
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agents.domain_classifier import DomainClassifierAgent
from agents.agent_assembler import AgentAssemblerAgent
from agents.hypothesis_generator import HypothesisGeneratorAgent
from agents.evidence_agent import EvidenceAgent
from agents.critic_agent import CriticAgent
from agents.synthesis_agent import SynthesisAgent
from agents.peer_review_agent import PeerReviewAgent
from agents.report_publisher import ReportPublisherAgent
from models import (
    AssemblyPlan, CriticReport, DomainClassification, ErrorReport, EvidenceFinding,
    Hypothesis, PeerReviewReport, ResearchReport, SpecialistSpec, SynthesisReport,
)
from rag.document_store import DocumentStore


class ResearchState(TypedDict):
    question: str
    domain_classification: Optional[DomainClassification]
    assembly_plan: Optional[AssemblyPlan]
    hypothesis: Optional[Hypothesis]
    findings: Annotated[List[EvidenceFinding], operator.add]
    critic_report: Optional[CriticReport]
    final_findings: List[EvidenceFinding]
    synthesis: Optional[SynthesisReport]
    peer_review: Optional[PeerReviewReport]
    report: Optional[ResearchReport]
    errors: Annotated[List[ErrorReport], operator.add]


_store = DocumentStore()
_classifier = DomainClassifierAgent()
_assembler = AgentAssemblerAgent()
_hypothesis_gen = HypothesisGeneratorAgent()
_critic = CriticAgent()
_synthesizer = SynthesisAgent()
_peer_reviewer = PeerReviewAgent()
_publisher = ReportPublisherAgent()


def _classify_domain_node(state: ResearchState) -> dict:
    classification = _classifier.classify(state["question"], _store.available_domains())
    return {"domain_classification": classification}


def _assemble_team_node(state: ResearchState) -> dict:
    plan = _assembler.assemble(state["question"], state["domain_classification"].domain)
    return {"assembly_plan": plan}


def _generate_hypothesis_node(state: ResearchState) -> dict:
    hypothesis = _hypothesis_gen.generate(state["question"], state["assembly_plan"])
    return {"hypothesis": hypothesis}


#dynamic fanout - the Assembler decides how many evidence_agent calls to make (3-5) based on the number of distinct angles in the AssemblyPlan. 
def _route_to_evidence_agents(state: ResearchState):

    domain = state["domain_classification"].domain.value
    sends = []
    for i, spec in enumerate(state["assembly_plan"].specialists, start=1):
        sends.append(Send("evidence_agent", {
            "spec": spec, "domain": domain, "finding_id": f"ev-{i}",
        }))
    return sends


def _evidence_agent_node(payload: dict) -> dict:

    spec: SpecialistSpec = payload["spec"]
    domain: str = payload["domain"]
    finding_id: str = payload["finding_id"]

    try:
        agent = EvidenceAgent(spec=spec, domain=domain, store=_store, finding_id=finding_id)
        finding = agent.investigate()
        return {"findings": [finding]}
    except Exception as e:  # noqa: BLE001 — one specialist failing must not crash the run
        return {"errors": [ErrorReport(
            agent_name=f"EvidenceAgent[{spec.persona_name}]",
            error_type=type(e).__name__, message=str(e),
        )]}


def _critique_node(state: ResearchState) -> dict:
    if not state["findings"]:
        raise RuntimeError(f"All evidence agents failed; no findings available. Errors: {state.get('errors')}")
    critic_report, final_findings = _critic.critique(state["hypothesis"], state["findings"])
    return {"critic_report": critic_report, "final_findings": final_findings}


def _synthesize_node(state: ResearchState) -> dict:
    synthesis = _synthesizer.synthesize(
        state["question"], state["hypothesis"], state["final_findings"], state["critic_report"]
    )
    return {"synthesis": synthesis}


def _peer_review_node(state: ResearchState) -> dict:
    peer_review = _peer_reviewer.review(state["synthesis"], state["final_findings"])
    return {"peer_review": peer_review}


def _publish_node(state: ResearchState) -> dict:
    report = _publisher.publish(
        question=state["question"],
        domain=state["domain_classification"].domain,
        assembly_plan=state["assembly_plan"],
        hypothesis=state["hypothesis"],
        findings=state["final_findings"],
        critic_report=state["critic_report"],
        synthesis=state["synthesis"],
        peer_review=state["peer_review"],
    )
    return {"report": report}


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("classify_domain", _classify_domain_node)
    graph.add_node("assemble_team", _assemble_team_node)
    graph.add_node("generate_hypothesis", _generate_hypothesis_node)
    graph.add_node("evidence_agent", _evidence_agent_node)
    graph.add_node("critique", _critique_node)
    graph.add_node("synthesize", _synthesize_node)
    graph.add_node("peer_review", _peer_review_node)
    graph.add_node("publish", _publish_node)

    graph.add_edge(START, "classify_domain")
    graph.add_edge("classify_domain", "assemble_team")
    graph.add_edge("assemble_team", "generate_hypothesis")

    # Dynamic fan-out: generate_hypothesis -> N evidence_agent Sends
    graph.add_conditional_edges("generate_hypothesis", _route_to_evidence_agents, ["evidence_agent"])

    # Fan-in: all evidence_agent calls converge on critique
    graph.add_edge("evidence_agent", "critique")

    graph.add_edge("critique", "synthesize")
    graph.add_edge("synthesize", "peer_review")
    graph.add_edge("peer_review", "publish")
    graph.add_edge("publish", END)

    return graph.compile()


def run_research(question: str) -> ResearchState:
    app = build_graph()
    return app.invoke({
        "question": question, "domain_classification": None, "assembly_plan": None,
        "hypothesis": None, "findings": [], "critic_report": None, "final_findings": [],
        "synthesis": None, "peer_review": None, "report": None, "errors": [],
    })


if __name__ == "__main__":
    app = build_graph()
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Saved graph.png")

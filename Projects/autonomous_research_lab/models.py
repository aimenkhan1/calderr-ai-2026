"""
Typed message schemas for the Autonomous AI Research Lab.

Every phase boundary — domain classification, agent assembly, hypothesis,
evidence gathering, critique, synthesis, peer review, and the final
published report — is a Pydantic model. No raw strings/dicts cross
phase boundaries.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ResearchDomain(str, Enum):
    """
    Supported domains. The seed document store (data/seed_corpus/) has one
    folder per domain, so the Domain Classifier is constrained to pick from
    this list rather than inventing domains with no evidence to retrieve.
    """
    AI_SAFETY = "ai_safety"
    BIOTECHNOLOGY = "biotechnology"
    CLIMATE_TECH = "climate_tech"
    FINTECH = "fintech"
    QUANTUM_COMPUTING = "quantum_computing"


class DomainClassification(BaseModel):
    domain: ResearchDomain
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


class SpecialistSpec(BaseModel):
    """One dynamically-assembled evidence agent's assignment."""
    persona_name: str = Field(..., description="e.g. 'RegulatoryAnalyst', 'TechnicalFeasibilityExpert'")
    expertise_description: str
    sub_question: str = Field(..., description="The specific sub-question this specialist will investigate")


class AssemblyPlan(BaseModel):
    domain: ResearchDomain
    specialists: List[SpecialistSpec] = Field(..., min_length=3, max_length=5)
    assembly_rationale: str


class Hypothesis(BaseModel):
    statement: str
    rationale: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class EvidenceFinding(BaseModel):
    finding_id: str = Field(..., description="Short unique id, e.g. 'ev-1'")
    persona_name: str
    sub_question: str
    summary: str
    supporting_points: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list, description="Seed document filenames cited")
    used_tool_call: bool = Field(default=False, description="Whether this agent called the document-lookup tool")
    # Populated only if the Critic Agent challenges and modifies this finding
    weakened: bool = False
    weakness_note: Optional[str] = None
    original_confidence: Optional[float] = None


class CriticChallenge(BaseModel):
    target_finding_id: str
    issue: str
    severity: str = Field(..., description="'minor', 'moderate', or 'major'")
    revised_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class CriticReport(BaseModel):
    challenges: List[CriticChallenge] = Field(default_factory=list)
    hypothesis_alignment_note: str
    overall_evidence_quality: str


class ReportSection(BaseModel):
    heading: str
    content: str
    cited_finding_ids: List[str] = Field(default_factory=list)


class SynthesisReport(BaseModel):
    title: str
    executive_summary: str
    sections: List[ReportSection]
    conclusion: str


class PeerReviewReport(BaseModel):
    approved: bool
    contradictions_found: List[str] = Field(default_factory=list)
    unsupported_claims_found: List[str] = Field(default_factory=list)
    notes: str


class ResearchReport(BaseModel):
    """Final published output."""
    question: str
    domain: ResearchDomain
    assembly_plan: AssemblyPlan
    hypothesis: Hypothesis
    findings: List[EvidenceFinding]
    critic_report: CriticReport
    synthesis: SynthesisReport
    peer_review: PeerReviewReport
    published_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ErrorReport(BaseModel):
    agent_name: str
    error_type: str
    message: str
    recoverable: bool = True


# ── Internal schemas used only for structured LLM calls ──────────────────

class _EvidenceOutput(BaseModel):
    summary: str
    supporting_points: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)


class _CriticOutput(BaseModel):
    challenges: List[CriticChallenge] = Field(default_factory=list)
    hypothesis_alignment_note: str
    overall_evidence_quality: str


class _ToolDecision(BaseModel):
    wants_tool_call: bool
    source_to_fetch: Optional[str] = None
    reasoning: str

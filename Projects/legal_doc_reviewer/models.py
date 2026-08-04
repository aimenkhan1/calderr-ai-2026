"""
Typed message schemas for the Multi-Agent Legal Document Reviewer.

Every agent boundary — specialist review, debate challenge/response,
and final judge verdict — is a Pydantic model. No raw strings/dicts
cross agent boundaries.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Severity(int, Enum):
    TRIVIAL = 1
    MINOR = 2
    MODERATE = 3
    MAJOR = 4
    CRITICAL = 5


class ClauseCategory(str, Enum):
    RISK = "risk"
    COMPLIANCE = "compliance"
    LIABILITY = "liability"
    OBLIGATION = "obligation"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClauseFinding(BaseModel):
    """A single flagged clause or extracted obligation."""
    finding_id: str = Field(..., description="Short unique id, e.g. 'risk-1', 'oblig-3'")
    clause_excerpt: str = Field(..., description="The relevant excerpt or paraphrase of the clause")
    title: str
    description: str
    category: ClauseCategory
    severity: Severity
    suggestion: Optional[str] = None
    # Populated only after the debate round, if this finding was challenged & revised
    revised: bool = False
    revision_note: Optional[str] = None


class AgentReview(BaseModel):
    """Structured output produced by each specialist agent in round 1."""
    agent_name: str
    overall_assessment: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    findings: List[ClauseFinding] = Field(default_factory=list)


class DebateChallenge(BaseModel):
    """One challenge raised by the Debate Facilitator against a specific finding."""
    challenge_id: str
    challenging_rationale: str = Field(
        ..., description="Why this finding might be overstated, understated, or contested "
                          "from another agent's perspective"
    )
    target_agent: str
    target_finding_id: str
    resolution: Optional[str] = Field(
        None, description="'upheld' or 'revised' — filled in after the target agent responds"
    )


class DebateTranscript(BaseModel):
    challenges: List[DebateChallenge] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    """Final output from the Judge Agent."""
    overall_risk_level: RiskLevel
    summary: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    final_findings: List[ClauseFinding]
    dissent_log: List[str] = Field(
        default_factory=list, description="Cases where agents disagreed and were not fully reconciled"
    )


class ErrorReport(BaseModel):
    agent_name: str
    error_type: str
    message: str
    recoverable: bool = True


# ── Internal schemas used only for structured LLM calls (not exposed as final output) ──


class _FacilitatorOutput(BaseModel):
    challenges: List[DebateChallenge] = Field(default_factory=list)


class _ChallengeResponse(BaseModel):
    stands_by_original: bool = Field(..., description="True = upheld, False = revised")
    updated_severity: Optional[Severity] = None
    updated_description: Optional[str] = None
    response_reasoning: str


class _JudgeSynthesis(BaseModel):
    overall_risk_level: RiskLevel
    summary: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    dissent_log: List[str] = Field(default_factory=list)

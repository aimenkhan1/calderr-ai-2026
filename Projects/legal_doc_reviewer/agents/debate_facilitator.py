"""
Debate Facilitator — runs one structured challenge round.

After all 4 specialists review independently, the Facilitator looks
across every finding for genuine cross-perspective tension (e.g. a
Risk finding that Liability's indemnification clause actually
mitigates, or an Obligations deadline that Compliance thinks is
regulatorily impossible to meet) and raises challenges. This is what
produces "at least one changed finding" rather than four reports
stapled together.
"""

from typing import List

from llm_client import structured_completion
from models import AgentReview, DebateChallenge, _FacilitatorOutput


class DebateFacilitator:
    name = "DebateFacilitator"

    def run_challenge_round(self, reviews: List[AgentReview]) -> List[DebateChallenge]:
        findings_blob = "\n\n".join(
            f"### {r.agent_name} (confidence={r.confidence})\n"
            f"Overall: {r.overall_assessment}\n"
            + "\n".join(
                f"  - [{f.finding_id}] [{f.severity.name}] {f.title}: {f.description}"
                for f in r.findings
            )
            for r in reviews
        )

        system_prompt = (
            "You are the Debate Facilitator for a multi-agent legal document review. "
            "Four specialists (RiskAgent, ComplianceAgent, LiabilityAgent, "
            "ObligationsAgent) have each reviewed the same contract independently, "
            "shown below. Your job is to find genuine cross-perspective tension — "
            "cases where one agent's finding looks different once you consider "
            "another agent's perspective. Examples: a Risk finding that a Liability "
            "clause actually addresses; an Obligations deadline that looks "
            "unreasonable given a Compliance constraint; a severity rating that "
            "seems inflated or understated once you weigh a competing concern.\n\n"
            "Raise AT LEAST ONE real challenge if there is genuine substance for it. "
            "Do NOT invent artificial disagreement — only challenge findings where "
            "you can articulate a specific, substantive reason. Reference the exact "
            "finding_id you are challenging."
        )

        result: _FacilitatorOutput = structured_completion(
            system_prompt, findings_blob, _FacilitatorOutput
        )
        return result.challenges


ee
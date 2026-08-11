"""
Judge Agent — weighs all arguments (post-debate), assigns final
severity per clause, and produces the final risk assessment with
confidence + an explicit dissent log for anything never fully
reconciled during the debate round.
"""

from typing import List

from llm_client import structured_completion
from models import AgentReview, ClauseFinding, DebateChallenge, JudgeVerdict, _JudgeSynthesis


class JudgeAgent:
    name = "JudgeAgent"

    def render_verdict(
        self,
        reviews: List[AgentReview],
        challenges: List[DebateChallenge],
        final_findings: List[ClauseFinding],
    ) -> JudgeVerdict:
        findings_blob = "\n".join(
            f"- [{f.finding_id}] [{f.category.value}] [{f.severity.name}] {f.title}"
            + (f" (REVISED during debate: {f.revision_note})" if f.revised else "")
            for f in final_findings
        )
        challenges_blob = "\n".join(
            f"- Challenge on {c.target_finding_id} (targeting {c.target_agent}): "
            f"{c.challenging_rationale} -> resolution: {c.resolution}"
            for c in challenges
        ) or "No challenges were raised during debate."

        reviews_blob = "\n".join(
            f"- {r.agent_name}: {r.overall_assessment} (confidence={r.confidence})"
            for r in reviews
        )

        system_prompt = (
            "You are the Judge Agent, the final decision-maker in a multi-agent "
            "legal document review. You have: (1) each specialist's overall "
            "assessment, (2) the final list of findings after the debate round "
            "(some may have been revised), and (3) the debate transcript. "
            "Produce an overall risk level for the contract, an executive summary, "
            "your confidence in this assessment, and a dissent_log listing any "
            "specific disagreements between agents that the debate round did NOT "
            "fully resolve — do not hide unresolved disagreement, name it plainly. "
            "If everything was resolved or agents never disagreed, dissent_log can "
            "be empty."
        )
        user_prompt = (
            f"Specialist assessments:\n{reviews_blob}\n\n"
            f"Final findings (post-debate):\n{findings_blob}\n\n"
            f"Debate transcript:\n{challenges_blob}"
        )

        synthesis: _JudgeSynthesis = structured_completion(system_prompt, user_prompt, _JudgeSynthesis)

        return JudgeVerdict(
            overall_risk_level=synthesis.overall_risk_level,
            summary=synthesis.summary,
            confidence=synthesis.confidence,
            final_findings=final_findings,
            dissent_log=synthesis.dissent_log,
        )


"""
Peer Review Agent — Phase 5. Independent second-pass check on the
SYNTHESIZED REPORT ITSELF (not the raw findings) for internal
contradictions and unsupported claims — catching mistakes the
Synthesis Agent introduced while writing, which the Critic Agent
(which only ever saw raw findings) never had a chance to review.
"""

from llm_client import structured_completion
from models import EvidenceFinding, PeerReviewReport, SynthesisReport


class PeerReviewAgent:
    name = "PeerReviewAgent"

    def review(self, synthesis: SynthesisReport, findings: list[EvidenceFinding]) -> PeerReviewReport:
        report_blob = "\n\n".join(
            f"## {s.heading}\n{s.content}\n(cites: {s.cited_finding_ids})"
            for s in synthesis.sections
        )
        findings_ids = {f.finding_id for f in findings}

        system_prompt = (
            "You are the Peer Review Agent, doing an independent second-pass quality "
            "check on a finished research report — you did NOT write it. Check for:\n"
            "1. Internal contradictions — does any section contradict another section, "
            "or contradict the executive summary/conclusion?\n"
            "2. Unsupported claims — any specific factual claim in the report that "
            "does NOT trace back to a cited finding_id, or where the cited finding "
            "doesn't actually support what's claimed.\n"
            "3. Whether the conclusion's stated relationship to the hypothesis is "
            "actually justified by the body of the report.\n\n"
            "Set approved=false if you find ANY major issue. Be a genuine second pair "
            "of eyes, not a rubber stamp."
        )
        user_prompt = (
            f"Report title: {synthesis.title}\n\n"
            f"Executive summary: {synthesis.executive_summary}\n\n"
            f"Sections:\n{report_blob}\n\n"
            f"Conclusion: {synthesis.conclusion}\n\n"
            f"Valid finding_ids that exist: {sorted(findings_ids)}"
        )

        return structured_completion(system_prompt, user_prompt, PeerReviewReport)

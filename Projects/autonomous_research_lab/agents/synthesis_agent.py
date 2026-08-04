"""
Synthesis Agent — Phase 4. Writes the actual report, citing evidence
finding IDs for every claim so the Peer Review Agent (and any human
reader) can trace every statement back to its source finding.
"""

from llm_client import structured_completion
from models import CriticReport, EvidenceFinding, Hypothesis, SynthesisReport


class SynthesisAgent:
    name = "SynthesisAgent"

    def synthesize(
        self, question: str, hypothesis: Hypothesis,
        findings: list[EvidenceFinding], critic_report: CriticReport,
    ) -> SynthesisReport:
        findings_blob = "\n\n".join(
            f"[{f.finding_id}] {f.persona_name}: {f.summary}"
            + (f" (NOTE: critic flagged this — {f.weakness_note})" if f.weakened else "")
            for f in findings
        )

        system_prompt = (
            "You are the Synthesis Agent. Write a professional research report "
            "synthesising the hypothesis, evidence findings, and critic assessment "
            "below into a coherent narrative. Requirements:\n"
            "- Organize into logical sections (not one per finding — group by theme).\n"
            "- Every substantive claim must cite the finding_id(s) it's based on in "
            "cited_finding_ids for that section.\n"
            "- Explicitly address findings the Critic flagged — either explain why "
            "they're still credible or acknowledge the limitation in the text, don't "
            "silently ignore critic warnings.\n"
            "- The conclusion must state whether the evidence, on balance, supports, "
            "partially supports, or fails to support the original hypothesis."
        )
        user_prompt = (
            f"Research question: {question}\n\n"
            f"Original hypothesis: {hypothesis.statement}\n\n"
            f"Evidence findings:\n{findings_blob}\n\n"
            f"Critic's hypothesis alignment note: {critic_report.hypothesis_alignment_note}\n"
            f"Critic's overall evidence quality assessment: {critic_report.overall_evidence_quality}"
        )

        return structured_completion(system_prompt, user_prompt, SynthesisReport)

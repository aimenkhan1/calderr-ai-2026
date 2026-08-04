"""
Report Publisher — formats everything into the final ResearchReport
object and can render it to Markdown for the file-based deliverables.
No LLM call needed here — this is pure assembly/formatting, deliberately
kept deterministic so the final report structure is never at the mercy
of an extra LLM call that could omit something.
"""

from models import (
    AssemblyPlan, CriticReport, EvidenceFinding, Hypothesis,
    PeerReviewReport, ResearchDomain, ResearchReport, SynthesisReport,
)


class ReportPublisherAgent:
    name = "ReportPublisher"

    def publish(
        self, question: str, domain: ResearchDomain, assembly_plan: AssemblyPlan,
        hypothesis: Hypothesis, findings: list[EvidenceFinding], critic_report: CriticReport,
        synthesis: SynthesisReport, peer_review: PeerReviewReport,
    ) -> ResearchReport:
        return ResearchReport(
            question=question,
            domain=domain,
            assembly_plan=assembly_plan,
            hypothesis=hypothesis,
            findings=findings,
            critic_report=critic_report,
            synthesis=synthesis,
            peer_review=peer_review,
        )

    @staticmethod
    def to_markdown(report: ResearchReport) -> str:
        lines = [
            f"# {report.synthesis.title}",
            "",
            f"**Research question:** {report.question}  ",
            f"**Domain:** {report.domain.value}  ",
            f"**Published:** {report.published_at}  ",
            f"**Peer review status:** {'✅ Approved' if report.peer_review.approved else '⚠️ Flagged for issues'}",
            "",
            "## Executive Summary",
            report.synthesis.executive_summary,
            "",
            "## Hypothesis",
            f"> {report.hypothesis.statement}",
            "",
            f"*Initial confidence: {report.hypothesis.confidence}*  ",
            f"*Rationale: {report.hypothesis.rationale}*",
            "",
            "## Dynamically Assembled Research Team",
        ]
        for s in report.assembly_plan.specialists:
            lines.append(f"- **{s.persona_name}** — {s.expertise_description}  \n  Sub-question: _{s.sub_question}_")
        lines.append("")

        for section in report.synthesis.sections:
            lines.append(f"## {section.heading}")
            lines.append(section.content)
            if section.cited_finding_ids:
                lines.append(f"\n*Sources: {', '.join(section.cited_finding_ids)}*")
            lines.append("")

        lines.append("## Conclusion")
        lines.append(report.synthesis.conclusion)
        lines.append("")

        lines.append("## Evidence Findings (Raw)")
        for f in report.findings:
            tag = f" — ⚠️ *revised by Critic: {f.weakness_note}*" if f.weakened else ""
            lines.append(f"### [{f.finding_id}] {f.persona_name}{tag}")
            lines.append(f"**Sub-question:** {f.sub_question}  ")
            lines.append(f"**Confidence:** {f.confidence}" + (f" (was {f.original_confidence})" if f.weakened else "") + "  ")
            lines.append(f"**Sources:** {', '.join(f.sources) if f.sources else 'none'}  ")
            if f.used_tool_call:
                lines.append("**Used tool call:** fetch_full_document  ")
            lines.append(f"\n{f.summary}\n")
        lines.append("")

        lines.append("## Critic Report")
        lines.append(f"**Overall evidence quality:** {report.critic_report.overall_evidence_quality}  ")
        lines.append(f"**Hypothesis alignment:** {report.critic_report.hypothesis_alignment_note}")
        if report.critic_report.challenges:
            lines.append("\n**Challenges raised:**")
            for c in report.critic_report.challenges:
                lines.append(f"- [{c.severity.upper()}] `{c.target_finding_id}`: {c.issue}")
        lines.append("")

        lines.append("## Peer Review")
        lines.append(f"**Status:** {'Approved' if report.peer_review.approved else 'Flagged'}  ")
        lines.append(f"**Notes:** {report.peer_review.notes}")
        if report.peer_review.contradictions_found:
            lines.append(f"\n**Contradictions found:** {'; '.join(report.peer_review.contradictions_found)}")
        if report.peer_review.unsupported_claims_found:
            lines.append(f"\n**Unsupported claims found:** {'; '.join(report.peer_review.unsupported_claims_found)}")

        return "\n".join(lines)

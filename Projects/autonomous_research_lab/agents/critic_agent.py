"""
Critic Agent — Phase 3. Reviews the hypothesis against all evidence
findings and explicitly marks weak links: findings with thin support,
low source diversity, or that contradict other findings get flagged and
have their confidence revised downward (or occasionally upward, if the
critic finds the original agent was unduly hedgy relative to strong
evidence). This directly satisfies "Critic Agent modifies at least one
finding" — the critic can rewrite confidence scores itself, no separate
debate round required.
"""

from llm_client import structured_completion
from models import CriticReport, EvidenceFinding, Hypothesis, _CriticOutput


class CriticAgent:
    name = "CriticAgent"

    def critique(self, hypothesis: Hypothesis, findings: list[EvidenceFinding]) -> tuple[CriticReport, list[EvidenceFinding]]:
        findings_blob = "\n\n".join(
            f"[{f.finding_id}] {f.persona_name} (confidence={f.confidence}, sources={f.sources})\n"
            f"Sub-question: {f.sub_question}\n"
            f"Summary: {f.summary}\n"
            f"Supporting points: {'; '.join(f.supporting_points)}"
            for f in findings
        )

        system_prompt = (
            "You are the Critic Agent for a research pipeline. Your job is to find "
            "weak links, not to be agreeable. For the hypothesis and evidence findings "
            "below, evaluate:\n"
            "1. Does the evidence actually support the hypothesis, partially support it, "
            "or contradict it?\n"
            "2. For EACH finding, is the confidence score justified? Flag any finding "
            "with thin sourcing, vague supporting points, or an unjustifiably high "
            "confidence given weak evidence. You may also flag a finding as "
            "UNDER-confident if its own evidence is actually strong.\n"
            "3. Note any findings that contradict each other.\n\n"
            "For every challenge you raise, specify target_finding_id, the issue, a "
            "severity (minor/moderate/major), and a revised_confidence if you believe "
            "the score should change. Only challenge findings where you have a specific, "
            "articulable reason — do not manufacture criticism for its own sake."
        )
        user_prompt = (
            f"Hypothesis: {hypothesis.statement} (confidence={hypothesis.confidence})\n"
            f"Hypothesis rationale: {hypothesis.rationale}\n\n"
            f"Evidence findings:\n{findings_blob}"
        )

        result: _CriticOutput = structured_completion(system_prompt, user_prompt, _CriticOutput)

        findings_by_id = {f.finding_id: f for f in findings}
        for challenge in result.challenges:
            target = findings_by_id.get(challenge.target_finding_id)
            if target is None:
                continue
            target.original_confidence = target.confidence
            target.weakened = True
            target.weakness_note = f"[{challenge.severity.upper()}] {challenge.issue}"
            if challenge.revised_confidence is not None:
                target.confidence = challenge.revised_confidence

        critic_report = CriticReport(
            challenges=result.challenges,
            hypothesis_alignment_note=result.hypothesis_alignment_note,
            overall_evidence_quality=result.overall_evidence_quality,
        )
        return critic_report, list(findings_by_id.values())

"""
Hypothesis Generator — Phase 1. Proposes a structured hypothesis BEFORE
evidence is gathered, so the Critic Agent later has something concrete
to check evidence alignment against.
"""

from llm_client import structured_completion
from models import AssemblyPlan, Hypothesis


class HypothesisGeneratorAgent:
    name = "HypothesisGenerator"

    def generate(self, question: str, assembly_plan: AssemblyPlan) -> Hypothesis:
        specialists_blob = "\n".join(
            f"- {s.persona_name}: {s.sub_question}" for s in assembly_plan.specialists
        )
        system_prompt = (
            "You are the Hypothesis Generator for a research system. Before any "
            "evidence has been gathered, propose your best-reasoned hypothesis "
            "for how the research question will likely resolve, based on general "
            "domain knowledge alone. Be specific and falsifiable — a vague hedge "
            "like 'it depends on many factors' is not a usable hypothesis. State "
            "your genuine confidence; this hypothesis WILL be checked against "
            "evidence later, so do not inflate confidence to sound authoritative."
        )
        user_prompt = (
            f"Research question: {question}\n\n"
            f"Planned evidence-gathering specialists:\n{specialists_blob}"
        )
        return structured_completion(system_prompt, user_prompt, Hypothesis)

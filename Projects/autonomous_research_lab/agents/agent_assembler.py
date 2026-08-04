"""
Dynamic Agent Assembler — the core "adapts to the question" piece.

Given the classified domain and the actual research question, this
generates 3-5 SpecialistSpecs on the fly (persona + sub-question each).
This is what makes the Evidence phase genuinely dynamic rather than a
fixed set of hardcoded agents: the number and focus of specialists
changes per question.
"""

from llm_client import structured_completion
from models import AssemblyPlan, ResearchDomain


class AgentAssemblerAgent:
    name = "AgentAssembler"

    def assemble(self, question: str, domain: ResearchDomain) -> AssemblyPlan:
        system_prompt = (
            "You are the Dynamic Agent Assembler for a multi-agent research system. "
            f"Given a research question in the '{domain.value}' domain, design a team "
            "of 3 to 5 specialist evidence-gathering agents tailored to THIS specific "
            "question — not a generic template team. Each specialist needs:\n"
            "- persona_name: a short role name (e.g. 'RegulatoryAnalyst', "
            "'TechnicalFeasibilityExpert', 'EconomicImpactAnalyst')\n"
            "- expertise_description: what this persona focuses on\n"
            "- sub_question: ONE specific, answerable sub-question this specialist will "
            "investigate — sub-questions together should meaningfully decompose the "
            "main research question, with minimal overlap between specialists.\n\n"
            "Choose the number of specialists (3, 4, or 5) based on how many genuinely "
            "distinct angles this specific question has — do not always default to 5."
        )
        user_prompt = f"Research question: {question}"

        plan = structured_completion(system_prompt, user_prompt, AssemblyPlan)
        plan.domain = domain  # ensure consistency regardless of what the model echoed back
        return plan

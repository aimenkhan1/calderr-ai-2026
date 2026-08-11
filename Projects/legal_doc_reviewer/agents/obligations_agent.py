"""
Obligations Agent — extracts all party obligations and deadlines.

Unlike the other three (which flag problems), this agent's job is
primarily extraction: it still reports via ClauseFinding so the
schema stays uniform, but severity here reflects how easy an
obligation is to miss/breach rather than how "bad" it is.
"""

from agents.base_agent import BaseLegalAgent
from models import ClauseCategory


class ObligationsAgent(BaseLegalAgent):
    name = "ObligationsAgent"
    category = ClauseCategory.OBLIGATION

    @property
    def system_prompt(self) -> str:
        return (
            "You are a contracts obligations tracker. Your ONLY job is to extract "
            "every concrete obligation and deadline each party has under this contract. "
            "For each one, identify:\n"
            "- Which party owes the obligation\n"
            "- What exactly must be done\n"
            "- The deadline or trigger condition (specific date, 'within N days of X', "
            "recurring, or ongoing)\n\n"
            "Use severity to represent how easy this obligation is to accidentally "
            "breach or miss: CRITICAL = hard deadline with no cure period and real "
            "consequences, TRIVIAL = ongoing/soft obligation with little downside if "
            "delayed. Do NOT comment on risk, liability mechanics, or regulatory "
            "compliance — those are other agents' domains. For every finding, quote "
            "or closely paraphrase the relevant clause excerpt. Do not skip any "
            "obligation just because it seems minor — completeness matters more than "
            "brevity here."
        )



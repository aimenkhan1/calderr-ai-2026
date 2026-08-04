"""
Base class for the 4 specialist legal review agents (Risk, Compliance,
Liability, Obligations). Each subclass only supplies its persona's
system prompt and category — the review + debate-response contract
is shared.
"""

from abc import ABC, abstractmethod

from llm_client import structured_completion
from models import AgentReview, ClauseCategory, DebateChallenge, _ChallengeResponse


class BaseLegalAgent(ABC):
    name: str = "BaseLegalAgent"
    category: ClauseCategory = ClauseCategory.RISK

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    def review(self, contract_text: str) -> AgentReview:
        user_prompt = (
            "Review the following contract strictly from your assigned specialty. "
            "Give each finding a short unique finding_id prefixed with your category "
            f"(e.g. '{self.category.value}-1', '{self.category.value}-2').\n\n"
            f"CONTRACT:\n{contract_text}"
        )
        review = structured_completion(self.system_prompt, user_prompt, AgentReview)
        review.agent_name = self.name
        for f in review.findings:
            f.category = self.category
        return review

    def respond_to_challenge(self, finding_title: str, finding_description: str,
                              challenge_rationale: str) -> _ChallengeResponse:
        """Called during the debate round when another perspective contests a finding."""
        system_prompt = (
            f"{self.system_prompt}\n\n"
            "You previously flagged a finding in a contract review. The Debate "
            "Facilitator has raised a challenge to it from a different perspective. "
            "Consider the challenge honestly: uphold your original finding if you "
            "still believe it's correct, or revise it (lower/raise severity, update "
            "the description) if the challenge has merit. Do not be stubborn or "
            "concede for its own sake — decide on the substance."
        )
        user_prompt = (
            f"Your original finding: {finding_title} — {finding_description}\n\n"
            f"Challenge raised: {challenge_rationale}"
        )
        return structured_completion(system_prompt, user_prompt, _ChallengeResponse)

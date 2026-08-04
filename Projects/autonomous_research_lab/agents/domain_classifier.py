"""
Domain Classifier — identifies which research domain a question belongs
to, constrained to the domains the seed document store actually covers.
"""

from llm_client import structured_completion
from models import DomainClassification, ResearchDomain


class DomainClassifierAgent:
    name = "DomainClassifier"

    def classify(self, question: str, available_domains: list[str]) -> DomainClassification:
        domains_list = ", ".join(available_domains)
        system_prompt = (
            "You are a research domain classifier. Given a research question, "
            f"classify it into exactly ONE of these supported domains: {domains_list}. "
            "Even if the question spans multiple areas, pick the single domain that "
            "is most central to answering it. Give a confidence score reflecting how "
            "cleanly the question fits that domain (lower confidence if it's a stretch)."
        )
        user_prompt = f"Research question: {question}"
        classification = structured_completion(system_prompt, user_prompt, DomainClassification)
        return classification

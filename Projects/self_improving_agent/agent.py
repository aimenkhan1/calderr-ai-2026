"""
agent.py

SelfImprovingAgent - wires together every piece into the full loop from
the system architecture diagram:

    User Input
      -> is it a correction? (feedback_handler)
           -> YES: extract a rule from it (correction_extractor),
                    save/reinforce it (rule_store), mark the PREVIOUS
                    response as wrong (performance_tracker)
           -> NO:  find relevant learned rules (rule_store),
                    generate a response using them (response_generator),
                    log the response (performance_tracker)

Call `handle_turn(user_input, interaction_number)` once per message in a
conversation -- it figures out which path to take and returns what
happened (useful for both the demo script and the Streamlit UI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from feedback_handler import is_correction
from correction_extractor import CorrectionExtractor
from response_generator import ResponseGenerator
from rule_store import RuleStore, Rule
from performance_tracker import PerformanceTracker


@dataclass
class TurnResult:
    interaction_number: int
    user_input: str
    was_correction: bool
    agent_reply: str
    applied_rules: List[Rule]           # rules used to shape a normal response
    new_or_reinforced_rule: Optional[Rule]  # set only if this turn was a correction


class SelfImprovingAgent:
    def __init__(self, rules_db_path: str = "rules.db",
                 performance_db_path: str = "performance.db",
                 generator_mode: str = "auto", extractor_mode: str = "auto"):
        self.rule_store = RuleStore(db_path=rules_db_path)
        self.performance_tracker = PerformanceTracker(db_path=performance_db_path)
        self.response_generator = ResponseGenerator(mode=generator_mode)
        self.correction_extractor = CorrectionExtractor(mode=extractor_mode)

        self._last_response: str = ""  # needed for correction context ("what did I say wrong?")

    def handle_turn(self, user_input: str, interaction_number: int) -> TurnResult:
        if is_correction(user_input):
            return self._handle_correction(user_input, interaction_number)
        return self._handle_question(user_input, interaction_number)

    # Path 1: the user is correcting the previous response

    def _handle_correction(self, user_input: str, interaction_number: int) -> TurnResult:
        extracted = self.correction_extractor.extract(user_input, previous_response=self._last_response)

        rule = self.rule_store.add_or_reinforce_correction(
            original_mistake=extracted.original_mistake,
            correction=extracted.correction,
            rule_text=extracted.rule_text,
            domain=extracted.domain,
        )

        # The response we gave last turn is now confirmed wrong.
        self.performance_tracker.mark_previous_as_corrected()

        acknowledgement = f"Got it -- I'll remember: {rule.rule_text}"
        return TurnResult(
            interaction_number=interaction_number,
            user_input=user_input,
            was_correction=True,
            agent_reply=acknowledgement,
            applied_rules=[],
            new_or_reinforced_rule=rule,
        )

    # Path 2: a normal question -- retrieve rules, generate, log

    def _handle_question(self, user_input: str, interaction_number: int) -> TurnResult:
        # If the previous response was never corrected, the user moving on
        # to a new question is itself a (soft) signal it was accepted.
        self.performance_tracker.mark_previous_as_accepted()

        relevant_rules = self.rule_store.get_relevant_rules(user_input, top_k=5, min_similarity=0.06)
        response = self.response_generator.generate(user_input, relevant_rules)

        for rule in relevant_rules:
            self.rule_store.record_application(rule.id)

        self.performance_tracker.record_response(
            interaction_number=interaction_number,
            user_input=user_input,
            agent_response=response,
            applied_rule_ids=[r.id for r in relevant_rules],
        )

        self._last_response = response
        return TurnResult(
            interaction_number=interaction_number,
            user_input=user_input,
            was_correction=False,
            agent_reply=response,
            applied_rules=relevant_rules,
            new_or_reinforced_rule=None,
        )

    def close(self) -> None:
        self.rule_store.close()
        self.performance_tracker.close()

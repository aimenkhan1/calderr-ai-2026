"""
3-AGENT DEBATE
==========================
Proposer, Challenger, Arbiter in which Arbiter weighs argument QUALITY, not recency.-used by addition
"""

from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field




class Argument(BaseModel):

    text: str
    evidence_strength: float = Field(ge=0.0, le=1.0)





class Challenge(BaseModel):
    sender: str = "Challenger"
    topic: str
    counterarguments: List[Argument]


class Verdict(BaseModel):

    topic: str
    winner: Literal["Proposer", "Challenger", "Tie"]
    proposer_total_strength: float
    challenger_total_strength: float
    reasoning: str
    decided_by: str = "cumulative_evidence_strength"   # never who told later is better 




class ProposerAgent:
    name = "Proposer"

    def propose(self, topic: str, claim: str, arguments: List[Argument]) -> Proposal:
        return Proposal(topic=topic, claim=claim, arguments=arguments)


class ChallengerAgent:
    name = "Challenger"

    def challenge(self, proposal: Proposal, counterarguments: List[Argument]) -> Challenge:
        return Challenge(topic=proposal.topic, counterarguments=counterarguments)


class ArbiterAgent:
    name = "Arbiter"

    def judge(self, proposal: Proposal, challenge: Challenge) -> Verdict:
        proposer_total = round(sum(a.evidence_strength for a in proposal.arguments), 2)
        challenger_total = round(sum(a.evidence_strength for a in challenge.counterarguments), 2)

        if proposer_total > challenger_total:
            winner = "Proposer"
        elif challenger_total > proposer_total:
            winner = "Challenger"
        else:
            winner = "Tie"

        reasoning = (
            f"Proposer's {len(proposal.arguments)} argument(s) totaled "
            f"{proposer_total} evidence strength. "
            f"Challenger's {len(challenge.counterarguments)} counterargument(s) totaled "
            f"{challenger_total} evidence strength. "
            f"Decision based on cumulative evidence strength, NOT on which side spoke "
            f"most recently (the Challenger always speaks last in this format, "
            f"but that has no bearing on this verdict)."
        )

        return Verdict(
            topic=proposal.topic,
            winner=winner,
            proposer_total_strength=proposer_total,
            challenger_total_strength=challenger_total,
            reasoning=reasoning,
        )




def run_debate(topic: str, claim: str,
                proposer_args: List[Argument],
                challenger_args: List[Argument]) -> Verdict:
    proposer = ProposerAgent()
    challenger = ChallengerAgent()
    arbiter = ArbiterAgent()

    print(f"\n--- Debate: {topic} ---")

    proposal = proposer.propose(topic, claim, proposer_args)
    print(f"[Proposer] Claim: {proposal.claim}")
    for a in proposal.arguments:
        print(f"    - ({a.evidence_strength}) {a.text}")

    challenge = challenger.challenge(proposal, challenger_args)
    print(f"[Challenger] Counterarguments:")
    for a in challenge.counterarguments:
        print(f"    - ({a.evidence_strength}) {a.text}")

    verdict = arbiter.judge(proposal, challenge)
    print(f"[Arbiter] Winner: {verdict.winner} "
          f"(Proposer {verdict.proposer_total_strength} vs "
          f"Challenger {verdict.challenger_total_strength})")
    print(f"[Arbiter] Reasoning: {verdict.reasoning}")

    return verdict




def verify_no_recency_bias():
    print("===================================================================")
    print("SCENARIO A: Proposer has STRONG arguments, Challenger has WEAK ones")
    print("(Challenger still speaks LAST - if the Arbiter favored recency,")
    print(" it would wrongly pick the Challenger anyway)")
    print("===================================================================")

    verdict_a = run_debate(
        topic="Should the company adopt a 4-day work week?",
        claim="Yes - a 4-day work week improves productivity.",
        proposer_args=[
            Argument(text="A 2024 controlled trial across 60 companies showed a 5% "
                           "revenue increase with no productivity loss.", evidence_strength=0.9),
            Argument(text="Employee turnover dropped 15% in trial companies.", evidence_strength=0.85),
        ],
        challenger_args=[
            Argument(text="Some people just don't like change.", evidence_strength=0.1),
            Argument(text="It might not work here, who knows.", evidence_strength=0.15),
        ],
    )

    assert verdict_a.winner == "Proposer", (
        "FAILURE: Arbiter picked the Challenger despite weaker arguments - "
        "this suggests a recency bias (favoring whoever spoke last)."
    )
    print("\n✓ PASSED: Proposer won despite Challenger speaking last - "
          "proves the Arbiter is NOT just favoring the most recent speaker.\n")

    print("======================================================================")
    print("SCENARIO B: Proposer has WEAK arguments, Challenger has STRONG ones")
    print("(Here the Challenger SHOULD win, on merit - not because it spoke last)")
    print("======================================================================")

    verdict_b = run_debate(
        topic="Should we migrate the database to a new provider this quarter?",
        claim="Yes - migrate now.",
        proposer_args=[
            Argument(text="The new provider seems nice.", evidence_strength=0.1),
        ],
        challenger_args=[
            Argument(text="Migration during peak season historically causes 20+ hours "
                           "of downtime based on last year's incident report.", evidence_strength=0.9),
            Argument(text="The current contract has 8 months left with an early-exit "
                           "penalty of $50,000.", evidence_strength=0.85),
        ],
    )

    assert verdict_b.winner == "Challenger", (
        "FAILURE: Arbiter picked the Proposer despite much weaker arguments."
    )
    print("\n✓ PASSED: Challenger won on merit - and critically, this is the SAME "
          "outcome recency bias would have predicted, which is exactly why "
          "Scenario A was necessary to prove the rule is actually about quality.\n")


if __name__ == "__main__":
    verify_no_recency_bias()
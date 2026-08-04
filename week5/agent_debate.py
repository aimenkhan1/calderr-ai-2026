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


class Proposal(BaseModel):
    sender: str = "Proposer"
    topic: str
    claim: str
    arguments: List[Argument]


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





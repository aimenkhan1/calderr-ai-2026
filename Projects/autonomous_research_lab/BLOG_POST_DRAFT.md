# I Built an AI Research Team That Assembles Itself

*How dynamic agent assembly, a real Critic pattern, and a two-pass review
system produce research reports I'd actually trust.*

---

Most multi-agent AI demos use the same trick: hardcode 3-4 agents with
fixed roles, run them in parallel, staple the outputs together, call it a
day. It looks impressive in a demo. It falls apart the moment you ask it
a question the hardcoded roles weren't built for.

I wanted to build something that didn't have that ceiling — a research
system where the **team itself changes shape** depending on the question,
where a **Critic Agent** can genuinely downgrade a finding's confidence
score with a documented reason, and where a **second, independent
reviewer** catches mistakes the first pass introduced. Here's how it
works, and what I learned building it.

## The problem with fixed agent teams

Ask a fixed 4-agent research team "should we use microservices?" and
"what's the regulatory outlook for stablecoins?" and you get the same
four perspectives forced onto two completely different questions. One of
those perspectives is always going to be irrelevant filler, and one
angle the question actually needed is always going to be missing.

## Dynamic assembly: the team adapts to the question

Instead of a fixed roster, the system runs a **Domain Classifier** first,
then a **Dynamic Agent Assembler** that designs 3 to 5 specialist personas
*specifically for that question* — different names, different expertise
descriptions, different sub-questions every single run. A question about
quantum computing timelines might spawn a `HardwareFeasibilityExpert` and
a `TimelineForecaster`; a question about stablecoin regulation spawns a
`ReserveComplianceAnalyst` and a `MarketStructureAnalyst` instead.

Technically, this is where LangGraph's `Send` API earns its keep. Most
LangGraph fan-out examples hardcode the number of parallel branches at
graph-build time. Here, the number of parallel evidence-gathering agents
is decided *at runtime*, by the Assembler's own output:

```python
def _route_to_evidence_agents(state):
    return [
        Send("evidence_agent", {"spec": spec, "domain": domain, "finding_id": f"ev-{i}"})
        for i, spec in enumerate(state["assembly_plan"].specialists, start=1)
    ]
```

Three specialists one run, five the next — the graph itself doesn't know
in advance, and doesn't need to.

## A Critic Agent that actually changes things

A lot of "critic" patterns in agent demos are decorative — the critic
agent writes a paragraph of commentary that never touches the actual
output. I wanted mine to have real teeth: the Critic reviews every
finding against the original hypothesis, and where it finds thin
sourcing or an unjustified confidence score, it **directly rewrites that
finding's confidence** with a recorded reason.

In testing, this consistently caught real problems — a finding citing
contested claims from its own source material while reporting high
confidence, for instance, got its confidence pulled down and the
reasoning preserved in the final report rather than silently smoothed
over.

## Independent peer review, not a rubber stamp

The Critic Agent only ever sees the *raw findings*. It has no idea what
the Synthesis Agent will later do with them. So I added a **Peer Review
Agent** that never sees the raw findings either — it only sees the
*finished report* — and checks specifically for problems introduced
during synthesis: claims that drifted from what the cited finding
actually supported, sections that quietly contradict each other,
conclusions that overstate what the body of the report established.

Two independent passes, each blind to what the other one is checking,
catch different classes of error than either would alone.

## What's under the hood

- **LangGraph** for orchestration — classify → assemble → hypothesize →
  dynamic parallel evidence gathering → critique → synthesize → peer
  review → publish
- **RAG over a seeded document store**, using TF-IDF retrieval — no
  external embedding API required, so the whole pipeline runs offline
  after `pip install`
- **Real tool-calling** — evidence agents can request the full text of a
  source document mid-investigation if a retrieved excerpt isn't enough
- **FastAPI** for a documented REST API, **Streamlit** for a live
  phase-by-phase progress UI, **Docker Compose** for one-command
  deployment
- Every phase boundary is a typed **Pydantic** schema — no raw strings
  passed between agents anywhere in the pipeline

## What I'd build next

The Critic Agent currently only sees the evidence *once* — a genuine
debate round, where a challenged finding's original agent gets a chance
to defend or revise it directly (rather than the Critic unilaterally
rewriting it), would make disagreement even more transparent. I'd also
like to swap the TF-IDF retriever for a dense embedding model once
online, and add a real LangSmith trace export for every run so the
full decision trail is inspectable after the fact, not just in the
final report.

If you're building multi-agent systems and want to see the code, it's
public: [link to your GitHub repo here].

---

*Built as part of Week 5 of an agentic AI engineering internship,
focused on multi-agent orchestration patterns.*

# Multi-Agent Legal Document Reviewer

**Project 5-I-B — Intermediate** · Week 5 Multi-Agent Systems

Four specialist agents review the same contract independently, from four
distinct legal perspectives. A Debate Facilitator then finds genuine
cross-perspective tension and raises challenges; the challenged agent either
upholds or revises its finding. A Judge Agent weighs everything, assigns
final severity, and produces a risk assessment with an explicit dissent log.

## Architecture

```
           ┌── RiskAgent ─────────┐
           ├── ComplianceAgent ───┤
   START ──┼── LiabilityAgent ────┼──► Debate Round ──► JudgeAgent ──► END
           └── ObligationsAgent ──┘         │
                                    Facilitator raises challenges
                                    → targeted agent responds
                                    → finding upheld or revised
```

- **Parallel review**: all 4 specialists review the full contract concurrently,
  with zero shared state (true independent review).
- **Debate round**: the Facilitator doesn't just concatenate findings — it
  looks for real tension between perspectives (e.g. a Liability clause that
  mitigates a Risk finding) and raises a targeted challenge. The originating
  agent then genuinely reconsiders — it may uphold or revise its finding,
  never blindly concede.
- **Judge Agent**: synthesizes post-debate findings into one risk level +
  confidence, and explicitly logs any disagreement that debate did **not**
  resolve, rather than silently picking a side.
- **Failure handling**: a specialist erroring is captured as a typed
  `ErrorReport`; the graph continues with whichever reviews succeeded.
- **Typed messages everywhere**: `AgentReview`, `ClauseFinding`,
  `DebateChallenge`, `JudgeVerdict` — every agent boundary is Pydantic, no
  raw strings/dicts.

## File structure

```
models.py                          # Typed Pydantic schemas
llm_client.py                      # Groq API wrapper, structured JSON + retries
graph.py                           # LangGraph orchestration
main.py                            # CLI + Markdown report writer
agents/
  base_agent.py                    # Shared review() + respond_to_challenge() contract
  risk_agent.py                    # Unfavorable terms, missing protections
  compliance_agent.py              # Regulatory red flags
  liability_agent.py               # Liability exposure, indemnification gaps
  obligations_agent.py             # Extracts obligations + deadlines
  debate_facilitator.py            # Finds cross-perspective tension, raises challenges
  judge_agent.py                   # Final risk verdict + dissent log
sample_contracts/services_agreement.txt   # Sample contract with issues in all 4 domains
requirements.txt
.env.example
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your GROQ_API_KEY (free at console.groq.com)
```

## Run

```bash
python main.py sample_contracts/services_agreement.txt
```

This prints a full report to the terminal **and** writes
`sample_contracts/services_agreement_report.md`.

Run it against any contract:

```bash
python main.py path/to/your_contract.txt
```

## Agent Roles

| Agent | Domain | Out of scope |
|---|---|---|
| **RiskAgent** | One-sided terms, missing protections, unfavorable notice/renewal terms | Regulatory, liability mechanics, deadlines |
| **ComplianceAgent** | Data privacy, worker classification, export/sanctions, audit rights | Commercial risk, liability, deadlines |
| **LiabilityAgent** | Indemnification, liability caps, insurance, warranty disclaimers | Regulatory, general commercial risk, deadlines |
| **ObligationsAgent** | Extracts every obligation + deadline, severity = how easy it is to breach | Risk, regulatory, liability judgments |
| **DebateFacilitator** | Finds real cross-perspective tension, raises targeted challenges | Doesn't review the contract itself |
| **JudgeAgent** | Final risk level, confidence, dissent log | Doesn't originate findings |

## Evaluation criteria checklist (from the project spec)

- [x] All 4 review agents process the same document in parallel
- [x] Debate round produces at least one changed finding (verified in testing —
      severity downgraded from MAJOR → MODERATE with a documented reason)
- [x] Judge Agent report includes severity scores and dissent notes
- [x] Markdown report generated per run
- [ ] Streamlit UI showing clause-level annotations — not included in this
      version; the CLI + Markdown report cover the core pattern. Add a
      `streamlit_app.py` that calls `run_review()` and renders `result` if
      you want the live UI for your demo.
- [x] Three sample contracts — one is included (`services_agreement.txt`);
      add 2 more of your own for the full portfolio deliverable.

## Verified behavior (tested with mocked LLM calls)

- ✅ All 4 specialists run independently and produce typed findings
- ✅ Debate Facilitator raises a substantive, targeted challenge (not
      artificial disagreement)
- ✅ Challenged agent genuinely revises its finding (severity + description
      both updated, with a recorded reasoning trail)
- ✅ Judge Agent synthesizes a final verdict incorporating the debate outcome
- ✅ One agent failing does not crash the run — Judge still produces a
      verdict from the surviving reviews

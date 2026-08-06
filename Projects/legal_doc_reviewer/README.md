# ⚖️ Multi-Agent Legal Document Reviewer

A CLI agent team that reviews a contract from **four independent legal
perspectives in parallel**, runs a structured debate round to challenge
weak findings, and uses Groq to render a final judged risk verdict — saved
as a Markdown report.

---

## 📸 What It Does

Give it a contract file, and the agent team will:
- Review the contract from 4 independent angles **at the same time**
  (Risk, Compliance, Liability, Obligations)
- Call all 4 specialist agents **in parallel**, not sequentially
- Run one structured **debate round** where a Facilitator challenges
  findings it finds contestable across perspectives
- Let the challenged agent uphold or **revise** its own finding based on
  the challenge — real reconsideration, not forced agreement
- Have a Judge Agent weigh everything and hand down a final risk level,
  confidence score, and a dissent log for anything left unresolved
- Save the final report as a Markdown `.md` file next to the contract

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM Backend | Groq (llama-3.3-70b-versatile) |
| Orchestration | LangGraph (parallel fan-out → debate → judge) |
| Typed Messages | Pydantic |
| Environment | python-dotenv |
| Language | Python 3.11+ |

---

## 📂 Project Structure
legal_doc_reviewer/

│

├── models.py                  # Typed Pydantic schemas

├── llm_client.py               # Groq API wrapper (structured JSON + retries)

├── graph.py                    # LangGraph orchestration

├── main.py                     # CLI + Markdown report writer

├── requirements.txt            # Dependencies

├── README.md                   # This file

├── agents/

│   ├── base_agent.py           # Shared review + debate-response contract

│   ├── risk_agent.py           # Unfavorable terms, missing protections

│   ├── compliance_agent.py     # Regulatory red flags

│   ├── liability_agent.py      # Liability exposure, indemnification gaps

│   ├── obligations_agent.py    # Extracts obligations + deadlines

│   ├── debate_facilitator.py   # Raises cross-perspective challenges

│   └── judge_agent.py          # Final risk verdict + dissent log

└── sample_contracts/

    └── services_agreement.txt  # Sample contract with issues in all 4 domains

---

## 📰 Report Sections

| Section | Emoji | Source |
|---------|-------|--------|
| Specialist Findings | 🔍 | RiskAgent, ComplianceAgent, LiabilityAgent, ObligationsAgent |
| Debate Transcript | ⚡ | DebateFacilitator + challenged agent's response |
| Revised Findings | 🔁 | Any finding upheld or changed during debate |
| Final Risk Verdict | ⚖️ | JudgeAgent |
| Dissent Log | 🗣 | JudgeAgent (unresolved disagreements) |

---

## 🏗 Architecture
Contract Input (.txt file)

↓

Scheduler fires 4 specialist agents in parallel

↓

RiskAgent ‖ ComplianceAgent ‖ LiabilityAgent ‖ ObligationsAgent (all running at once)

↓

Debate Facilitator (finds cross-perspective tension, raises challenges)

↓

Challenged Agent Responds (upholds or revises its own finding)

↓

Judge Agent (weighs everything, assigns final risk level + confidence)

↓

Formatted Markdown Report (.md file saved to disk)

↓

Printed to terminal + saved next to the contract

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

cdddddd
```
Get a free Groq key at console.groq.com

### 3. Run the agent team
```bash
python main.py sample_contracts/services_agreement.txt
```

---

## ⌨️ CLI Usage

| Argument | Description |
|----------|--------------|
| `<path_to_contract.txt>` | Required — path to the contract file to review |

---

## 💬 Example Run

```
========================================================================
MULTI-AGENT LEGAL DOCUMENT REVIEW
========================================================================

4 specialist(s) reported in 8.2s

── RiskAgent ─────────────────────────────
   Overall: One-sided termination and payment terms favor Vendor.
   🔴 [risk-1] [MAJOR] One-sided termination
       Vendor may terminate at will with 5 days notice; Client needs
       60 days plus uncured breach.

── LiabilityAgent ────────────────────────
   Overall: Indemnification is mutual and reasonably balanced.
   🟡 [liab-1] [MINOR] Mutual indemnification
       This partially offsets the termination risk flagged elsewhere.

========================================================================
DEBATE ROUND
========================================================================
Challenge on [risk-1] (targeting RiskAgent):
   Rationale: Mutual indemnification may reduce the practical impact
   of the one-sided termination clause.
   Resolution: revised

🔁 1 finding(s) changed as a result of debate:
   [risk-1] now MODERATE — Conceded partial mitigation from
   the indemnification clause.

========================================================================
JUDGE VERDICT
========================================================================
🟡 Overall risk: MEDIUM  (confidence=0.8)

Summary:
Moderate termination risk, partially mitigated by mutual
indemnification; one hard obligation deadline to track closely.

✅ No unresolved dissent.
========================================================================

📄 Markdown report written to sample_contracts/services_agreement_report.md
```

---

## 📝 Notes

- Risk, Compliance, Liability, and Obligations reviews all run in parallel
  via LangGraph fan-out, so total wait time is close to the *slowest*
  single agent, not the sum of all four
- Each specialist has independent error handling — one agent failing
  never crashes the run; the Judge synthesizes from whichever reviews
  succeeded
- The debate round only raises challenges with real substance — it will
  not manufacture disagreement just to have something to show
- Every run writes a fresh `<contract-name>_report.md` next to the
  input file

---

## 👩‍💻 Built By

Aiman Nadeem Khan
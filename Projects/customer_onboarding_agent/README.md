# 🧾 Onboard Flow — Customer Onboarding Agent

A LangGraph-powered onboarding workflow that validates new applicants,
automatically routes large accounts to a human reviewer, pauses execution
mid-workflow using real interrupt/resume, and persists state to SQLite so
a paused application can be picked up again minutes, hours, or days later.

---

## What It Does

Feed it an applicant's signup details, and Onboard Flow will:

- **Collect** — Take in applicant name, email, company, seats requested, and monthly value
- **Validate** — Check the submission is complete and well-formed before proceeding
- **Categorize** — Classify the account as `standard` or `large` based on seats/value thresholds
- **Route** — Standard accounts auto-approve; large accounts pause for a human decision
- **Pause & Resume** — Large accounts genuinely halt execution via `interrupt()`, checkpointed
  to disk, resumable at any later time with the same `thread_id`
- **Create** — Generate an account ID once approved
- **Notify** — Simulate sending a welcome email
- **Schedule** — Simulate scheduling a follow-up check-in

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph `StateGraph` |
| Human-in-the-Loop | `interrupt()` / `Command(resume=...)` |
| Persistence | `SqliteSaver` (survives process restarts) |
| Batch Demo | In-memory `MemorySaver` + simulated reviewer decisions |
| Language | Python 3.11+ |

---

## Project Structure

```
customer_onboarding_agent/
│
├── main.py       # Main graph + interactive CLI (new / resume)
├── simulation.py     # Runs all sample applicants end-to-end, auto-simulates review
├── graph.py     # Produces graph.png
├── sample_applicants.json       # 6 test applicants (standard, large, and one invalid)
├── requirements.txt
├── graph.png                    # Generated graph visualization
├── onboarding_checkpoints.sqlite # Generated on first run - persisted graph state
└── README.md                    # This file
```

---

## Architecture

```
                         collect_info
                              │
                          validate
                              │
                 ┌────────────┴────────────┐
              invalid                    valid
                 │                         │
          reject_invalid           categorize_account
                 │                         │
                END              ┌─────────┴─────────┐
                              standard              large
                                 │                     │
                          auto_approve           human_review
                                 │                (interrupt)
                                 └──────────┬──────────┘
                                     ┌──────┴──────┐
                                 approved        rejected
                                    │                │
                              create_account   reject_account
                                    │                │
                                notify             END
                                    │
                            schedule_followup
                                    │
                                   END
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Try it interactively (single applicant, real pause/resume)
```bash
python main.py
```
- Type `new`, fill in an applicant's details
- If it's a **large account**, the program prints a `thread_id` and pauses
- Close the terminal entirely, come back later, and run:
  ```bash
  python main.py
  ```
  Type `resume`, paste the same `thread_id`, and give a decision (`approve`/`reject`) —
  execution continues exactly where it stopped.

### 3. Run the full batch demo (all 6 sample applicants, no manual input needed)
```bash
python demo_batch_simulation.py
```
This exercises every path — standard auto-approval, large-account human review
(simulated automatically), and the invalid-submission rejection path — in one run.

### 4. Generate the graph visualization
```bash
python graph.py
```
Produces `graph.png`, a rendered diagram of the full workflow.

---

## Features

| Feature | Description |
|---------|-------------|
| 🔀 Conditional Routing | Standard vs. large account paths, and valid vs. invalid submissions |
| ⏸️ Real Human-in-the-Loop | Genuine `interrupt()` pause, not a simulated branch |
| 💾 True Persistence | `SqliteSaver` — paused applications survive closing the program entirely |
| 📝 Full Audit Trail | Every node appends to an accumulating `log`, viewable at the end of any run |
| 🧪 Batch Simulation | Run all test applicants end-to-end without manual approval clicks |
| 🛡️ Input Validation | Bad submissions are rejected before ever reaching account creation |

---

## Sample Output

```
[1] Ayesha Raza @ SmallCo Solutions (seats=5, value=$250.0)
  -> Onboarding complete. Account: ACC-3F9A21BC

[3] Carla Mendes @ BigCorp Industries (seats=120, value=$8500.0)
  -> paused for human review, simulated decision: 'approve'
  -> Onboarding complete. Account: ACC-7C21E4F0

[5] Fatima Sheikh @ Suspicious Inc (seats=3, value=$9000.0)
  -> paused for human review, simulated decision: 'reject'
  -> Rejected during human review.

[6]  @ Broken Signup Co (seats=0, value=$0)
  -> Rejected at validation: Invalid email format. Missing applicant name. Seats requested must be greater than 0.
```

---

## Notes

- Large-account thresholds (`50` seats or `$5,000`/month) are configurable
  constants at the top of `main.py` — tune them without touching
  the graph structure
- `thread_id` is the key to resumability: as long as it's reused, `SqliteSaver`
  reconstructs the exact paused state, even across separate program runs
- The batch demo uses a simple heuristic (`value_per_seat > 200 → reject`) to
  simulate a human reviewer flagging suspicious pricing — easy to swap for a
  real reviewer prompt or dashboard later
- State transitions are intentionally granular (`collect_info` and `validate`
  as separate nodes) so each stage's audit log entry is independently traceable

---

## Built By
Aiman Nadeem Khan
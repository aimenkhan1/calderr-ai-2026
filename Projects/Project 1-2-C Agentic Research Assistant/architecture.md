# Nova — Architecture Diagram

```
┌─────────────────────────────────────────────┐
│              STREAMLIT UI                   │
│  User types question → clicks Start Research│
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│              VALIDATOR                      │
│  Checks if input is a valid research topic  │
│  INVALID → Nova asks for proper question    │
│  VALID   → continues to Planner Agent       │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│           PLANNER AGENT (Groq call 1)       │
│  Input:  research question                  │
│  Output: main_topic + 3 subtopics + goal    │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│        RESEARCH LOOP (Groq calls 2-4)       │
│                                             │
│  Subtopic 1 → findings + confidence + points│
│  Subtopic 2 → findings + confidence + points│
│  Subtopic 3 → findings + confidence + points│
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│         SYNTHESIS AGENT (Groq call 5)       │
│  Input:  all research findings              │
│  Output: summary + conclusion               │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│           REPORT FORMATTER                  │
│  Pydantic validates all structured data     │
│  Calculates overall confidence score        │
│  Formats into ResearchReport object         │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│           STREAMLIT UI DISPLAY              │
│  Shows: Research Plan                       │
│  Shows: Metrics (confidence, sections, kp)  │
│  Shows: Each section with confidence bar    │
│  Shows: Sources for each section            │
│  Shows: Conclusion                          │
│  Offers: Download report as .txt            │
│  Saves:  To session history                 │
└─────────────────────────────────────────────┘

Total Groq API calls per research: 5
Models used: llama-3.1-8b-instant
```
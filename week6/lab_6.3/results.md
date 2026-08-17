# Lab 6.3 — GraphRAG Evaluation Results

## Category Summary (accuracy per category, out of 5 questions)

| Category | Vector-only | Graph-only | Hybrid | Expected Winner |
|---|---|---|---|---|
| factual | 100% | 100% | 100% | vector |
| relational | 100% | 100% | 100% | graph |
| complex | 20% | 60% | 100% | hybrid |

## Average Context Size (lower = more precise, less noise)

| Category | Vector-only | Graph-only | Hybrid |
|---|---|---|---|
| factual | 5.0 items | 9.2 items | 16.2 items |
| relational | 5.0 items | 11.8 items | 21.0 items |
| complex | 5.0 items | 11.4 items | 23.0 items |

## Per-Question Results

| ID | Category | Question | Vector | Graph | Hybrid | Expected Route | Routed To | Router OK? |
|---|---|---|---|---|---|---|---|---|
| F1 | factual | In what year was NimbusCloud founded? | ✅ | ✅ | ✅ | vector | vector | ✅ |
| F2 | factual | Where is PixelForge Studios based? | ✅ | ✅ | ✅ | vector | vector | ✅ |
| F3 | factual | What product did Verdant Energy Co create? | ✅ | ✅ | ✅ | vector | vector | ✅ |
| F4 | factual | In what city is Solstice AI based? | ✅ | ✅ | ✅ | vector | vector | ✅ |
| F5 | factual | What year did Tomasz Nowak found Orbital Dynamics? | ✅ | ✅ | ✅ | vector | vector | ✅ |
| R1 | relational | Who co-founded QuantumLeap Robotics alongside Marcus Bell? | ✅ | ✅ | ✅ | graph | graph | ✅ |
| R2 | relational | Which company did Marcus Bell work at before founding QuantumLeap Robotics? | ✅ | ✅ | ✅ | graph | graph | ✅ |
| R3 | relational | What company did NimbusCloud acquire? | ✅ | ✅ | ✅ | graph | graph | ✅ |
| R4 | relational | Which person worked at both PixelForge Studios and NimbusCloud? | ✅ | ✅ | ✅ | graph | graph | ✅ |
| R5 | relational | Which two companies partnered to build the Solstice Assistant? | ✅ | ✅ | ✅ | graph | graph | ✅ |
| C1 | complex | What product did the company founded by the person who previously worked at Verdant Energy Co go on to create? | ❌ | ❌ | ✅ | hybrid | hybrid | ✅ |
| C2 | complex | In what city is the company based that Marcus Bell founded after leaving the company that created the PixelForge Engine? | ❌ | ✅ | ✅ | hybrid | hybrid | ✅ |
| C3 | complex | Who is the CTO of the company that acquired the company Marcus Bell originally founded? | ❌ | ❌ | ✅ | hybrid | hybrid | ✅ |
| C4 | complex | What navigation product does the company that partnered with QuantumLeap Robotics in 2022 make? | ✅ | ✅ | ✅ | hybrid | hybrid | ✅ |
| C5 | complex | Which product was created through a partnership involving the company where Priya Rao worked before co-founding QuantumLeap Robotics? | ❌ | ✅ | ✅ | hybrid | hybrid | ✅ |

**Router accuracy: 15/15**

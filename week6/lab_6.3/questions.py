"""
questions.py

The 15-question evaluation set for Lab 6.3, split into three categories of
5 questions each:

  - factual    -> a single, standalone fact in which VECTOR search should win
  - relational -> a direct connection between two entities in which GRAPH should win
  - complex    -> needs a relationship AND a specific fact chained together;
                  HYBRID should win

"""

QUESTIONS = [
    # ---------------- FACTUAL (vector should win) ---------------- #
    {
        "id": "F1",
        "category": "factual",
        "question": "In what year was NimbusCloud founded?",
        "expected_route": "vector",
        "answer_keywords": ["2015"],
        "required_fact_groups": [['NimbusCloud', '2015']],
    },
    {
        "id": "F2",
        "category": "factual",
        "question": "Where is PixelForge Studios based?",
        "expected_route": "vector",
        "answer_keywords": ["Seattle"],
        "required_fact_groups": [['PixelForge Studios', 'Seattle']],
    },
    {
        "id": "F3",
        "category": "factual",
        "question": "What product did Verdant Energy Co create?",
        "expected_route": "vector",
        "answer_keywords": ["Verdant Grid"],
        "required_fact_groups": [['Verdant Energy Co', 'Verdant Grid']],
    },
    {
        "id": "F4",
        "category": "factual",
        "question": "In what city is Solstice AI based?",
        "expected_route": "vector",
        "answer_keywords": ["Boston"],
        "required_fact_groups": [['Solstice AI', 'Boston']],
    },
    {
        "id": "F5",
        "category": "factual",
        "question": "What year did Tomasz Nowak found Orbital Dynamics?",
        "expected_route": "vector",
        "answer_keywords": ["2019"],
        "required_fact_groups": [['Orbital Dynamics', '2019']],
    },
    # ---------------- RELATIONAL (graph should win) ---------------- #
    {
        "id": "R1",
        "category": "relational",
        "question": "Who co-founded QuantumLeap Robotics alongside Marcus Bell?",
        "expected_route": "graph",
        "answer_keywords": ["Priya Rao"],
        "required_fact_groups": [['Priya Rao', 'QuantumLeap Robotics']],
    },
    {
        "id": "R2",
        "category": "relational",
        "question": "Which company did Marcus Bell work at before founding QuantumLeap Robotics?",
        "expected_route": "graph",
        "answer_keywords": ["PixelForge Studios"],
        "required_fact_groups": [['Marcus Bell', 'PixelForge Studios']],
    },
    {
        "id": "R3",
        "category": "relational",
        "question": "What company did NimbusCloud acquire?",
        "expected_route": "graph",
        "answer_keywords": ["PixelForge Studios"],
        "required_fact_groups": [['NimbusCloud', 'PixelForge Studios']],
    },
    {
        "id": "R4",
        "category": "relational",
        "question": "Which person worked at both PixelForge Studios and NimbusCloud?",
        "expected_route": "graph",
        "answer_keywords": ["Kenji Watanabe"],
        "required_fact_groups": [['Kenji Watanabe', 'PixelForge Studios'], ['Kenji Watanabe', 'NimbusCloud']],
    },
    {
        "id": "R5",
        "category": "relational",
        "question": "Which two companies partnered to build the Solstice Assistant?",
        "expected_route": "graph",
        "answer_keywords": ["Verdant Energy Co", "Solstice AI"],
        "required_fact_groups": [['Verdant Energy Co', 'Solstice Assistant'], ['Solstice AI', 'Solstice Assistant']],
    },
    # ---------------- COMPLEX (hybrid should win) ---------------- #
    {
        "id": "C1",
        "category": "complex",
        "question": (
            "What product did the company founded by the person who previously "
            "worked at Verdant Energy Co go on to create?"
        ),
        "expected_route": "hybrid",
        "answer_keywords": ["LeapBot X1"],
        "required_fact_groups": [['Priya Rao', 'Verdant Energy Co'], ['Priya Rao', 'QuantumLeap Robotics'], ['QuantumLeap Robotics', 'LeapBot X1']],
    },
    {
        "id": "C2",
        "category": "complex",
        "question": (
            "In what city is the company based that Marcus Bell founded after "
            "leaving the company that created the PixelForge Engine?"
        ),
        "expected_route": "hybrid",
        "answer_keywords": ["Denver"],
        "required_fact_groups": [['PixelForge Studios', 'PixelForge Engine'], ['QuantumLeap Robotics', 'Denver']],
    },
    {
        "id": "C3",
        "category": "complex",
        "question": "Who is the CTO of the company that acquired the company Marcus Bell originally founded?",
        "expected_route": "hybrid",
        "answer_keywords": ["Kenji Watanabe"],
        "required_fact_groups": [['Marcus Bell', 'PixelForge Studios'], ['NimbusCloud', 'PixelForge Studios'], ['Kenji Watanabe', 'NimbusCloud']],
    },
    {
        "id": "C4",
        "category": "complex",
        "question": (
            "What navigation product does the company that partnered with "
            "QuantumLeap Robotics in 2022 make?"
        ),
        "expected_route": "hybrid",
        "answer_keywords": ["Orbital Nav"],
        "required_fact_groups": [['Orbital Dynamics', 'QuantumLeap Robotics'], ['Orbital Dynamics', 'Orbital Nav']],
    },
    {
        "id": "C5",
        "category": "complex",
        "question": (
            "Which product was created through a partnership involving the "
            "company where Priya Rao worked before co-founding QuantumLeap Robotics?"
        ),
        "expected_route": "hybrid",
        "answer_keywords": ["Solstice Assistant"],
        "required_fact_groups": [['Priya Rao', 'Verdant Energy Co'], ['Verdant Energy Co', 'Solstice Assistant']],
    },
]

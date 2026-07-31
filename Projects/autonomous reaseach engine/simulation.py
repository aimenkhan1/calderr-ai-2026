"""
Batch runner: produces 5 research reports on complex topics.
Satisfies the project deliverable of "5 research reports on complex topics."
Each report is saved to reports/<n>_<slug>.md
"""

import os
import re as regex
from main import graph

TOPICS = [
    "What are the long-term economic effects of remote work on urban commercial real estate?",
    "How does gut microbiome diversity influence mental health and mood regulation?",
    "What are the main technical and ethical challenges in deploying autonomous vehicles at scale?",
    "How is generative AI reshaping entry-level hiring practices in the tech industry?",
    "What strategies are proving most effective for reducing plastic waste in coastal cities?",
]


def slugify(text: str) -> str:
    text = text.lower()
    text = regex.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:50]


def run_topic(index: int, topic: str):
    print(f"\n[{index}/5] Researching: {topic}")

    result = graph.invoke({
        "topic": topic,
        "research_plan": [],
        "research_findings": [],
        "synthesis": "",
        "gaps": [],
        "gap_analysis_complete": False,
        "iteration": 0,
        "final_report": "",
    })

    print(f"  -> {result['iteration']} research round(s), "
          f"{len(result['research_findings'])} findings, "
          f"gap analysis {'complete' if result['gap_analysis_complete'] else 'reached max iterations'}")

    os.makedirs("reports", exist_ok=True)
    filename = f"reports/{index}_{slugify(topic)}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Research Report {index}\n\n")
        f.write(f"**Topic:** {topic}\n\n")
        f.write(f"**Research rounds:** {result['iteration']}\n\n")
        f.write(f"**Findings gathered:** {len(result['research_findings'])}\n\n")
        f.write("\n\n")
        f.write(result["final_report"])
        f.write("\n\n---\n\n## Raw Research Findings\n\n")
        for finding in result["research_findings"]:
            f.write(f"{finding}\n\n")

    print(f"  -> saved to {filename}")
    return result


if __name__ == "__main__":
    print(f"Running {len(TOPICS)} research topics through the engine...")
    all_results = []
    for i, topic in enumerate(TOPICS, start=1):
        all_results.append(run_topic(i, topic))

    print()
    print("SUMMARY")
    print()
    for i, (topic, result) in enumerate(zip(TOPICS, all_results), start=1):
        print(f"[{i}] rounds={result['iteration']} "
              f"findings={len(result['research_findings'])} "
              f"gap_complete={result['gap_analysis_complete']}")
        print(f"    {topic[:70]}...")
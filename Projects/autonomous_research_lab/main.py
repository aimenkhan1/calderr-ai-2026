"""
CLI entry point for the Autonomous AI Research Lab.

Usage:
    python main.py "Your research question here"
"""

import sys
import time

from agents.report_publisher import ReportPublisherAgent
from graph import run_research


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "Your research question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Researching: {question}\n")

    start = time.time()
    state = run_research(question)
    elapsed = time.time() - start

    if state["errors"]:
        print("⚠️  Some agents failed (pipeline degraded gracefully):")
        for e in state["errors"]:
            print(f"   - {e.agent_name}: {e.error_type} — {e.message}")
        print()

    report = state["report"]
    if report is None:
        print("Pipeline failed to produce a report.")
        sys.exit(1)

    markdown = ReportPublisherAgent.to_markdown(report)
    print(markdown)
    print(f"\n\n(completed in {elapsed:.1f}s)")

    out_path = "research_report.md"
    with open(out_path, "w") as f:
        f.write(markdown)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

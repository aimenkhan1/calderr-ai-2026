"""
main.py-Command-line interface for the Multi-Agent Legal Document Reviewer.

Loads a contract, runs the review pipeline, displays the results,
and exports a Markdown report.
"""

import sys
import time
from graph import run_review
from models import Severity, RiskLevel

_SEVERITY_ICON = {
    Severity.TRIVIAL: "⚪", Severity.MINOR: "🟡", Severity.MODERATE: "🟠",Severity.MAJOR: "🔴", Severity.CRITICAL: "🔥",}
_RISK_ICON = {
    RiskLevel.LOW: "✅", RiskLevel.MEDIUM: "🟡", RiskLevel.HIGH: "🔴", RiskLevel.CRITICAL: "🔥",}


def print_report(result, elapsed_s: float) -> None:
    print("=================================")
    print("MULTI-AGENT LEGAL DOCUMENT REVIEW")
    print("=================================")

    if result["errors"]:
        print("\n  AGENT FAILURES (degraded gracefully):")
        for err in result["errors"]:
            print(f"   - {err.agent_name}: {err.error_type} — {err.message}")

    print(f"\n{len(result['reviews'])} specialist(s) reported in {elapsed_s:.1f}s\n")

    for r in result["reviews"]:
        print(f"── {r.agent_name} " + "─" * max(1, 45 - len(r.agent_name)))
        print(f"   Overall: {r.overall_assessment}  (confidence={r.confidence})")
        for f in r.findings:
            icon = _SEVERITY_ICON.get(f.severity, "•")
            print(f"   {icon} [{f.finding_id}] [{f.severity.name}] {f.title}")
            print(f"       {f.description}")
            if f.suggestion:
                print(f"       → Suggestion: {f.suggestion}")
        print()

    print("============")
    print("DEBATE ROUND")
    print("============")
    if not result["challenges"]:
        print("No challenges raised — all specialists' findings stood uncontested.\n")
    for c in result["challenges"]:
        print(f"Challenge on [{c.target_finding_id}] (targeting {c.target_agent}):")
        print(f"   Rationale: {c.challenging_rationale}")
        print(f"   Resolution: {c.resolution}\n")

    revised = [f for f in result["final_findings"] if f.revised]
    if revised:
        print(f" {len(revised)} finding(s) changed as a result of debate:")
        for f in revised:
            print(f"   [{f.finding_id}] now {f.severity.name} — {f.revision_note}")
        print()

    verdict = result["verdict"]
    print("=============")
    print("JUDGE VERDICT")
    print("=============")
    print(f"{_RISK_ICON.get(verdict.overall_risk_level, '')} Overall risk: "
          f"{verdict.overall_risk_level.value.upper()}  (confidence={verdict.confidence})")
    print(f"\nSummary:\n{verdict.summary}")

    if verdict.dissent_log:
        print("\n DISSENT LOG (unresolved disagreements):")
        for d in verdict.dissent_log:
            print(f"   - {d}")
    else:
        print("\n No unresolved dissent.")
    print("=============================")


def write_markdown_report(result, out_path: str) -> None:
    verdict = result["verdict"]
    lines = [
        "# Legal Document Review Report\n",
        f"**Overall risk level:** {verdict.overall_risk_level.value.upper()}  ",
        f"**Confidence:** {verdict.confidence}\n",
        f"## Summary\n{verdict.summary}\n",
        "## Findings by Specialist\n",
    ]
    for r in result["reviews"]:
        lines.append(f"### {r.agent_name}")
        lines.append(f"_{r.overall_assessment}_ (confidence={r.confidence})\n")
        for f in r.findings:
            tag = " *(revised during debate)*" if f.revised else ""
            lines.append(f"- **[{f.severity.name}] {f.title}**{tag}  \n  {f.description}")
            if f.revision_note:
                lines.append(f"  \n  _Revision note: {f.revision_note}_")
        lines.append("")

    lines.append("## Debate Transcript\n")
    if result["challenges"]:
        for c in result["challenges"]:
            lines.append(f"- Challenge on `{c.target_finding_id}` ({c.target_agent}): "
                          f"{c.challenging_rationale} → **{c.resolution}**")
    else:
        lines.append("_No challenges raised._")

    lines.append("\n## Dissent Log\n")
    if verdict.dissent_log:
        for d in verdict.dissent_log:
            lines.append(f"- {d}")
    else:
        lines.append("_No unresolved dissent._")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📄 Markdown report written to {out_path}")


def main():
    filepath = input("Enter the path to the legal document (.txt): ").strip()

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            contract_text = f.read()
    except FileNotFoundError:
        print("❌ File not found.")
        return

    start = time.time()
    result = run_review(contract_text)
    elapsed = time.time() - start

    print_report(result, elapsed)

    out_path = filepath.rsplit(".", 1)[0] + "_report.md"
    write_markdown_report(result, out_path)

if __name__ == "__main__":
    main()

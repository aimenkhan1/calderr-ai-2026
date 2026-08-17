"""
evaluate.py

Runs the full Lab 6.3 evaluation:

  1. For every question, runs all THREE retrieval modes (vector-only,
     graph-only, hybrid) and scores each one using `required_fact_groups`
     (see questions.py for why this scoring method was chosen over naive
     keyword matching).
  2. Runs the query ROUTER on every question and checks whether it picked
     the expected mode.
  3. Prints a structured comparison table (accuracy by category x mode).
  4. Checks the two validation criteria from the lab spec:
       a) Hybrid must outperform both vector-only and graph-only on the
          complex-question category.
       b) The router must correctly classify at least 12 of 15 questions.

Run it with:  python3 evaluate.py
"""

from __future__ import annotations
from typing import List
from pipeline import run_vector_only, run_graph_only, run_hybrid
from router import route
from questions import QUESTIONS


#scoring

def fact_group_found(context: List[dict], keyword_group: List[str]) -> bool:
    for item in context:
        text_lower = item["text"].lower()
        if all(kw.lower() in text_lower for kw in keyword_group):
            return True
    return False


def is_correct(context: List[dict], required_fact_groups: List[List[str]]) -> bool:
    return all(fact_group_found(context, group) for group in required_fact_groups)


#main

def run_evaluation() -> dict:
    per_question_results = []

    for q in QUESTIONS:
        question_text = q["question"]

        vector_result = run_vector_only(question_text)
        graph_result = run_graph_only(question_text)
        hybrid_result = run_hybrid(question_text)

        routed_mode = route(question_text)

        per_question_results.append(
            {
                "id": q["id"],
                "category": q["category"],
                "question": question_text,
                "expected_route": q["expected_route"],
                "routed_mode": routed_mode,
                "router_correct": routed_mode == q["expected_route"],
                "vector_correct": is_correct(vector_result["context"], q["required_fact_groups"]),
                "graph_correct": is_correct(graph_result["context"], q["required_fact_groups"]),
                "hybrid_correct": is_correct(hybrid_result["context"], q["required_fact_groups"]),
                "vector_context_size": len(vector_result["context"]),
                "graph_context_size": len(graph_result["context"]),
                "hybrid_context_size": len(hybrid_result["context"]),
                "vector_answer": vector_result["answer"],
                "graph_answer": graph_result["answer"],
                "hybrid_answer": hybrid_result["answer"],
            }
        )

    return per_question_results

#

def print_per_question_table(results: List[dict]) -> None:
    print()
    print("PER-QUESTION RESULTS  (V = vector-only, G = graph-only, H = hybrid)")
    print()
    header = f"{'ID':4s} {'Category':10s} {'V':5s} {'G':5s} {'H':5s} {'Router->':10s} {'Correct?':8s}"
    print(header)
    print()
    for r in results:
        def mark(b: bool) -> str:
            return "✅" if b else "❌"

        router_note = "OK" if r["router_correct"] else f"got {r['routed_mode']}"
        print(
            f"{r['id']:4s} {r['category']:10s} "
            f"{mark(r['vector_correct']):5s}{mark(r['graph_correct']):5s}{mark(r['hybrid_correct']):5s} "
            f"{r['expected_route']:10s} {router_note:8s}"
        )


def print_category_summary_table(results: List[dict]) -> None:
    categories = ["factual", "relational", "complex"]
    expected_winner = {"factual": "vector", "relational": "graph", "complex": "hybrid"}

    print()
    print("CATEGORY SUMMARY  (accuracy = correct questions / 5, per category)")
    print()
    print(f"{'Category':12s} {'Vector-only':13s} {'Graph-only':13s} {'Hybrid':13s} {'Expected winner':16s}")
    print()

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)
        vector_acc = sum(r["vector_correct"] for r in cat_results) / n
        graph_acc = sum(r["graph_correct"] for r in cat_results) / n
        hybrid_acc = sum(r["hybrid_correct"] for r in cat_results) / n
        print(
            f"{cat:12s} {vector_acc:>6.0%}{'':7s} {graph_acc:>6.0%}{'':7s} "
            f"{hybrid_acc:>6.0%}{'':7s} {expected_winner[cat]:16s}"
        )


    print()
    print("AVG CONTEXT SIZE  (lower = more precise/concise -- less noise for the LLM to sift through)")
    print()
    print(f"{'Category':12s} {'Vector-only':13s} {'Graph-only':13s} {'Hybrid':13s}")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)
        v_size = sum(r["vector_context_size"] for r in cat_results) / n
        g_size = sum(r["graph_context_size"] for r in cat_results) / n
        h_size = sum(r["hybrid_context_size"] for r in cat_results) / n
        print(f"{cat:12s} {v_size:>6.1f} items  {g_size:>6.1f} items  {h_size:>6.1f} items")


def print_router_summary(results: List[dict]) -> None:
    correct = sum(r["router_correct"] for r in results)
    total = len(results)
    print()
    print(f"ROUTER ACCURACY: {correct}/{total} questions correctly classified")
    print()
    for r in results:
        if not r["router_correct"]:
            print(f"  MISCLASSIFIED: {r['id']} expected={r['expected_route']} got={r['routed_mode']}")


def check_validation_criteria(results: List[dict]) -> bool:
    print()
    print("VALIDATION CRITERIA")
    print()

    all_passed = True

    # Criterion A: hybrid must beat both vector-only and graph-only on complex questions.
    complex_results = [r for r in results if r["category"] == "complex"]
    n = len(complex_results)
    vector_acc = sum(r["vector_correct"] for r in complex_results) / n
    graph_acc = sum(r["graph_correct"] for r in complex_results) / n
    hybrid_acc = sum(r["hybrid_correct"] for r in complex_results) / n

    criterion_a_passed = hybrid_acc > vector_acc and hybrid_acc > graph_acc
    status_a = "PASS" if criterion_a_passed else "FAIL"
    print(
        f"[{status_a}] Criterion A: Hybrid ({hybrid_acc:.0%}) must outperform "
        f"vector-only ({vector_acc:.0%}) and graph-only ({graph_acc:.0%}) on complex questions."
    )
    all_passed &= criterion_a_passed

    # Criterion B: router must classify at least 12/15 questions correctly.
    router_correct = sum(r["router_correct"] for r in results)
    criterion_b_passed = router_correct >= 12
    status_b = "PASS" if criterion_b_passed else "FAIL"
    print(
        f"[{status_b}] Criterion B: Router correctly classified {router_correct}/15 "
        f"questions (needs >= 12/15)."
    )
    all_passed &= criterion_b_passed

    print()
    print("OVERALL:", "ALL CRITERIA PASSED ✅" if all_passed else "SOME CRITERIA FAILED ❌")
    return all_passed


def save_markdown_report(results: List[dict], path: str = "results.md") -> None:
    categories = ["factual", "relational", "complex"]
    expected_winner = {"factual": "vector", "relational": "graph", "complex": "hybrid"}

    lines = ["# Lab 6.3 — GraphRAG Evaluation Results\n"]

    lines.append("## Category Summary (accuracy per category, out of 5 questions)\n")
    lines.append("| Category | Vector-only | Graph-only | Hybrid | Expected Winner |")
    lines.append("|---|---|---|---|---|")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)
        vector_acc = sum(r["vector_correct"] for r in cat_results) / n
        graph_acc = sum(r["graph_correct"] for r in cat_results) / n
        hybrid_acc = sum(r["hybrid_correct"] for r in cat_results) / n
        lines.append(
            f"| {cat} | {vector_acc:.0%} | {graph_acc:.0%} | {hybrid_acc:.0%} | {expected_winner[cat]} |"
        )

    lines.append("\n## Average Context Size (lower = more precise, less noise)\n")
    lines.append("| Category | Vector-only | Graph-only | Hybrid |")
    lines.append("|---|---|---|---|")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)
        v_size = sum(r["vector_context_size"] for r in cat_results) / n
        g_size = sum(r["graph_context_size"] for r in cat_results) / n
        h_size = sum(r["hybrid_context_size"] for r in cat_results) / n
        lines.append(f"| {cat} | {v_size:.1f} items | {g_size:.1f} items | {h_size:.1f} items |")

    lines.append("\n## Per-Question Results\n")
    lines.append("| ID | Category | Question | Vector | Graph | Hybrid | Expected Route | Routed To | Router OK? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        v = "✅" if r["vector_correct"] else "❌"
        g = "✅" if r["graph_correct"] else "❌"
        h = "✅" if r["hybrid_correct"] else "❌"
        router_ok = "✅" if r["router_correct"] else "❌"
        lines.append(
            f"| {r['id']} | {r['category']} | {r['question']} | {v} | {g} | {h} | "
            f"{r['expected_route']} | {r['routed_mode']} | {router_ok} |"
        )

    router_correct = sum(r["router_correct"] for r in results)
    lines.append(f"\n**Router accuracy: {router_correct}/15**\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved markdown report to {path}")


if __name__ == "__main__":
    results = run_evaluation()
    print_per_question_table(results)
    print_category_summary_table(results)
    print_router_summary(results)
    check_validation_criteria(results)
    save_markdown_report(results)

"""
learning_curve.py

Turns the performance log into a chart: error rate over the course of the
conversation, in windows of 5 interactions (1-5, 6-10, 11-15, 16-20). If the
agent is genuinely learning from corrections, this line should trend
downward that visual trend IS the "proof" this project is about.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt

from performance_tracker import PerformanceTracker


def compute_windowed_error_rates(tracker: PerformanceTracker, window_size: int = 5,
                                  total_interactions: int = 20) -> List[Tuple[str, float]]:
    """Returns [("1-5", 1.0), ("6-10", 0.33), ...] -- error rate per window."""
    results = []
    for start in range(1, total_interactions + 1, window_size):
        end = min(start + window_size - 1, total_interactions)
        rate = tracker.get_error_rate(start, end)
        results.append((f"{start}-{end}", rate))
    return results


def plot_learning_curve(tracker: PerformanceTracker, save_path: str = "learning_curve.png",
                         window_size: int = 5, total_interactions: int = 20):

    windows = compute_windowed_error_rates(tracker, window_size, total_interactions)
    labels = [w[0] for w in windows]
    rates = [w[1] * 100 for w in windows]  # as percentages

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(labels, rates, marker="o", linewidth=2, color="#d62728")
    ax.fill_between(range(len(labels)), rates, alpha=0.1, color="#d62728")

    ax.set_title("Agent Error Rate Over Time")
    ax.set_xlabel("Interaction window")
    ax.set_ylabel("Error rate (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    for i, rate in enumerate(rates):
        ax.annotate(f"{rate:.0f}%", (i, rate), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    tracker = PerformanceTracker(db_path="performance.db")
    windows = compute_windowed_error_rates(tracker)
    print("Error rate by interaction window:")
    for label, rate in windows:
        print(f"  {label}: {rate:.0%}")
    plot_learning_curve(tracker)
    print("\nSaved chart to learning_curve.png")
    tracker.close()

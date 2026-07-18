"""
Statistical Analyzer — takes raw per-question scores from every
configuration and produces defensible conclusions: not just "which mean
is highest" but "is that difference actually significant, or noise?"
It Uses a PAIRED t-test (scipy.stats.ttest_rel) rather than an independent
t-test, because every configuration is evaluated on the exact same
questions in the exact same order - that pairing is real information a
plain average throws away, and using it gives a more sensitive test.
"""
import statistics
from scipy import stats

#takes a single configuration's raw results and produces a summary with mean, stddev, and 95% confidence interval for hit@k and any RAGAS metrics.
def summarize_config(result):
    scores = result["hit_at_k_scores"]
    n = len(scores)
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if n > 1 else 0.0

    # Standard error of the mean -> 95% CI (normal approximation)
    se = std / (n ** 0.5) if n > 1 else 0.0
    ci_margin = 1.96 * se

    summary = {
        "name": result["name"],
        "config": result["config"],
        "num_chunks": result["num_chunks"],
        "elapsed_seconds": result["elapsed_seconds"],
        "n_questions": n,
        "hit_at_k_mean": mean,
        "hit_at_k_std": std,
        "hit_at_k_ci95_low": max(0.0, mean - ci_margin),
        "hit_at_k_ci95_high": min(1.0, mean + ci_margin),
    }

    if result.get("ragas_scores"):
        for metric, values in result["ragas_scores"].items():
            if values:
                clean = [v for v in values if v is not None]
                if clean:
                    summary[f"{metric}_mean"] = statistics.mean(clean)
                    summary[f"{metric}_std"] = statistics.stdev(clean) if len(clean) > 1 else 0.0

    return summary


#this covers the case where two configurations have identical scores for every question, which can happen if the corpus is small or the questions are easy. In that case, the t-test returns NaN, but we want to treat it as a non-significant result (p=1.0).
#it compares the best configuration against every other configuration to see if the difference is statistically significant at p<0.05.
def paired_significance_test(best_result, other_result):

    a = best_result["hit_at_k_scores"]
    b = other_result["hit_at_k_scores"]

    if a == b:
        return 1.0, False

    try:
        t_stat, p_value = stats.ttest_rel(a, b)
        if p_value != p_value:  # NaN check
            return 1.0, False
        return float(p_value), bool(p_value < 0.05)
    except Exception:
        return 1.0, False


#main analysis function: takes all raw results, summarizes each configuration, sorts by hit@k mean, identifies the best configuration, and compares every other configuration against it for statistical significance.
def analyze(results):

    summaries = [summarize_config(r) for r in results]
    summaries.sort(key=lambda s: -s["hit_at_k_mean"])

    best = next(r for r in results if r["name"] == summaries[0]["name"])

    comparisons = []
    for r in results:
        if r["name"] == best["name"]:
            continue
        p_value, significant = paired_significance_test(best, r)
        comparisons.append({
            "config_name": r["name"],
            "vs_best": best["name"],
            "p_value": round(p_value, 4),
            "significant_at_0.05": significant,
        })

    return {
        "leaderboard": summaries,
        "best_config": summaries[0]["name"],
        "significance_comparisons": comparisons,
    }
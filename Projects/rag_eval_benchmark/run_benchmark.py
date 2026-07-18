"""
RAG Evaluation Benchmark — main entry point.

Reads config.yaml, runs every listed configuration in parallel, analyzes
results statistically, and generates a publishable HTML report.

Run:
    python run_benchmark.py
    python run_benchmark.py --config my_other_config.yaml
"""
import argparse #this lib is used to parse command line arguments
import json

import yaml

from benchmark.eval_runner import run_benchmark
from benchmark.stats_analyzer import analyze
from benchmark.report_generator import generate_report


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Benchmark")
    parser.add_argument("--config", default="config.yaml",
                         help="Path to the config YAML file")
    parser.add_argument("--output", default="docs/index.html",
                         help="Path to write the HTML report")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    corpus_path = config["corpus"]
    with open(config["qa_pairs"]) as f:
        qa_pairs = json.load(f)
    max_workers = config.get("max_workers", 3)
    configurations = config["configurations"]

    print(f"Corpus: {corpus_path}")
    print(f"Test questions: {len(qa_pairs)}")
    print(f"Configurations to benchmark: {len(configurations)}")
    print(f"Names: {[c['name'] for c in configurations]}\n")

    results = run_benchmark(configurations, corpus_path, qa_pairs, max_workers=max_workers)

    if not results:
        print("No results produced - all configurations failed. Check errors above.")
        return

    print("\nRunning statistical analysis...")
    analysis = analyze(results)

    print(f"\nBest configuration: {analysis['best_config']} "
          f"({analysis['leaderboard'][0]['hit_at_k_mean']:.0%} accuracy)")

    with open("benchmark_results.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print("Saved raw analysis to benchmark_results.json")

    report_path = generate_report(analysis, output_path=args.output)
    print(f"Generated HTML report: {report_path}")


if __name__ == "__main__":
    main()
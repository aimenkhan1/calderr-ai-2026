"""
HTML Report Generator — turns the Statistical Analyzer's output into a
publishable static HTML page (bar chart + leaderboard table + significance
findings). No server needed - designed to be published via GitHub Pages.
"""
from datetime import datetime, timezone

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAG Evaluation Benchmark Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px;
         margin: 40px auto; padding: 0 20px; color: #1a1a2e; background: #fafafa; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 32px; }}
  .card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 24px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #eee; }}
  th {{ background: #f5f5f7; font-size: 0.85rem; text-transform: uppercase; color: #666; }}
  tr:hover {{ background: #fafafa; }}
  .best {{ background: #eafbf0; font-weight: 600; }}
  .sig-yes {{ color: #c0392b; font-weight: 600; }}
  .sig-no {{ color: #7f8c8d; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
           font-size: 0.75rem; font-weight: 600; }}
  .badge-best {{ background: #2ecc71; color: white; }}
  canvas {{ max-height: 320px; }}
  .findings {{ line-height: 1.6; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>RAG Evaluation Benchmark Report</h1>
<div class="subtitle">Generated {timestamp} · {n_configs} configurations · {n_questions} test questions per config</div>

<div class="card">
  <h2>Accuracy by Configuration</h2>
  <canvas id="accuracyChart"></canvas>
</div>

<div class="card">
  <h2>Leaderboard</h2>
  <table>
    <tr>
      <th>Configuration</th>
      <th>Chunk Size</th>
      <th>Embedding Model</th>
      <th>Retrieval Strategy</th>
      <th>k</th>
      <th>Hit@k Accuracy</th>
      <th>95% CI</th>
      <th>Chunks</th>
      <th>Time</th>
    </tr>
    {leaderboard_rows}
  </table>
</div>

<div class="card">
  <h2>Statistical Significance</h2>
  <p style="color:#666;font-size:0.9rem;">Paired t-test against the best-performing configuration
  (<code>{best_config}</code>), same test questions, same order. p &lt; 0.05 marked significant.</p>
  <table>
    <tr>
      <th>Configuration</th>
      <th>p-value</th>
      <th>Significant?</th>
    </tr>
    {significance_rows}
  </table>
</div>

<div class="card findings">
  <h2>Findings</h2>
  {findings_text}
</div>

<script>
const ctx = document.getElementById('accuracyChart');
new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: {chart_labels},
    datasets: [{{
      label: 'Hit@k Accuracy',
      data: {chart_values},
      backgroundColor: {chart_colors},
      borderRadius: 6,
    }}]
  }},
  options: {{
    scales: {{ y: {{ beginAtZero: true, max: 1, ticks: {{ callback: v => (v*100)+'%' }} }} }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});
</script>

</body>
</html>
"""


def _fmt_pct(x):
    return f"{x*100:.0f}%"


def generate_report(analysis, output_path="docs/index.html"):
    leaderboard = analysis["leaderboard"]
    best_name = analysis["best_config"]

    rows = []
    for s in leaderboard:
        is_best = s["name"] == best_name
        row_class = ' class="best"' if is_best else ""
        badge = ' <span class="badge badge-best">BEST</span>' if is_best else ""
        rows.append(
            f"<tr{row_class}><td>{s['name']}{badge}</td>"
            f"<td>{s['config']['chunk_size']}</td>"
            f"<td>{s['config'].get('embedding_model', 'default')}</td>"
            f"<td>{s['config'].get('retrieval_strategy', 'semantic')}</td>"
            f"<td>{s['config']['k']}</td>"
            f"<td>{_fmt_pct(s['hit_at_k_mean'])}</td>"
            f"<td>[{_fmt_pct(s['hit_at_k_ci95_low'])}, {_fmt_pct(s['hit_at_k_ci95_high'])}]</td>"
            f"<td>{s['num_chunks']}</td>"
            f"<td>{s['elapsed_seconds']}s</td></tr>"
        )
    leaderboard_rows = "\n".join(rows)

    sig_rows = []
    for c in analysis["significance_comparisons"]:
        sig_class = "sig-yes" if c["significant_at_0.05"] else "sig-no"
        sig_text = "Yes - real difference" if c["significant_at_0.05"] else "No - within noise"
        sig_rows.append(
            f"<tr><td>{c['config_name']}</td><td>{c['p_value']}</td>"
            f"<td class='{sig_class}'>{sig_text}</td></tr>"
        )
    significance_rows = "\n".join(sig_rows)

    chart_labels = [s["name"] for s in leaderboard]
    chart_values = [round(s["hit_at_k_mean"], 3) for s in leaderboard]
    chart_colors = ["#2ecc71" if s["name"] == best_name else "#95a5a6" for s in leaderboard]

    significant_configs = [c["config_name"] for c in analysis["significance_comparisons"]
                            if c["significant_at_0.05"]]
    not_significant = [c["config_name"] for c in analysis["significance_comparisons"]
                        if not c["significant_at_0.05"]]

    findings_lines = [
        f"<p><strong>{best_name}</strong> was the top-performing configuration, "
        f"with {_fmt_pct(leaderboard[0]['hit_at_k_mean'])} hit@k accuracy.</p>"
    ]
    if significant_configs:
        findings_lines.append(
            f"<p>Its advantage over <code>{', '.join(significant_configs)}</code> "
            f"is statistically significant (p &lt; 0.05) - not just noise from a small test set.</p>"
        )
    if not_significant:
        findings_lines.append(
            f"<p>Its advantage over <code>{', '.join(not_significant)}</code> did "
            f"<strong>not</strong> reach statistical significance - the difference could plausibly "
            f"be due to chance given the sample size, and either configuration would be a defensible choice.</p>"
        )
    findings_text = "\n".join(findings_lines)

    html = REPORT_TEMPLATE.format(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_configs=len(leaderboard),
        n_questions=leaderboard[0]["n_questions"] if leaderboard else "?",
        leaderboard_rows=leaderboard_rows,
        best_config=best_name,
        significance_rows=significance_rows,
        findings_text=findings_text,
        chart_labels=chart_labels,
        chart_values=chart_values,
        chart_colors=chart_colors,
    )

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
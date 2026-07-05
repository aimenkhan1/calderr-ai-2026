"""
visualizer.py - Chart Generator

Takes the text output from the code executor and
generates an appropriate chart using matplotlib.
Chart type is determined by the # CHART_TYPE comment
that Groq adds to the generated code.

Supported chart types:
    bar  - for comparisons between categories
    line - for trends over time
    pie  - for proportions of a whole
    none - no chart, just show the text result
"""

import os
import re
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, works in Streamlit
import matplotlib.pyplot as plt
from datetime import datetime


#chart styling

CHART_STYLE = {
    "figure.facecolor":  "#0f0f1a",
    "axes.facecolor":    "#1a1a2e",
    "axes.edgecolor":    "#333355",
    "axes.labelcolor":   "#c8c8e8",
    "xtick.color":       "#c8c8e8",
    "ytick.color":       "#c8c8e8",
    "text.color":        "#c8c8e8",
    "grid.color":        "#333355",
    "grid.alpha":        0.4,
}

ACCENT_COLORS = ["#7c6af7", "#a78bfa", "#6ee7b7", "#f472b6", "#fbbf24", "#60a5fa"]

def parse_output_to_dict(output: str) -> dict:
    """
    Parse pandas text output into a label:value dict for charting.
    Handles irregular spacing from pandas .to_string() output.
    """
    result = {}
    lines  = output.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip metadata and header lines
        if any(skip in line.lower() for skip in ["name:", "dtype:", "length", "index"]):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        # Last token should be the number, everything before is the label
        label     = " ".join(parts[:-1]).strip()
        value_str = parts[-1].replace(",", "").replace("$", "").strip()

        try:
            value = float(value_str)
            result[label] = value
        except ValueError:
            continue

    return result


def generate_chart(output: str, chart_type: str, question: str, output_dir: str, csv_path: str = None) -> str | None:
    """
    Generate a chart from executor output.
    Supports: bar, line, pie, histogram, scatter, heatmap, box
    """
    if chart_type == "none":
        return None

    plt.rcParams.update(CHART_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f1a")

    title = question[:60] + "..." if len(question) > 60 else question

    # For scatter, heatmap, box, histogram we need raw data
    # Parse output to dict for bar, line, pie
    data = parse_output_to_dict(output)

    if not data and chart_type in ["bar", "line", "pie"]:
        return None

    labels = list(data.keys())
    values = list(data.values())

    try:

        if chart_type == "bar":
            bars = ax.bar(labels, values, color=ACCENT_COLORS[:len(labels)], width=0.6)
            ax.bar_label(bars, fmt="%.0f", padding=4, color="#c8c8e8", fontsize=9)
            ax.set_xlabel("Category", labelpad=10)
            ax.set_ylabel("Value", labelpad=10)
            plt.xticks(rotation=30, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        elif chart_type == "line":
            ax.plot(labels, values, color=ACCENT_COLORS[0],
                    linewidth=2.5, marker="o", markersize=5)
            ax.fill_between(range(len(labels)), values,
                            alpha=0.15, color=ACCENT_COLORS[0])
            ax.set_xlabel("", labelpad=10)
            ax.set_ylabel("Value", labelpad=10)
            # Show every 5th label to avoid crowding
            step = max(1, len(labels) // 10)
            ax.set_xticks(range(0, len(labels), step))
            ax.set_xticklabels(labels[::step], rotation=30, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        elif chart_type == "scatter":
            if csv_path:
                import pandas as pd
                df = pd.read_csv(csv_path)

                # Find two numeric columns to plot
                # Try to guess from question keywords
                numeric_cols = df.select_dtypes(include="number").columns.tolist()

                x_col = None
                y_col = None

                q = question.lower()

                # Try to match column names from question
                for col in numeric_cols:
                    if col.lower() in q:
                        if x_col is None:
                            x_col = col
                        elif y_col is None:
                            y_col = col
                            break

                # Fallback to first two numeric columns
                if not x_col and len(numeric_cols) >= 2:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]

                if x_col and y_col:
                    ax.scatter(
                        df[x_col], df[y_col],
                        color=ACCENT_COLORS[0],
                        alpha=0.4,
                        s=20,
                        edgecolors="none"
                    )

                    # Add regression line
                    import numpy as np
                    z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
                    p = np.poly1d(z)
                    x_sorted = sorted(df[x_col].dropna())
                    ax.plot(x_sorted, p(x_sorted),
                            color=ACCENT_COLORS[3],
                            linewidth=2,
                            linestyle="--",
                            label="Trend line")
                    ax.legend(facecolor="#1a1a2e", labelcolor="#c8c8e8")
                    ax.set_xlabel(x_col, labelpad=10)
                    ax.set_ylabel(y_col, labelpad=10)
                    ax.grid(linestyle="--", alpha=0.3)
                else:
                    return None
            else:
                return None

        elif chart_type == "pie":
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                colors=ACCENT_COLORS[:len(labels)],
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.85
            )
            for text in autotexts:
                text.set_color("#c8c8e8")

        elif chart_type == "histogram":
            if csv_path:
                import pandas as pd
                df       = pd.read_csv(csv_path)
                num_cols = df.select_dtypes(include="number").columns.tolist()

                # Find which column the question is about
                q      = question.lower()
                target = None
                for col in num_cols:
                    if col.lower() in q:
                        target = col
                        break

                # Fallback to first numeric column
                if not target and num_cols:
                    target = num_cols[0]

                if target:
                    ax.hist(
                        df[target].dropna(),
                        bins=30,
                        color=ACCENT_COLORS[0],
                        edgecolor="#0f0f1a",
                        alpha=0.85
                    )
                    ax.set_xlabel(target, labelpad=10)
                    ax.set_ylabel("Frequency", labelpad=10)
                    ax.grid(axis="y", linestyle="--", alpha=0.3)
                else:
                    return None
            else:
                return None

        elif chart_type == "box":
            if csv_path:
                import pandas as pd
                df     = pd.read_csv(csv_path)
                num_df = df.select_dtypes(include="number")

                q      = question.lower()
                target = None

                for col in num_df.columns:
                    if col.lower() in q:
                        target = col
                        break

                if not target:
                    target = "charges" if "charges" in num_df.columns else num_df.columns[0]

                # Use full column data not just outlier rows
                data = num_df[target].dropna().values.tolist()

                ax.boxplot(
                    data,              # flat list, not nested
                    patch_artist=True,
                    boxprops=    dict(facecolor=ACCENT_COLORS[0], color="#c8c8e8"),
                    medianprops= dict(color="#f472b6", linewidth=2.5),
                    whiskerprops=dict(color="#c8c8e8", linewidth=1.5),
                    capprops=    dict(color="#c8c8e8", linewidth=1.5),
                    flierprops=  dict(
                        markerfacecolor=ACCENT_COLORS[3],
                        marker="o",
                        markersize=4,
                        alpha=0.5
                    )
                )
                # Set label manually to avoid FixedLocator mismatch
                ax.set_xticks([1])
                ax.set_xticklabels([target])
                ax.set_ylabel(target, labelpad=10)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
            else:
                return None

        elif chart_type == "heatmap":
            if csv_path:
                import pandas as pd
                import numpy as np
                df         = pd.read_csv(csv_path)
                numeric_df = df.select_dtypes(include="number")
                corr       = numeric_df.corr()

                # Make a copy before modifying to avoid read-only error
                matrix = corr.values.copy()
                cols   = list(corr.columns)

                im = ax.imshow(matrix, cmap="RdPu", aspect="auto",
                               vmin=-1, vmax=1)
                fig.colorbar(im, ax=ax)

                ax.set_xticks(range(len(cols)))
                ax.set_yticks(range(len(cols)))
                ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
                ax.set_yticklabels(cols, fontsize=9)

                for i in range(len(cols)):
                    for j in range(len(cols)):
                        ax.text(j, i, f"{matrix[i][j]:.2f}",
                                ha="center", va="center",
                                color="white", fontsize=8)
                ax.grid(False)
            else:
                return None

        ax.set_title(title, pad=16, fontsize=12,
                     color="#c8c8e8", fontweight="bold")
        plt.tight_layout()

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(output_dir, f"chart_{timestamp}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()

        return chart_path

    except Exception as e:
        plt.close()
        import streamlit as st
        st.error(f"VISUALIZER ERROR: {e}")
        return None
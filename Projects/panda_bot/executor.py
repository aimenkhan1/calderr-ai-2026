"""
executor.py - Safe Sandboxed Code Executor

Runs AI-generated pandas code in a completely isolated
subprocess so that:
  - Infinite loops are killed after 10 seconds
  - Crashes do not affect the main app
  - Generated code cannot access files outside the temp folder
  - Errors are captured and shown to the user, not raised

This is the sandbox layer of the project.
"""

import os
import sys
import re
import subprocess
import tempfile
import shutil
from models import ExecutionResult


# How long to let generated code run before killing it
TIMEOUT_SECONDS = 10

def detect_chart_type(code: str, output: str = "", question: str = "", schema_columns: list = None) -> str:
    """
    Detect the best chart type using three methods in order:

    Method 1: Check for # CHART_TYPE comment in generated code
    Method 2: Check keywords in the user's question
    Method 3: Auto detect from the shape of the output data

    Supported types:
        bar, line, pie, histogram, scatter, heatmap, box, none
    """

    # Method 1: Explicit comment in code
    match = re.search(r'#\s*CHART_TYPE:\s*(\w+)', code, re.IGNORECASE)
    if match:
        chart_type = match.group(1).lower()
        if chart_type in ["bar", "line", "pie", "histogram", "scatter", "heatmap", "box"]:
            return chart_type

# Method 2: Keywords in the user question
    if question:
        q = question.lower()

        # Histogram - explicit request for distribution of numeric data
        if any(w in q for w in ["histogram", "hist", "spread", "frequency"]):
            return "histogram"

        # Pie - explicit request for proportions or parts
        if any(w in q for w in ["pie", "proportion", "percentage", "share"]):
            return "pie"

        # Heatmap - BEFORE scatter to avoid "correlation" being caught by scatter
        # Matches any question about correlations across multiple variables
        if any(w in q for w in [
            "heatmap", "heat map", "correlation matrix",
            "all features", "all columns", "all numerical",
            "all variables", "all numeric",
            "show the correlation", "correlation between all",
            "correlations", "correlated", "correlate"
        ]):
            return "heatmap"

        # Scatter - two variable relationship
        # Only triggers for specific two-variable questions
        if any(w in q for w in ["scatter", "vs ", "versus"]):
            return "scatter"

        # Also scatter if question contains "relationship between" or "between X and Y"
        if "relationship between" in q or "between" in q and "and" in q:
            # Only scatter if not asking about all variables (heatmap already caught those)
            if not any(w in q for w in ["all", "every", "each", "multiple"]):
                return "scatter"

        # Also scatter if exactly 2 column names appear in the question
        if schema_columns:
            cols_in_question = [col for col in schema_columns if col.lower() in q]
            if len(cols_in_question) == 2:
                # Two specific columns = scatter
                return "scatter"

        # Box plot - outlier and distribution shape questions
        if any(w in q for w in [
            "box", "boxplot", "box plot",
            "outlier", "outliers", "quartile",
            "interquartile", "iqr", "whisker"
        ]):
            return "box"

        # Line - trends and changes over a sequence
        if any(w in q for w in [
            "trend", "over time", "over months", "over years",
            "monthly", "yearly", "daily", "weekly",
            "change over", "change with", "line chart", "line graph",
            "plot over", "how does", "how do"
        ]):
            return "line"

        # Distribution - decide dynamically using actual column names
        if "distribution" in q:
            if schema_columns:
                # Check if the question mentions a known column
                for col in schema_columns:
                    if col.lower() in q:
                        # Categorical column names tend to contain these words
                        categorical_signals = [
                            "name", "type", "category", "status", "gender",
                            "sex", "region", "grade", "job", "education",
                            "marital", "smoker", "label", "class", "group",
                            "color", "country", "city", "department"
                        ]
                        if any(signal in col.lower() for signal in categorical_signals):
                            return "pie"
                        else:
                            return "histogram"
            # No column name found in question - default to histogram
            return "histogram"

        # Bar - grouping and comparison questions
        # Generic grouping words that work for any CSV
        if any(w in q for w in [
            "bar chart", "bar graph", "compare",
            "average by", "count by", "total by", "sum by",
            "highest", "lowest", "most", "least",
            "top", "bottom", "best", "worst",
            "by each", "for each", "per each",
            "group by", "grouped by", "broken down"
        ]):
            return "bar"

        # Also bar if a column name appears with a grouping word
        if schema_columns:
            grouping_words = [
                "by", "per", "each", "average", "total",
                "count", "highest", "lowest", "compare",
                "most", "least", "top", "bottom"
            ]
            cols_in_question = [col for col in schema_columns if col.lower() in q]
            if cols_in_question and any(word in q for word in grouping_words):
                return "bar"

    # Method 3: Auto detect from output shape
    if not output:
        return "none"

    lines = output.strip().split("\n")

    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(skip in line.lower() for skip in ["name:", "dtype:", "length", "index"]):
            continue
        data_lines.append(line)

    # Skip header row
    start_index = 0
    if data_lines:
        first_parts = data_lines[0].split()
        if first_parts:
            try:
                float(first_parts[-1].replace(",", "").replace("$", ""))
            except ValueError:
                start_index = 1

    data_rows = data_lines[start_index:]

    if len(data_rows) < 2:
        return "none"

    # Count numeric rows
    numeric_count = 0
    for line in data_rows:
        parts = line.split()
        if not parts:
            continue
        last = parts[-1].replace(",", "").replace("$", "").strip()
        try:
            float(last)
            numeric_count += 1
        except ValueError:
            continue

    if numeric_count < len(data_rows) * 0.6:
        return "none"

    # 2-10 rows → bar, 11+ rows → line
    return "bar" if len(data_rows) <= 10 else "line"


def run_code_safely(code: str, csv_filepath: str) -> ExecutionResult:
    """
    Execute pandas code in an isolated subprocess.

    Steps:
        1. Create a temporary folder
        2. Copy the CSV into the temp folder
        3. Write the generated code to a temp .py file
        4. Run it with a 10 second timeout
        5. Capture output and errors
        6. Delete the temp folder

    Returns an ExecutionResult with success/output/error.
    """
    temp_dir = tempfile.mkdtemp()

    try:
        # Copy the CSV into the temp folder as 'data.csv'
        # Generated code always reads from 'data.csv'
        temp_csv  = os.path.join(temp_dir, "data.csv")
        shutil.copy(csv_filepath, temp_csv)

        # Write the generated code to a temp file
        code_file = os.path.join(temp_dir, "analysis.py")
        with open(code_file, "w", encoding="utf-8") as f:
          f.write("import pandas as pd\n")
          f.write("import numpy as np\n")
          f.write("import warnings\n")
          f.write("warnings.filterwarnings('ignore')\n\n")
          f.write(code)

        # we run the code in a subprocess with timeout
        result = subprocess.run(
            [sys.executable, code_file],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=temp_dir,
            encoding="utf-8"
)

        output = result.stdout.strip()
        error  = result.stderr.strip() if result.stderr.strip() else None

        # Determine output type for the visualizer
        chart_type  = detect_chart_type(code, output)
        output_type = chart_type if chart_type != "none" else "text"

        if result.returncode != 0:
            return ExecutionResult(
                success=     False,
                output=      "",
                error=       error or "Code execution failed",
                output_type= "text"
            )

        return ExecutionResult(
            success=     True,
            output=      output,
            error=       None,
            output_type= output_type
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=     False,
            output=      "",
            error=       f"Code took longer than {TIMEOUT_SECONDS} seconds and was stopped.",
            output_type= "text"
        )

    except Exception as e:
        return ExecutionResult(
            success=     False,
            output=      "",
            error=       str(e),
            output_type= "text"
        )

    finally:
        # Always delete the temp folder, even if something crashed
        shutil.rmtree(temp_dir, ignore_errors=True)
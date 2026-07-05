"""
analyzer.py - CSV Schema Analyzer

Reads an uploaded CSV file and extracts its structure:
column names, data types, row count, and a sample preview.
This schema is sent to Groq so it understands the data
before generating pandas code.
"""

import pandas as pd
from models import CSVSchema


def analyze_csv(filepath: str, filename: str) -> CSVSchema:

    df = pd.read_csv(filepath)

    # Convert dtype names to readable strings
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Take first 3 rows as a string sample
    sample = df.head(3).to_string(index=False)

    return CSVSchema(
        filename=    filename,
        rows=        len(df),
        columns=     list(df.columns),
        dtypes=      dtypes,
        sample_data= sample
    )


def build_schema_prompt(schema: CSVSchema) -> str:

    columns_list = "\n".join(
        f"  - {col} ({dtype})"
        for col, dtype in schema.dtypes.items()
    )

    return (
        f"File: {schema.filename} ({schema.rows} rows)\n"
        f"Columns:\n{columns_list}\n\n"
        f"Sample data (first 3 rows):\n{schema.sample_data}"
    )
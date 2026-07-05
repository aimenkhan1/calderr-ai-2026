"""
code_generator.py - Groq-Powered Pandas Code Generator

Sends the CSV schema and user question to Groq.
Groq returns Python pandas code that answers the question.
The code is extracted and validated before execution.
"""

import os
import re
from groq import Groq
from dotenv import load_dotenv
from models import CSVSchema, GeneratedCode
from analyzer import build_schema_prompt

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "openai/gpt-oss-120b"


""" 
System Prompt
Tells Groq exactly what kind of code to generate.
Rules prevent dangerous code from being written.
"""

CODE_GENERATION_PROMPT = """
You are a Python pandas expert. The user has uploaded a CSV file
and asked a data analysis question.

Your job is to write clean, correct pandas code that answers the question.

Rules:
- Always read the CSV using: df = pd.read_csv('data.csv')
- Always import pandas as pd at the top
- Always import numpy as np at the top if you use np anywhere in the code
- Print the final result using print()
- When asked for highest or lowest, ALWAYS print ALL groups first so a chart can be made, then print the winner separately
- For numeric results: print a clear number or summary
- For table results: print df.to_string() or a grouped result
- Never use matplotlib or plt.show() - charts are handled separately
- Never use input(), never make network requests
- Never delete or modify any files
- For correlation analysis, always use df.select_dtypes(include='number').corr() not df.corr()
- When using np.fill_diagonal(), always call .copy() first. Example: arr = df.corr().values.copy(), then np.fill_diagonal(arr, 0)
- Keep the code short and focused on the question
- End with a comment: # CHART_TYPE: bar|line|pie|none

Return your response in this exact format:

CODE:
```python
your code here
```

EXPLANATION:
one sentence explaining what the code does
"""

#this function takes the schema and question, sends it to Groq, and returns the generated code and explanation
def generate_code(schema: CSVSchema, question: str) -> GeneratedCode:

    schema_description = build_schema_prompt(schema)

    user_message = (
        f"CSV Data:\n{schema_description}\n\n"
        f"Question: {question}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CODE_GENERATION_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.1    # low temperature for accurate code generation
    )

    raw_output = response.choices[0].message.content or ""

    # Extract code block
    code_match = re.search(r'```python\s*(.*?)```', raw_output, re.DOTALL)
    code        = code_match.group(1).strip() if code_match else raw_output.strip()

    # Extract explanation
    explanation_match = re.search(r'EXPLANATION:\s*(.*?)$', raw_output, re.DOTALL)
    explanation       = explanation_match.group(1).strip() if explanation_match else "Code generated."

    # Clean any special unicode characters Groq might add
    # Windows cannot handle fancy characters like curly quotes or special hyphens
    code = code.encode("ascii", errors="ignore").decode("ascii")

    return GeneratedCode(
        question=    question,
        code=        code,
        explanation= explanation
    )

#this function checks if the question is actually about the uploaded data, returns True if valid, False if off-topic
def is_valid_data_question(question: str, schema: CSVSchema) -> bool:

    schema_description = build_schema_prompt(schema)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict validator. "
                    "The user has uploaded a CSV file with specific columns. "
                    "Your job is to decide if the user's question is actually "
                    "asking something about that data. "
                    "Reply with only one word: YES or NO. "
                    "YES means the question can be answered using the CSV columns. "
                    "NO means the question is off-topic, random, or unrelated to the data."
                )
            },
            {
                "role": "user",
                "content": (
                    f"CSV columns available:\n{schema_description}\n\n"
                    f"User question: {question}\n\n"
                    f"Is this question answerable using the CSV data? Reply YES or NO only."
                )
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content or ""
    return "YES" in answer.upper()

def generate_narrative(question: str, result: str, schema: CSVSchema) -> str:

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data analyst. Explain the analysis result "
                    "in 2-3 clear, friendly sentences. No bullet points. "
                    "Just plain conversational English."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Data result: {result}\n"
                    f"Dataset: {schema.filename} with {schema.rows} rows"
                )
            }
        ],
        temperature=0.5
    )

    return response.choices[0].message.content or "Analysis complete."
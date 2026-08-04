"""
Lab 2.1 - Structured Output Extractor

Uses LangChain's with_structured_output() and Pydantic to
extract structured information from unstructured job postings
into validated Python objects.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from models import JobPosting


load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


structured_llm = llm.with_structured_output(JobPosting)


job_postings = [

"""
We're hiring a Senior Python Developer at TechCorp.

Salary range: $90,000 to $130,000.

Must know Python, Django, PostgreSQL and AWS.

Location: San Francisco.

This is a fully remote position.
""",

"""
AI Engineer wanted at DataFlow Inc.

Pay: $80,000-$110,000.

Required:
Python,
LangChain,
Machine Learning,
SQL.

Based in New York.

Hybrid work model.
""",

"""
Frontend Developer needed at WebStudio.

Compensation:
70000 to 95000 dollars.

Skills:
React,
TypeScript,
CSS,
Git.

Office location:
Austin, Texas.

Onsite role.
"""
]


print()

print("STRUCTURED OUTPUT EXTRACTOR")

print()


for i, posting in enumerate(job_postings, start=1):

    print("-----------------------------------------")

    print(f"JOB POSTING {i}")

    print()

    print("Original Text")

    print(posting)

    print()

    result = structured_llm.invoke(posting)

    print("Extracted Data")

    print()

    print("Title :", result.title)

    print("Company :", result.company)

    print("Salary Min :", result.salary_min)

    print("Salary Max :", result.salary_max)

    print("Skills :", result.required_skills)

    print("Location :", result.location)

    print("Remote :", result.remote_status.value)

    print()
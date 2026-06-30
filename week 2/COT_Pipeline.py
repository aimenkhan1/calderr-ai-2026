"""
Lab - Chain of Thought Prompting Pipeline

This program compares LLM answers WITH and WITHOUT Chain-of-Thought (CoT)
prompting on 10 math and logic problems.

It demonstrates how CoT improves reasoning by asking the model to
think step by step before giving a final answer.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 10 Math and Logic Problems with correct answers

problems = [
    {
        "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
        "answer": "11"
    },
    {
        "question": "A cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?",
        "answer": "9"
    },
    {
        "question": "Jean has 30 lollipops. Jean eats 2 of the lollipops. With the remaining lollipops, Jean wants to package 2 lollipops in one bag. How many bags can Jean fill?",
        "answer": "14"
    },
    {
        "question": "If a train travels 60 miles in 1.5 hours, what is its speed in miles per hour?",
        "answer": "40"
    },
    {
        "question": "Tom has 3 times as many apples as Jerry. Jerry has 4 apples. How many apples does Tom have?",
        "answer": "12"
    },
    {
        "question": "A store sells pens at $2 each. If you buy 5 pens and get a $3 discount, how much do you pay?",
        "answer": "7"
    },
    {
        "question": "Sarah is twice as old as her brother. Her brother is 8 years old. How old will Sarah be in 5 years?",
        "answer": "21"
    },
    {
        "question": "A box contains 24 chocolates. If they are shared equally among 6 children, how many chocolates does each child get?",
        "answer": "4"
    },
    {
        "question": "If a is heads up and gets flipped 3 times, is it still heads up?",
        "answer": "no"
    },
    {
        "question": "John reads 15 pages per day. How many pages will he read in 2 weeks?",
        "answer": "210"
    }
]


def ask_standard(question):
    """Standard prompting - no chain of thought"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer the question directly with just the final answer."},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content


def ask_with_cot(question):
    """Chain of thought prompting - think step by step"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Think step by step to solve this problem. Show your reasoning, then give the final answer."},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content


# Run the comparison

print("\nCHAIN OF THOUGHT VS STANDARD PROMPTING\n")

standard_correct = 0
cot_correct = 0

for i, problem in enumerate(problems, 1):

    question = problem["question"]
    expected = problem["answer"]

    print(f"\nProblem {i}: {question}")
    print(f"Expected Answer: {expected}")

    # Standard prompting
    standard_answer = ask_standard(question)
    print(f"\nStandard Prompting Answer:\n{standard_answer}")

    # CoT prompting
    cot_answer = ask_with_cot(question)
    print(f"\nCoT Prompting Answer:\n{cot_answer}")

    # Check if correct (simple check if expected answer appears in response)
    if expected.lower() in standard_answer.lower():
        standard_correct += 1

    if expected.lower() in cot_answer.lower():
        cot_correct += 1

    print()


# Final Results

print("\nFINAL RESULTS\n")
print(f"Standard Prompting Correct: {standard_correct}/10")
print(f"CoT Prompting Correct: {cot_correct}/10")
print(f"Improvement: {cot_correct - standard_correct} more correct with CoT")


"""
Findings:

Standard prompting often gives direct answers without showing reasoning,
which can lead to incorrect answers especially on multi-step problems.

Chain of thought prompting breaks the problem into intermediate steps,
which significantly improves accuracy on problems that require multiple
calculations or logical steps.

The improvement is most noticeable on harder multi-step problems,
while simple one-step problems show little to no difference.

This matches the findings from the Wei et al. (2022) paper -
CoT prompting helps most on complex reasoning tasks and is an
emergent ability that works best with larger language models.
"""
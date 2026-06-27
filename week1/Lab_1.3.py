"""

Lab 1.3 - Prompt Engineering A/B Test

This program compares five different system prompts for the same task:
summarizing a news article.

Each prompt uses a different prompting technique:
1. Basic Prompt
2. Detailed Instruction
3. Persona-Based Prompt
4. Chain-of-Thought Prompt
5. Few-Shot Prompt

The program measures:
- Summary quality
- Word count
- Token usage

This demonstrates how prompt engineering affects LLM output.
"""


import os
from dotenv import load_dotenv
from groq import Groq

# Load API Key
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------------
# News Article
# -----------------------------------

article = """
Apple reported record quarterly earnings.

Revenue reached $94.8 billion.

iPhone sales were $51.3 billion.

Services earned $23.2 billion.

Sales in India increased by 33%.

Mac and iPad sales decreased.

Apple announced a $110 billion share buyback.

Earnings were higher than analysts expected.
"""

# -----------------------------------
# Five Different System Prompts
# -----------------------------------

prompts = [

    "Summarize this article.",

    "Summarize this article in exactly 3 bullet points.",

    "You are a financial journalist. Focus only on business numbers and important facts.",

    """Think step by step.
First find the important points.
Then write a short summary.""",

    """Example:

Article:
Tesla sales increased by 10%.
Revenue reached $20 billion.

Summary:
Tesla reported strong growth with revenue reaching $20 billion.

Now summarize the given article."""
]

# -----------------------------------
# Run Every Prompt
# -----------------------------------

count = 1

for prompt in prompts:

    print("\n==============================")
    print("Prompt", count)
    print("==============================")

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": article
            }
        ],

        temperature=0
    )

    answer = response.choices[0].message.content

    tokens = response.usage.total_tokens

    words = len(answer.split())

    print("Summary:\n")
    print(answer)

    print("\nWords:", words)

    print("Tokens:", tokens)

    count = count + 1


"""
Findings :

Prompt 1 (Basic)
- Neutral summary
- Simple and accurate

Prompt 2 (Detailed Instruction)
- Most concise
- Easy to read because of bullet points

Prompt 3 (Persona-Based)
- Professional business tone
- Focused mainly on financial information

Prompt 4 (Chain-of-Thought)
- Most detailed
- Included the most important facts after reasoning

Prompt 5 (Few-Shot)
- Followed the example style
- Produced a well-structured summary

Overall Conclusion:
Different system prompts produce different outputs even when using the same model and article.
Prompt engineering changes the accuracy, conciseness, structure, and tone of the response.
"""
    
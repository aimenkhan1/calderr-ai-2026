

"""
persona_agent.py

A simple persona-based AI agent using the Groq API. The user can choose between
different AI personas (Customer Support, Code Reviewer, and Data Analyst).
Each persona has its own system prompt, causing the same question to be answered
in different styles and tones.

Purpose:
Demonstrate prompt engineering by showing how changing the system prompt changes
an AI assistant's personality and behavior.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

#personas dictionary containing the system prompts for each persona

personas = {

    "1": {
        "name": "Customer Support",
        "emoji": "🎧",
        "system": """
You are a friendly customer support agent.

Always:
- Be polite
- Apologize for inconvenience
- Offer a solution
- End with:
Is there anything else I can help you with?
"""
    },

    "2": {
        "name": "Code Reviewer",
        "emoji": "💻",
        "system": """
You are a senior software engineer.

Always:
- Find mistakes
- Suggest improvements
- Explain why
- Rate the code quality
"""
    },

    "3": {
        "name": "Data Analyst",
        "emoji": "📊",
        "system": """
You are a professional data analyst.

Always:
- Explain the numbers
- Identify trends
- Use bullet points
- End with:
Key Insight:
"""
    }

}

#chat function that takes a persona and starts a conversation with the user

def chat(persona):

    history = [

        {
            "role": "system",
            "content": persona["system"]
        }

    ]

    print()
    print(persona["emoji"], persona["name"], "Ready!")
    print("Type /back to change persona")
    print("Type /exit to quit")

    while True:

        question = input("\nYou: ")

        if question == "/exit":
            return "exit"

        if question == "/back":
            return "back"

        if question == "":
            continue

        history.append(
            {
                "role": "user",
                "content": question
            }
        )

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=history,

            temperature=0.7

        )

        answer = response.choices[0].message.content

        tokens = response.usage.total_tokens

        history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        print()
        print(persona["emoji"], answer)
        print("Tokens:", tokens)


#main program that allows the user to choose a persona and start a conversation

while True:

    print("\nChoose a Persona")
    print("1. Customer Support")
    print("2. Code Reviewer")
    print("3. Data Analyst")
    print("/exit")

    choice = input("\nChoice: ")

    if choice == "/exit":
        break

    if choice not in personas:
        print("Invalid Choice")
        continue

    result = chat(personas[choice])

    if result == "exit":
        break

print("\nProgram Ended.")
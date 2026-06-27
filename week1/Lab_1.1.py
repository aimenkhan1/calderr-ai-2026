"""
Lab_1.1 : chatbot_memory.py

A basic AI chatbot using the Groq API with conversation memory.
It stores previous messages to enable multi-turn conversations and 
supports /clear and /exit commands.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Conversation history — this is how we maintain memory!
history = [
    {"role": "system", "content": "You are a helpful AI engineering assistant."}
]

print("Groq Chatbot Ready!")
print("Commands: /clear to reset history | /exit to quit")
print("-" * 50)

while True:
    # Get user input
    user_input = input("\nYou: ").strip()

    # Handle commands
    if user_input == "/exit":
        print("Thankyou..exiting the chatbot. Goodbye!")
        break

    if user_input == "/clear":
        history = [
            {"role": "system", "content": "You are a helpful AI engineering assistant."}
        ]
        print("Conversation history cleared!")
        continue

    if not user_input:
        continue

    # Add user message to history
    history.append({"role": "user", "content": user_input})

    # Send to Groq
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=history,
        temperature=0.7
    )

    # Get AI reply
    ai_reply = response.choices[0].message.content
    tokens = response.usage.total_tokens

    # Add AI reply to history
    history.append({"role": "assistant", "content": ai_reply})

    print(f"\nAI: {ai_reply}")
    print(f"Tokens used: {tokens}")